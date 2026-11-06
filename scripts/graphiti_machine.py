"""Real local Graphiti/Kuzu experiment; curated ingestion and model extraction are separate."""
from __future__ import annotations

# These clients disable telemetry and remote model downloads before Graphiti imports.
from local_graphiti_clients import LocalE5, LocalQwen, LocalCosineReranker, EMBED_MODEL, LLM_MODEL

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import re
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from graphiti_core import Graphiti
from graphiti_core.driver.kuzu_driver import KuzuDriver
from graphiti_core.edges import EntityEdge, EpisodicEdge
from graphiti_core.nodes import EntityNode, EpisodicNode, EpisodeType
from graphiti_core.utils.bulk_utils import add_nodes_and_edges_bulk
from machine_expert import read_rows, search as baseline_search, normalize, CODE_RE, NOTICE

CORPUS = "data/machine_expert/chunks.jsonl"
DEFAULT_DB = "data/machine_expert/graphiti_curated.kuzu"
GROUP = "linuxcnc_2_9_4_curated"
REFERENCE = datetime(2025, 1, 25, 12, 52, tzinfo=timezone.utc)


def uid(kind, value):
    return str(uuid5(NAMESPACE_URL, f"slayer-machine-expert:{kind}:{value}"))


def save_report(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


def make_graph(db_path, group):
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    # Bind the logical group through the public driver API. Kuzu 0.31.2
    # otherwise leaves _database unset on the add_episode path.
    driver = KuzuDriver(str(db_path)).with_database(group)
    embedding = LocalE5()
    llm = LocalQwen(log_path=f"reports/{group}_llm.jsonl")
    graph = Graphiti(graph_driver=driver, embedder=embedding, llm_client=llm,
                     cross_encoder=LocalCosineReranker(embedding), max_coroutines=1)
    return graph


async def load_fts(graph, install=False):
    if install:
        await graph.driver.execute_query("INSTALL FTS")
    await graph.driver.execute_query("LOAD FTS")


async def create_indexes(graph):
    # Graphiti 0.31.2 KuzuDriver.build_indices_and_constraints is a no-op;
    # the real Kuzu maintenance operations provide FTS index creation.
    await graph.driver.graph_ops.build_indices_and_constraints(graph.driver)


async def close_graph(graph):
    await graph.close()
    # Kuzu's upstream close() intentionally does nothing. Release Windows file handles.
    graph.driver.client.close()
    graph.driver.db.close()


async def ingest(args):
    if Path(args.db).exists():
        raise FileExistsError("graph database already exists; use a new --db path, never overwrite evidence")
    chunks = {r["id"]:r for r in read_rows(args.corpus)}
    facts = read_rows(args.facts)
    for fact in facts:
        if fact["extraction"] != "human_curated" or fact["evidence_quote"] not in chunks[fact["chunk_id"]]["text"]:
            raise ValueError(f"unverified source span for {fact['id']}")
    started = time.perf_counter()
    graph = make_graph(args.db, GROUP)
    try:
        await load_fts(graph, install=args.install_fts)
        nodes = {}
        episodes = {}
        now = datetime.now(timezone.utc)
        for key, chunk in chunks.items():
            episodes[key] = EpisodicNode(uuid=uid("episode", key), name=chunk["section"], group_id=GROUP,
                                          source=EpisodeType.text, content=chunk["text"], valid_at=REFERENCE,
                                          source_description=json.dumps({k:chunk[k] for k in ("id","url","revision","line_start","line_end")}),
                                          entity_edges=[])
        edges = []
        mentions = {}
        for fact in facts:
            for name in (fact["source"], fact["target"]):
                if name not in nodes:
                    nodes[name] = EntityNode(uuid=uid("entity", name), name=name, group_id=GROUP,
                                             summary=name, labels=["CncConcept"])
                episode = episodes[fact["chunk_id"]]
                key = (episode.uuid, nodes[name].uuid)
                mentions[key] = EpisodicEdge(uuid=uid("mention", ":".join(key)), group_id=GROUP,
                                             source_node_uuid=key[0], target_node_uuid=key[1], created_at=now)
            edge = EntityEdge(uuid=uid("fact",fact["id"]), group_id=GROUP,
                              source_node_uuid=nodes[fact["source"]].uuid, target_node_uuid=nodes[fact["target"]].uuid,
                              name=fact["relation"], fact=fact["fact"], created_at=now,
                              episodes=[episodes[fact["chunk_id"]].uuid], valid_at=REFERENCE,
                              reference_time=REFERENCE, attributes={key:fact[key] for key in
                               ("chunk_id","url","evidence_quote","extraction")})
            edges.append(edge)
            episodes[fact["chunk_id"]].entity_edges.append(edge.uuid)
        node_vectors = await graph.embedder.create_batch([node.name for node in nodes.values()])
        for node, vector in zip(nodes.values(), node_vectors, strict=True):
            node.name_embedding = vector
        edge_vectors = await graph.embedder.create_batch([edge.fact for edge in edges])
        for edge, vector in zip(edges, edge_vectors, strict=True):
            edge.fact_embedding = vector
        await add_nodes_and_edges_bulk(graph.driver, list(episodes.values()), list(mentions.values()),
                                       list(nodes.values()), edges, graph.embedder)
        # Kuzu FTS indexes snapshot the populated tables; create them after structured import.
        await create_indexes(graph)
        counts, _, _ = await graph.driver.execute_query("MATCH (e:RelatesToNode_) RETURN count(e) AS facts")
        report = {"graphiti_version": importlib.metadata.version("graphiti-core"), "backend":"kuzu",
                  "backend_warning":"Kuzu is deprecated upstream; experimental local store only",
                  "embedding_model":EMBED_MODEL,"embedding_snapshot":graph.embedder.path,
                  "llm_calls":graph.llm_client.calls,"extraction":"human_curated_not_model_discovered",
                  "corpus_sha256":hashlib.sha256(Path(args.corpus).read_bytes()).hexdigest(),
                  "facts_sha256":hashlib.sha256(Path(args.facts).read_bytes()).hexdigest(),
                  "sections":len(episodes),"entities":len(nodes),"facts":len(edges),"persisted":counts,
                  "elapsed_seconds":round(time.perf_counter()-started,3),"db":args.db}
        save_report(args.output,report)
        print(json.dumps(report,ensure_ascii=False,indent=2))
    finally:
        await close_graph(graph)


def scope_reject(question, chunks):
    if re.search(r"\b(haas|fanuc|siemens|heidenhain)\b",normalize(question)):
        return True
    requested={re.sub(r"\s+","",s.upper()) for s in CODE_RE.findall(question)}
    known={re.sub(r"\s+","",s.upper()) for c in chunks.values() for s in CODE_RE.findall(c["text"])}
    return not requested <= known


async def graph_search(graph, question, chunks, top_k=3):
    edges = await graph.search(query=question,group_ids=[GROUP],num_results=30)
    results=[]
    seen=set()
    for edge in edges:
        key=edge.attributes.get("chunk_id")
        if key not in chunks:
            raise ValueError(f"graph edge has no verified corpus provenance: {edge.uuid}")
        if key in seen:
            continue
        seen.add(key)
        results.append(dict(chunks[key],fact=edge.fact,graph_rank=len(results)+1,
                            relation=edge.name,extraction=edge.attributes["extraction"]))
        if len(results)>=top_k:
            break
    return results


async def compare(args):
    if not Path(args.db).exists():
        raise FileNotFoundError("ingest the Graphiti corpus first")
    chunks={r["id"]:r for r in read_rows(args.corpus)}
    rows=read_rows(args.questions)
    if args.relationships:
        rows+=read_rows(args.relationships)
    graph=make_graph(args.db,GROUP)
    try:
        await load_fts(graph)
        results=[]
        for row in rows:
            started=time.perf_counter()
            baseline=baseline_search(args.baseline_db,row["question"],3)
            baseline_ms=1000*(time.perf_counter()-started)
            started=time.perf_counter()
            raw=await graph_search(graph,row["question"],chunks)
            graph_ms=1000*(time.perf_counter()-started)
            guarded=[] if scope_reject(row["question"],chunks) else raw
            record=dict(row,baseline_ids=[r["id"] for r in baseline],graphiti_raw_ids=[r["id"] for r in raw],
                        graphiti_guarded_ids=[r["id"] for r in guarded],baseline_ms=round(baseline_ms,3),
                        graphiti_ms=round(graph_ms,3),graphiti_facts=[r["fact"] for r in raw])
            results.append(record)
            print(f"{row['id']} baseline={record['baseline_ids']} graphiti={record['graphiti_raw_ids']}",flush=True)
        summary={}
        for label,field in (("baseline","baseline_ids"),("graphiti_raw","graphiti_raw_ids"),("graphiti_guarded","graphiti_guarded_ids")):
            supported=[r for r in results if r["answerable"]]
            unknown=[r for r in results if not r["answerable"]]
            summary[label]={"supported":len(supported),"hit_at_1":sum(bool(set(r['expected_ids'])&set(r[field][:1])) for r in supported),
                            "hit_at_3":sum(bool(set(r['expected_ids'])&set(r[field])) for r in supported),
                            "all_expected_at_3":sum(set(r['expected_ids'])<=set(r[field]) for r in supported),
                            "unknown":len(unknown),"unknown_abstained":sum(not r[field] for r in unknown)}
        summary["median_baseline_ms"]=statistics.median(r["baseline_ms"] for r in results)
        summary["median_graphiti_ms"]=statistics.median(r["graphiti_ms"] for r in results)
        report={"summary":summary,"results":results,"graphiti_extraction":"human_curated",
                "note":"Curated graph and domain-tuned lexical baseline. Not autonomous extraction or independent industrial safety validation. Guarded Graphiti applies explicit controller/code scope rules; raw Graphiti does not."}
        save_report(args.output,report)
        print(json.dumps(summary,ensure_ascii=False,indent=2))
    finally:
        await close_graph(graph)


async def neighbors(args):
    if not Path(args.db).exists():
        raise FileNotFoundError(args.db)
    graph=make_graph(args.db,GROUP)
    try:
        rows,_,_=await graph.driver.execute_query("MATCH (n:Entity) WHERE n.group_id = $group_id AND n.name = $entity_name RETURN n.uuid AS uuid",group_id=GROUP,entity_name=args.entity)
        frontier=[r["uuid"] for r in rows]
        visited=set(frontier)
        found={}
        for hop in range(1,args.hops+1):
            if not frontier:
                break
            rows,_,_=await graph.driver.execute_query(
                "MATCH (a:Entity)-[:RELATES_TO]->(e:RelatesToNode_)-[:RELATES_TO]->(b:Entity) "
                "WHERE e.group_id = $group_id AND (a.uuid IN $frontier OR b.uuid IN $frontier) "
                "RETURN a.uuid AS a_uuid,a.name AS source,b.uuid AS b_uuid,b.name AS target,e.uuid AS uuid,e.fact AS fact,e.attributes AS attributes",
                group_id=GROUP,frontier=frontier)
            next_frontier=set()
            for row in rows:
                meta=json.loads(row["attributes"])
                found.setdefault(row["uuid"],dict(source=row["source"],target=row["target"],fact=row["fact"],hop=hop,**meta))
                next_frontier.update((row["a_uuid"],row["b_uuid"]))
            frontier=sorted(next_frontier-visited)
            visited.update(frontier)
        report={"entity":args.entity,"hops":args.hops,"relations":list(found.values()),"notice":NOTICE}
        if args.output:
            save_report(args.output,report)
        print(json.dumps(report,ensure_ascii=False,indent=2))
    finally:
        await close_graph(graph)


async def extract(args):
    if Path(args.db).exists():
        raise FileExistsError("automatic extraction uses a fresh separate --db")
    chunks={r["id"]:r for r in read_rows(args.corpus)}
    section=chunks[args.section]
    group="linuxcnc_2_9_4_auto"
    graph=make_graph(args.db,group)
    started=time.perf_counter()
    report={"section":args.section,"extraction":"local_qwen_automatic","llm_model":LLM_MODEL,"db":args.db,
            "source_url":section["url"],"revision":section["revision"]}
    try:
        await load_fts(graph,install=args.install_fts)
        await create_indexes(graph)
        result=await graph.add_episode(name=section["section"],episode_body=section["text"],source=EpisodeType.text,
                                       source_description=section["url"],reference_time=REFERENCE,group_id=group,
                                       update_communities=False)
        report.update(status="ingested_unverified",nodes=[n.model_dump(mode="json") for n in result.nodes],
                      edges=[e.model_dump(mode="json") for e in result.edges])
    except Exception as exc:
        report.update(status="extraction_failed",error=f"{type(exc).__name__}: {exc}")
    finally:
        report.update(llm_calls=graph.llm_client.calls,elapsed_seconds=round(time.perf_counter()-started,3))
        save_report(args.output,report)
        await close_graph(graph)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if report["status"]=="extraction_failed":
        raise SystemExit(1)


async def run():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    for name in ("ingest","compare","search","neighbors","extract"):
        command=sub.add_parser(name)
        command.add_argument("--db",default="data/machine_expert/graphiti_auto.kuzu" if name=="extract" else DEFAULT_DB)
        command.add_argument("--corpus",default=CORPUS)
        if name in ("ingest","extract"):
            command.add_argument("--install-fts",action="store_true",help="one-time Kuzu FTS extension download; setup only")
        if name=="ingest":
            command.add_argument("--facts",default="data/machine_expert/graph_facts.jsonl")
            command.add_argument("--output",default="reports/machine_expert_graphiti_ingest.json")
        elif name=="compare":
            command.add_argument("--questions",default="data/machine_expert/questions.jsonl")
            command.add_argument("--relationships")
            command.add_argument("--baseline-db",default="data/machine_expert/index.sqlite")
            command.add_argument("--output",default="reports/machine_expert_graphiti_compare.json")
        elif name=="search":
            command.add_argument("--question",required=True)
            command.add_argument("--guarded",action="store_true")
        elif name=="neighbors":
            command.add_argument("--entity",required=True)
            command.add_argument("--hops",type=int,choices=(1,2),default=2)
            command.add_argument("--output")
        else:
            command.add_argument("--section",default="g20-g21")
            command.add_argument("--output",default="reports/machine_expert_graphiti_auto.json")
    args=parser.parse_args()
    if args.command=="ingest":
        await ingest(args)
    elif args.command=="compare":
        await compare(args)
    elif args.command=="neighbors":
        await neighbors(args)
    elif args.command=="extract":
        await extract(args)
    else:
        if not Path(args.db).exists():
            raise FileNotFoundError(args.db)
        chunks={r["id"]:r for r in read_rows(args.corpus)}
        graph=make_graph(args.db,GROUP)
        try:
            await load_fts(graph)
            hits=[] if args.guarded and scope_reject(args.question,chunks) else await graph_search(graph,args.question,chunks)
            print(json.dumps({"hits":hits,"guarded":args.guarded,"notice":NOTICE},ensure_ascii=False,indent=2))
        finally:
            await close_graph(graph)


if __name__=="__main__":
    asyncio.run(run())
