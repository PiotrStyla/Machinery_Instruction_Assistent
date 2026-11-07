"""Offline tool selection plus fail-closed evidence coverage; never machine control or generated advice."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from machine_expert import CODE_RE, GROUPS, NOTICE, normalize, read_rows, search

ROUTES = {"lexical", "graph", "hybrid", "refuse"}


@dataclass(frozen=True)
class Decision:
    route: str
    reason: str
    required_sections: tuple[str, ...] = ()
    codes: tuple[str, ...] = ()
    graph_anchors: tuple[str, ...] = ()
    hard_block: bool = False
    action: str = "retrieve"


def question_codes(question):
    return tuple(sorted({re.sub(r"\s+", "", code.upper()) for code in CODE_RE.findall(question)}))


def topic_sections(text):
    targets = set()
    if re.search(r"jednost.*dlugosc|dlugosc.*jednost|milimetr|\bcal[eai]?\b", text):
        targets.add("g20-g21")
    if "plaszczyz" in text:
        targets.add("g17-g19.1")
    if re.search(r"postoj|pauz|\bdwell\b", text):
        targets.add("g4")
    if re.search(r"odleglos.*osi|tryb.*przyrost|przyrost.*tryb|bezwzgled.*pozyc", text):
        targets.add("g90-g91")
    if ("luk" in text and re.search(r"srodk|przesun|przesuniec", text)) or re.search(r"\bi\s*,\s*j\s*(?:i|,)\s*k\b",text):
        targets.add("g90.1-g91.1")
    if "wspolrzed" in text and "maszyn" in text:
        targets.add("g53")
    if re.search(r"uklad.*robocz|robocz.*uklad", text):
        targets.add("g54-g59.3")
    return targets


def plan_question(question, corpus):
    text = normalize(question)
    codes = question_codes(question)
    if re.search(r"\b(?:haas|fanuc|siemens|heidenhain)\w*",text):
        return Decision("refuse","wrong_controller",codes=codes,hard_block=True,action="request_machine_documentation")
    if re.search(r"\b(?:obejdz|obejsc|omin|omij|wylacz)\w*.*(?:blokad|zabezpiecz|awaryj)|\b(?:steruj|uruchom|przesun)\w*.*(?:maszyn|wrzecion)",text):
        return Decision("refuse","unsafe_or_actuation_request",codes=codes,hard_block=True,action="refuse_action")
    if re.search(r"numer.*seryjn|moment.*dokrec|cisnien|hydraul|napraw|wymian.*(?:lozysk|pompy)|smarowan",text):
        return Decision("refuse","machine_specific_data_not_in_corpus",codes=codes,hard_block=True,action="request_machine_documentation")
    known = {code for chunk in corpus.values() for code in question_codes(chunk["text"])}
    if not set(codes) <= known:
        return Decision("refuse","unknown_code_or_alarm",codes=codes,hard_block=True,action="request_machine_documentation")
    primary = {key for key in corpus if set(codes).intersection(GROUPS.get(key, []))}
    # A mention inside another section is not a standalone definition.
    for code in codes:
        if not any(code in GROUPS.get(key,[]) for key in primary):
            if not primary or not any(code in question_codes(corpus[key]["text"]) for key in primary):
                return Decision("refuse","code_only_referenced_not_documented",codes=codes,hard_block=True,action="request_wider_documentation")
    topics = topic_sections(text)
    required = primary | topics
    relation = bool(re.search(r"powiaz|zalezn|wspoln|z jakimi|jakie (?:kody|polecenia).*kompens|polaczon",text))
    compensation = bool(re.search(r"kompens|korekc",text) and re.search(r"promien|frezu|narzedz",text))
    anchors = set(codes).intersection({c for values in GROUPS.values() for c in values})
    if compensation and (relation or not primary):
        anchors.add("cutter compensation")
        if not primary or re.search(r"z jakimi|jakie (?:kody|polecenia)|wspoln",text):
            required.update(("g53","g54-g59.3"))
        relation = True
    if not required:
        return Decision("refuse","no_identified_document_topic",codes=codes,hard_block=True,action="clarify_question")
    if not required <= corpus.keys():
        return Decision("refuse","requested_topic_missing_from_corpus",codes=codes,hard_block=True,action="request_wider_documentation")
    comparison = bool(re.search(r"porown|roznic|zestaw|a ktore",text))
    route = "hybrid" if comparison and len(required)>1 else "graph" if relation else "hybrid" if len(required)>1 else "lexical"
    return Decision(route,"multi_topic" if route=="hybrid" else "relationship_query" if route=="graph" else "scoped_lookup",
                    tuple(sorted(required)),codes,tuple(sorted(anchors)))


def apply_classifier(rule, proposal):
    label = proposal["route"]
    if label not in ROUTES:
        raise ValueError(f"invalid classifier route: {label}")
    if rule.hard_block:
        return rule
    return replace(rule,route=label,reason="local_classifier_proposal",
                   action="clarify_question" if label=="refuse" else "retrieve")


def assess_evidence(decision, hits, corpus, relations=(), backend_errors=()):
    valid = {}
    rejected = []
    fields = ("text","url","revision","source_path","line_start","line_end")
    for hit in hits:
        key = hit.get("id")
        expected = corpus.get(key)
        if expected is None or any(hit.get(field)!=expected.get(field) for field in fields):
            rejected.append({"id":key,"reason":"source_integrity_mismatch"})
        else:
            valid[key] = hit
    verified_relations = []
    for relation in relations:
        chunk = corpus.get(relation.get("chunk_id"))
        if (chunk is None or relation.get("url")!=chunk["url"] or
            relation.get("extraction")!="human_curated" or
            not relation.get("evidence_quote") or relation["evidence_quote"] not in chunk["text"]):
            rejected.append({"id":relation.get("chunk_id"),"reason":"graph_provenance_mismatch"})
        else:
            verified_relations.append(relation)
    missing = sorted(set(decision.required_sections)-valid.keys())
    if decision.route=="refuse":
        status="clarification_required" if decision.action=="clarify_question" else "refused"
    elif backend_errors:
        status="backend_incomplete"
    elif rejected:
        status="source_integrity_failed"
    elif not valid:
        status="no_evidence"
    elif missing:
        status="partial_evidence"
    elif not decision.required_sections:
        status="unscoped_evidence"
    elif decision.route in ("graph","hybrid") and not verified_relations:
        status="relationship_evidence_missing"
    else:
        status="target_coverage_complete"
    return {"status":status,"required_sections":list(decision.required_sections),"missing_sections":missing,
            "covered_sections":sorted(valid),"rejected":rejected,"relations":verified_relations,
            "sources":list(valid.values()),"generation_allowed":False,
            "scope":"Coverage of explicitly identified source topics only; not semantic entailment or machine compatibility.",
            "answer_policy":"source_excerpts_only"}


class EvidenceRouter:
    def __init__(self, corpus_path="data/machine_expert/chunks.jsonl", lexical_db="data/machine_expert/index.sqlite",
                 graph_db="data/machine_expert/graphiti_curated.kuzu"):
        path=Path(corpus_path)
        manifest=json.loads(path.with_name("manifest.json").read_text(encoding="utf-8"))
        if hashlib.sha256(path.read_bytes()).hexdigest()!=manifest["corpus_sha256"]:
            raise ValueError("corpus checksum does not match provenance manifest")
        self.corpus={r["id"]:r for r in read_rows(path)}
        if any(r["revision"]!=manifest["revision"] for r in self.corpus.values()):
            raise ValueError("mixed or unexpected documentation revisions")
        self.lexical_db=lexical_db
        self.graph_db=graph_db
        self.graph=None
        self.classifier=None

    def classify(self, question):
        if self.classifier is None:
            from routing_classifier import LayaRouteClassifier
            self.classifier=LayaRouteClassifier()
        return self.classifier.classify(question)

    def lexical_lookup(self, question, decision):
        # Decompose identified topics into actual code lookups, not benchmark-label lookups.
        queries=[question]+[GROUPS[key][0] for key in decision.required_sections]
        hits=[]
        for query in dict.fromkeys(queries):
            hits.extend(search(self.lexical_db,query,3))
        return hits,list(dict.fromkeys(queries))

    async def graph_lookup(self, question, decision):
        from graphiti_machine import GROUP, make_graph, load_fts
        if self.graph is None:
            if not Path(self.graph_db).exists():
                raise FileNotFoundError("Graphiti database missing; run graphiti_machine.py ingest in .venv-graphiti")
            self.graph=make_graph(self.graph_db,GROUP)
            await load_fts(self.graph)
        edges=await self.graph.search(query=question,group_ids=[GROUP],num_results=30)
        relations=[]
        selected_sections=set()
        for edge in edges:
            key=edge.attributes.get("chunk_id")
            if key in selected_sections or len(selected_sections)<3:
                selected_sections.add(key)
                relations.append(dict(edge.attributes,fact=edge.fact,relation=edge.name,origin="hybrid_search"))
        frontier=list(decision.graph_anchors)
        visited=set(frontier)
        for hop in range(1,3):
            if not frontier:
                break
            records,_,_=await self.graph.driver.execute_query(
                "MATCH (a:Entity)-[:RELATES_TO]->(e:RelatesToNode_)-[:RELATES_TO]->(b:Entity) "
                "WHERE e.group_id = $group_id AND (a.name IN $anchors OR b.name IN $anchors) "
                "RETURN a.name AS source,b.name AS target,e.name AS relation,e.fact AS fact,e.attributes AS attributes",
                group_id=GROUP,anchors=frontier)
            next_names=set()
            for record in records:
                meta=json.loads(record.pop("attributes"))
                relations.append(dict(meta,**record,origin="graph_traversal",hop=hop))
                next_names.update((record["source"],record["target"]))
            frontier=sorted(next_names-visited)
            visited.update(frontier)
        unique={}
        for relation in relations:
            unique[(relation.get("chunk_id"),relation["fact"])]=relation
        hits=[]
        for relation in unique.values():
            key=relation.get("chunk_id")
            if key in self.corpus:
                hits.append(dict(self.corpus[key],retriever="graphiti"))
        return hits,list(unique.values())

    async def ask(self, question, policy="rules", proposal=None):
        started=time.perf_counter()
        rule=plan_question(question,self.corpus)
        if policy=="laya":
            if rule.hard_block:
                decision=rule
            else:
                proposal=proposal if proposal is not None else self.classify(question)
                decision=apply_classifier(rule,proposal)
        elif policy=="rules":
            decision=rule
        else:
            raise ValueError("policy must be rules or laya")
        hits=[]
        relations=[]
        errors=[]
        executed=[]
        subqueries=[]
        if decision.route in ("lexical","hybrid"):
            executed.append("lexical")
            try:
                found,subqueries=self.lexical_lookup(question,decision)
                hits.extend(found)
            except Exception as exc:
                errors.append({"backend":"lexical","error":f"{type(exc).__name__}: {exc}"})
        if decision.route in ("graph","hybrid"):
            executed.append("graphiti")
            try:
                found,relations=await self.graph_lookup(question,decision)
                hits.extend(found)
            except Exception as exc:
                errors.append({"backend":"graphiti","error":f"{type(exc).__name__}: {exc}"})
        evidence=assess_evidence(decision,hits,self.corpus,relations,errors)
        return {"question":question,"policy":policy,"rule_decision":asdict(rule),"decision":asdict(decision),
                "classifier_proposal":proposal,"executed_backends":executed,"lexical_subqueries":subqueries,
                "backend_errors":errors,"evidence":evidence,"notice":NOTICE,
                "message":"Zwrócono wyłącznie źródła. Pokrycie tematów nie potwierdza poprawności wygenerowanej odpowiedzi ani zgodności z konkretną maszyną.",
                "elapsed_seconds":round(time.perf_counter()-started,3)}

    async def close(self):
        if self.graph is not None:
            from graphiti_machine import close_graph
            await close_graph(self.graph)
            self.graph=None


async def benchmark(args, router):
    rows=read_rows(args.cases)
    results=[]
    for row in rows:
        rule=plan_question(row["question"],router.corpus)
        started=time.perf_counter()
        # Rules are timed separately from retrieval and all model load costs.
        plan_question(row["question"],router.corpus)
        rule_ms=1000*(time.perf_counter()-started)
        result=dict(row,rule_decision=asdict(rule),rule_ms=round(rule_ms,4))
        proposal=None
        if args.classifier=="laya" or args.policy=="laya":
            try:
                proposal=router.classify(row["question"])
                result.update(classifier=proposal,applied_classifier_decision=asdict(apply_classifier(rule,proposal)))
            except Exception as exc:
                result["classifier_error"]=f"{type(exc).__name__}: {exc}"
        if args.execute:
            if args.policy=="laya" and proposal is None:
                result["execution_error"]="classifier unavailable; no silent fallback to rules"
            else:
                result["execution"]=await router.ask(row["question"],args.policy,proposal)
        results.append(result)
        print(f"{row['id']} rules={rule.route} classifier={proposal['route'] if proposal else 'not_available'}",flush=True)
    summary={"cases":len(rows),"rules_correct":sum(r["rule_decision"]["route"]==r["expected_route"] for r in results),
             "classifier_predictions":sum("classifier" in r for r in results),
             "classifier_errors":sum("classifier_error" in r for r in results),
             "classifier_raw_correct":sum(r.get("classifier",{}).get("route")==r["expected_route"] for r in results),
             "classifier_with_hard_guards_correct":sum(r.get("applied_classifier_decision",{}).get("route")==r["expected_route"] for r in results),
             "executed":sum("execution" in r for r in results)}
    if args.execute:
        summary["expected_source_coverage"]=sum(set(r["expected_ids"])<=set(r["execution"]["evidence"]["covered_sections"])
                                                for r in results if "execution" in r and r["expected_ids"])
        summary["source_bearing_cases"]=sum(bool(r["expected_ids"]) for r in rows)
    report={"summary":summary,"results":results,"classifier_load_seconds":router.classifier.load_seconds if router.classifier else None,
            "note":"Hand-authored development set. Routing labels are not industrial safety labels or answer correctness. Classifier confidence is not CNC-calibrated. No Slayer generation is enabled."}
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2))


async def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus",default="data/machine_expert/chunks.jsonl")
    parser.add_argument("--lexical-db",default="data/machine_expert/index.sqlite")
    parser.add_argument("--graph-db",default="data/machine_expert/graphiti_curated.kuzu")
    sub=parser.add_subparsers(dest="command",required=True)
    ask=sub.add_parser("ask")
    ask.add_argument("--question",required=True)
    ask.add_argument("--policy",choices=("rules","laya"),default="rules")
    ask.add_argument("--output")
    ask.add_argument("--model-dir", help="opt in to an experimental, unverified Slayer answer")
    ask.add_argument("--adapter")
    ask.add_argument("--max-input-tokens", type=int, default=1024)
    ask.add_argument("--max-new-tokens", type=int, default=64)
    ask.add_argument("--threads", type=int, default=4)
    bench=sub.add_parser("benchmark")
    bench.add_argument("--cases",default="data/machine_expert/router_cases.jsonl")
    bench.add_argument("--classifier",choices=("none","laya"),default="none")
    bench.add_argument("--policy",choices=("rules","laya"),default="rules")
    bench.add_argument("--execute",action="store_true")
    bench.add_argument("--output",default="reports/machine_expert_router.json")
    args=parser.parse_args()
    if args.command == "ask":
        if args.adapter and not args.model_dir:
            parser.error("--adapter requires --model-dir")
        if min(args.max_input_tokens, args.max_new_tokens, args.threads) < 1:
            parser.error("token budgets and threads must be positive")
    router=EvidenceRouter(args.corpus,args.lexical_db,args.graph_db)
    try:
        if args.command=="benchmark":
            await benchmark(args,router)
        else:
            report=await router.ask(args.question,args.policy)
            if args.model_dir:
                if report["evidence"]["status"] != "target_coverage_complete":
                    report["experimental_answer"] = {"status": "not_generated", "reason": report["evidence"]["status"]}
                else:
                    await router.close()
                    import torch
                    from grounded_slayer import generate_grounded, load_grounded_model
                    torch.set_num_threads(args.threads)
                    model, tokenizer, cfg = load_grounded_model(args.model_dir, args.adapter)
                    report["experimental_answer"] = generate_grounded(
                        model, tokenizer, cfg, args.question, report["evidence"]["sources"],
                        args.max_input_tokens, args.max_new_tokens)
                    report["experimental_answer"].update(model_dir=args.model_dir, adapter=args.adapter)
                    report["message"] = "Dodano eksperymentalną, niezweryfikowaną odpowiedź Slayera. Nie jest to zatwierdzona porada techniczna."
            if args.output:
                Path(args.output).parent.mkdir(parents=True,exist_ok=True)
                Path(args.output).write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
            print(json.dumps(report,ensure_ascii=False,indent=2))
    finally:
        await router.close()


if __name__=="__main__":
    asyncio.run(main())
