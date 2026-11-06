from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
HAS_GRAPHITI = importlib.util.find_spec("graphiti_core") is not None
if HAS_GRAPHITI:
    from graphiti_machine import ingest


@unittest.skipUnless(HAS_GRAPHITI, "requires isolated Graphiti environment")
class GraphProvenanceTest(unittest.TestCase):
    def test_unverified_quote_cannot_create_graph(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            corpus = root / "corpus.jsonl"
            facts = root / "facts.jsonl"
            db = root / "graph.kuzu"
            corpus.write_text(json.dumps({"id":"units", "text":"G21 selects millimeters."})+"\n", encoding="utf-8")
            facts.write_text(json.dumps({"id":"false-fact", "chunk_id":"units", "extraction":"human_curated",
                                         "evidence_quote":"G21 selects inches."})+"\n", encoding="utf-8")
            args = SimpleNamespace(db=str(db), corpus=str(corpus), facts=str(facts))
            with self.assertRaisesRegex(ValueError, "unverified source span"):
                asyncio.run(ingest(args))
            self.assertFalse(db.exists())

    def test_existing_graph_is_never_overwritten_by_import(self):
        with tempfile.TemporaryDirectory() as temp:
            db = Path(temp) / "existing.kuzu"
            db.write_bytes(b"existing evidence")
            with self.assertRaises(FileExistsError):
                asyncio.run(ingest(SimpleNamespace(db=str(db))))
            self.assertEqual(db.read_bytes(), b"existing evidence")


if __name__ == "__main__":
    unittest.main()
