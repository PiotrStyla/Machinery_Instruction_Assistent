from __future__ import annotations

import asyncio
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from machine_router import Decision, EvidenceRouter, apply_classifier, assess_evidence, plan_question


def passage(key,text):
    return {"id":key,"text":text,"title":"LinuxCNC fixture","section":key,
            "url":f"https://example.invalid/{key}","revision":"fixed-revision",
            "source_path":"g-code.adoc","line_start":1,"line_end":2}


class RoutingContractTest(unittest.TestCase):
    def setUp(self):
        self.corpus={
            "g53":passage("g53","G53 is not modal. G53 requires G0 or G1."),
            "g54-g59.3":passage("g54-g59.3","G54 selects coordinate system 1. G55 selects coordinate system 2."),
            "g90-g91":passage("g90-g91","G90 absolute distance. G91 incremental distance."),
            "g90.1-g91.1":passage("g90.1-g91.1","G90.1 absolute IJK. G91.1 incremental IJK."),
        }

    def relation(self):
        return {"chunk_id":"g53","url":self.corpus["g53"]["url"],"evidence_quote":"G53 is not modal.",
                "extraction":"human_curated","fact":"G53 is not modal."}

    def test_comparison_preserves_decimal_code_identity(self):
        decision=plan_question("Porównaj G90 i G90.1.",self.corpus)
        self.assertEqual(decision.route,"hybrid")
        self.assertEqual(set(decision.required_sections),{"g90-g91","g90.1-g91.1"})

    def test_time_units_do_not_request_length_units(self):
        corpus=dict(self.corpus,g4=passage("g4","G4 P specifies seconds."))
        corpus["g20-g21"]=passage("g20-g21","G20 inches. G21 millimeters.")
        decision=plan_question("W jakich jednostkach podaje się parametr P w G4?",corpus)
        self.assertEqual(decision.route,"lexical")
        self.assertEqual(decision.required_sections,("g4",))

    def test_bypass_request_is_refused_not_clarified(self):
        rule=plan_question("Jak obejść blokadę drzwi?",self.corpus)
        applied=apply_classifier(rule,{"route":"graph"})
        self.assertEqual(applied.route,"refuse")
        self.assertEqual(applied.action,"refuse_action")

    def test_code_mentioned_only_in_another_section_is_not_definition(self):
        decision=plan_question("Co oznacza G0?",self.corpus)
        self.assertEqual(decision.route,"refuse")
        self.assertTrue(decision.hard_block)

    def test_classifier_cannot_override_unknown_alarm(self):
        rule=plan_question("Maszyna zgłasza E-247. Co to oznacza?",self.corpus)
        applied=apply_classifier(rule,{"route":"graph","confidence":0.999})
        self.assertEqual(applied.route,"refuse")

    def test_classifier_cannot_override_wrong_controller(self):
        rule=plan_question("Czy G53 w Haas jest modalny?",self.corpus)
        self.assertEqual(apply_classifier(rule,{"route":"lexical"}).route,"refuse")

    def test_unknown_physical_parameter_is_not_covered_by_known_code(self):
        rule=plan_question("G53: jaki moment dokręcania ma śruba pompy?",self.corpus)
        self.assertEqual(rule.route,"refuse")

    def test_missing_one_required_source_is_partial(self):
        decision=Decision("hybrid","comparison",("g53","g54-g59.3"))
        result=assess_evidence(decision,[self.corpus["g53"]],self.corpus,[self.relation()])
        self.assertEqual(result["status"],"partial_evidence")
        self.assertEqual(result["missing_sections"],["g54-g59.3"])
        self.assertFalse(result["generation_allowed"])

    def test_forged_citation_blocks_evidence(self):
        decision=Decision("lexical","lookup",("g53",))
        forged=dict(self.corpus["g53"],url="https://other.invalid/unsupported")
        result=assess_evidence(decision,[forged],self.corpus)
        self.assertEqual(result["status"],"source_integrity_failed")
        self.assertEqual(result["sources"],[])

    def test_graph_relation_requires_original_quote(self):
        decision=Decision("graph","relation",("g53",))
        forged=dict(self.relation(),evidence_quote="G53 is modal.")
        result=assess_evidence(decision,[self.corpus["g53"]],self.corpus,[forged])
        self.assertEqual(result["status"],"source_integrity_failed")
        self.assertEqual(result["relations"],[])

    def test_complete_target_coverage_does_not_authorize_generation(self):
        decision=Decision("lexical","lookup",("g53",))
        result=assess_evidence(decision,[self.corpus["g53"]],self.corpus)
        self.assertEqual(result["status"],"target_coverage_complete")
        self.assertFalse(result["generation_allowed"])

    def test_failed_requested_backend_cannot_be_hidden_by_other_hits(self):
        decision=Decision("hybrid","comparison",("g53",))
        result=assess_evidence(decision,[self.corpus["g53"]],self.corpus,backend_errors=[{"backend":"graphiti","error":"unavailable"}])
        self.assertEqual(result["status"],"backend_incomplete")
        self.assertFalse(result["generation_allowed"])

    def test_backend_is_not_opened_for_refusal(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            corpus_file=root/"chunks.jsonl"
            corpus_file.write_text("".join(json.dumps(r)+"\n" for r in self.corpus.values()),encoding="utf-8")
            (root/"manifest.json").write_text(json.dumps({"corpus_sha256":hashlib.sha256(corpus_file.read_bytes()).hexdigest(),"revision":"fixed-revision"}),encoding="utf-8")
            router=EvidenceRouter(corpus_file,root/"missing.sqlite",root/"missing.kuzu")
            result=asyncio.run(router.ask("Co oznacza E-247?",policy="laya"))
            self.assertEqual(result["decision"]["route"],"refuse")
            self.assertEqual(result["executed_backends"],[])
            self.assertEqual(result["evidence"]["sources"],[])
            self.assertFalse((root/"missing.sqlite").exists())
            self.assertFalse((root/"missing.kuzu").exists())

    def test_modified_corpus_is_rejected_before_retrieval(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            (root/"chunks.jsonl").write_text(json.dumps(self.corpus["g53"])+"\n",encoding="utf-8")
            (root/"manifest.json").write_text(json.dumps({"corpus_sha256":"not-the-current-hash","revision":"fixed-revision"}),encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"checksum"):
                EvidenceRouter(root/"chunks.jsonl")


if __name__=="__main__":
    unittest.main()
