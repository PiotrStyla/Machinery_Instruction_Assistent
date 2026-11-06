from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from machine_expert import build_index, search


class OfflineRetrievalTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.db = root / "index.sqlite"
        corpus = root / "chunks.jsonl"
        examples = [
            ("g20-g21", "G20 G21 Units", "G20 uses inches. G21 uses millimeters for length units."),
            ("g90-g91", "G90 G91 Distance Mode", "G90 absolute distance; G91 incremental distance."),
            ("g90.1-g91.1", "G90.1 G91.1 Arc Distance Mode", "G90.1 absolute I J K offsets; G91.1 incremental offsets."),
            ("g53", "G53 Machine Coordinates", "G53 is not modal. Error if G0 or G1 is not active."),
        ]
        rows = [dict(id=key, section=title, text=text, title="LinuxCNC", url="https://example.invalid/source",
                     revision="fixture", source_path="g-code.adoc", line_start=1, line_end=2)
                for key, title, text in examples]
        corpus.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        build_index(corpus, self.db)

    def test_exact_decimal_code_does_not_match_integer_mode(self):
        self.assertEqual([hit["id"] for hit in search(self.db, "Co oznacza G90.1?")], ["g90.1-g91.1"])
        self.assertEqual([hit["id"] for hit in search(self.db, "Co oznacza G90?")], ["g90-g91"])

    def test_unknown_alarm_with_sentence_punctuation_abstains(self):
        self.assertEqual(search(self.db, "Maszyna zgłasza alarm E-247. Co to oznacza?"), [])
        self.assertEqual(search(self.db, "G53 oraz E-247: jakie współrzędne?"), [])

    def test_wrong_controller_does_not_reuse_linuxcnc_code(self):
        self.assertEqual(search(self.db, "Czy G53 w Haas jest modalny?"), [])

    def test_related_codes_do_not_hide_primary_evidence(self):
        self.assertEqual([hit["id"] for hit in search(self.db, "Czy G53 wymaga G0 albo G1?")], ["g53"])

    def test_polish_vocabulary_retrieves_english_evidence(self):
        self.assertEqual(search(self.db, "Który kod wybiera milimetry?")[0]["id"], "g20-g21")

    def test_unrelated_question_has_no_evidence(self):
        self.assertEqual(search(self.db, "Jaki numer seryjny ma urządzenie?"), [])

    def test_missing_index_is_not_silently_created(self):
        missing = Path(self.temp.name) / "missing.sqlite"
        with self.assertRaises(FileNotFoundError):
            search(missing, "G20")
        self.assertFalse(missing.exists())


if __name__ == "__main__":
    unittest.main()
