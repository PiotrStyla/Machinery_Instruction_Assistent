import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from training_measurements import answer_scores, read_events


class TrainingMeasurementTest(unittest.TestCase):
    def test_containing_reference_and_citation_does_not_make_extra_claims_correct(self):
        reference = "G53 jest niemodalny. [g53]"
        scores = answer_scores(reference + " W rzeczywistości jest modalny.", reference, ["g53"])
        self.assertFalse(scores["exact_match"])
        self.assertTrue(scores["correct_citation_ids"])

    def test_invented_extra_citation_fails_citation_identity_check(self):
        scores = answer_scores("G53 jest niemodalny. [g53] [other]", "G53 jest niemodalny. [g53]", ["g53"])
        self.assertFalse(scores["correct_citation_ids"])

    def test_only_case_and_whitespace_normalization_are_allowed(self):
        reference = "G53 jest niemodalny. [g53]"
        self.assertTrue(answer_scores("  g53 JEST  niemodalny.\n[g53]  ", reference, ["g53"])["exact_match"])
        self.assertFalse(answer_scores("G53 jest modalny. [g53]", reference, ["g53"])["exact_match"])

    def test_incomplete_tail_is_tolerated_but_complete_corrupt_line_is_reported(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "metrics.jsonl"
            path.write_bytes(b'{"event":"step","global_step":1,"loss":2}\n{"event":')
            events, errors = read_events(path)
            self.assertEqual([row["global_step"] for row in events], [1])
            self.assertEqual(errors, [])
            path.write_bytes(b'{"event":"step","global_step":1,"loss":2}\ninvalid\n{"event":"experiment_end"}\n')
            events, errors = read_events(path)
            self.assertEqual(len(errors), 1)
            self.assertEqual([row["event"] for row in events], ["step", "experiment_end"])


if __name__ == "__main__":
    unittest.main()
