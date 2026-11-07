import sys
import unittest
from pathlib import Path

import torch
from tokenizers import Tokenizer, models, pre_tokenizers

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from sft_slayer import _example


class SupervisedSequenceTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = Tokenizer(models.WordLevel({
            "<eos>": 0, "<unk>": 1, "Użytkownik": 2, ":": 3,
            "Pytanie": 4, "Asystent": 5, "Tak": 6, ".": 7,
        }, unk_token="<unk>"))
        self.tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    def pack(self, budget):
        return _example(self.tokenizer, "Pytanie", "Tak.", budget, torch.device("cpu"))

    def test_first_answer_token_and_eos_are_supervised_not_prompt(self):
        inputs, labels = self.pack(64)
        self.assertEqual(labels[labels != -100].tolist(), [6, 7, 0])
        first_answer_target = (labels[0] != -100).nonzero()[0].item()
        self.assertTrue(torch.all(labels[0, :first_answer_target] == -100).item())
        self.assertEqual(inputs[0, first_answer_target + 1].item(), 6)

    def test_complete_example_fits_exactly_but_is_never_silently_truncated(self):
        inputs, labels = self.pack(64)
        complete_length = inputs.shape[1] + 1
        exact_inputs, exact_labels = self.pack(complete_length)
        self.assertTrue(torch.equal(exact_inputs, inputs))
        self.assertTrue(torch.equal(exact_labels, labels))
        with self.assertRaises(ValueError):
            self.pack(complete_length - 1)

    def test_empty_response_cannot_create_an_eos_only_training_record(self):
        with self.assertRaises(ValueError):
            _example(self.tokenizer, "Pytanie", " ", 64, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
