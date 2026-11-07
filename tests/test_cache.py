from __future__ import annotations

import sys
import unittest
from types import SimpleNamespace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import slayer_runtime as runtime  # noqa: E402

MODEL_DIRS = [
    ROOT / "models" / "Slayer1B-1.05",
    ROOT / "models" / "Slayer1B-1.06",
]
MODEL_DIRS = [path for path in MODEL_DIRS if (path / "train_gpt_ref_r23.py").exists()]


@unittest.skipUnless(MODEL_DIRS, "download a Slayer release first")
class KVCacheParityTest(unittest.TestCase):
    def make_model(self, model_dir, n_layer=2):
        trainer = runtime._trainer_module(str(model_dir))
        cfg = SimpleNamespace(
            vocab=32,
            n_layer=n_layer,
            n_embd=16,
            n_head=2,
            block=16,
            norm="rmsnorm",
            norm_eps=1e-6,
            pos="rope",
            rope_theta=500000.0,
            ffn="swiglu",
            ffn_mult=2.667,
            value_residual=True,
            qk_norm=True,
            attn_bias=0,
            kv_heads=0,
            loop=1,
            mixer_layout=None,
            imha_t=0,
            emb_dim=0,
        )
        torch.manual_seed(7)
        return trainer.GPT(cfg.vocab, cfg.n_layer, cfg.n_embd, cfg.n_head, cfg.block, cfg).eval()

    def test_prefill_matches_reference(self):
        for model_dir in MODEL_DIRS:
            with self.subTest(release=model_dir.name):
                model = self.make_model(model_dir)
                ids = torch.randint(0, 32, (1, 5))
                expected = model(ids)[0][:, -1:]
                actual, _ = runtime.SlayerKVCache.prefill(model, ids)
                self.assertLess((expected - actual).abs().max().item(), 1e-6)

    def test_decode_matches_reference(self):
        for model_dir in MODEL_DIRS:
            with self.subTest(release=model_dir.name):
                model = self.make_model(model_dir)
                ids = torch.randint(0, 32, (1, 5))
                _, cache = runtime.SlayerKVCache.prefill(model, ids)
                next_id = torch.randint(0, 32, (1, 1))
                expected = model(torch.cat((ids, next_id), dim=1))[0][:, -1:]
                actual = cache.decode(next_id)
                self.assertLess((expected - actual).abs().max().item(), 1e-5)

    def test_generate_uses_prefill_logits(self):
        class FakeTokenizer:
            def encode(self, _text):
                return SimpleNamespace(ids=[1, 2])

            def decode(self, ids, skip_special_tokens=False):
                return list(ids)

        model = self.make_model(MODEL_DIRS[0])
        tokenizer = FakeTokenizer()
        ids = torch.tensor([[1, 2]], dtype=torch.long)
        prefill_logits, _ = runtime.SlayerKVCache.prefill(model, ids)
        expected_first = int(runtime.sample(prefill_logits).item())
        generated = runtime.generate(model, tokenizer, "prompt", 1)
        self.assertEqual(generated[:3], [1, 2, expected_first])

    def test_nonzero_lora_and_cpu_precision_match_reference_during_cached_decode(self):
        from sft_slayer import inject_lora, upcast_frozen_blocks, upcast_frozen_head

        for model_dir in MODEL_DIRS:
            with self.subTest(release=model_dir.name), torch.random.fork_rng(devices=[]):
                model = self.make_model(model_dir, n_layer=3).to(torch.bfloat16).requires_grad_(False)
                inject_lora(model, 2, 4, ("qkv", "proj", "down"), (1, 2))
                with torch.no_grad():
                    for name, parameter in model.named_parameters():
                        if "lora_b" in name:
                            parameter.normal_(mean=0.0, std=0.15)
                upcast_frozen_head(model)
                upcast_frozen_blocks(model, (1, 2))
                model.eval()
                ids = torch.tensor([[1, 2, 3]])
                with torch.inference_mode():
                    actual, cache = runtime.SlayerKVCache.prefill(model, ids)
                    expected = model(ids)[0][:, -1:]
                    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.002)
                    self.assertEqual(actual.argmax(-1).tolist(), expected.argmax(-1).tolist())
                    for token in (4, 5):
                        next_id = torch.tensor([[token]])
                        ids = torch.cat((ids, next_id), dim=1)
                        actual = cache.decode(next_id)
                        expected = model(ids)[0][:, -1:]
                        torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.002)
                        self.assertEqual(actual.argmax(-1).tolist(), expected.argmax(-1).tolist())

if __name__ == "__main__":
    unittest.main()
