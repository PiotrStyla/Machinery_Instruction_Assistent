import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from grounded_slayer import generate_grounded, load_grounded_model
from sft_slayer import FrozenFP32Linear, inject_lora, upcast_frozen_blocks


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(3, 2, bias=False, dtype=torch.bfloat16)

    def forward(self, value):
        return self.proj(value)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([TinyBlock()])

    def forward(self, value):
        return self.blocks[0](value)


class GroundedSlayerTest(unittest.TestCase):
    def test_wrong_release_is_rejected_before_loading_missing_base_weights(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = json.dumps({"step": 13848, "n_layer": 20}).encode()
            (root / "config.json").write_bytes(config)
            adapter = root / "adapter.pt"
            torch.save({"base_config_sha256": hashlib.sha256(b'other release').hexdigest(),
                        "fp32_head": True, "fp32_blocks": [18, 19]}, adapter)
            with self.assertRaises(ValueError):
                load_grounded_model(root, adapter)

    def test_mismatched_numeric_policy_is_rejected_before_loading_weights(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = json.dumps({"step": 13848, "n_layer": 20}).encode()
            (root / "config.json").write_bytes(config)
            adapter = root / "adapter.pt"
            torch.save({"base_config_sha256": hashlib.sha256(config).hexdigest(),
                        "fp32_head": True, "fp32_blocks": [19]}, adapter)
            with self.assertRaises(ValueError):
                load_grounded_model(root, adapter)

    def test_no_sources_cannot_trigger_experimental_generation(self):
        with self.assertRaises(ValueError):
            generate_grounded(None, None, {}, "G53?", [])

    def test_precision_conversion_keeps_lora_trainable_and_base_frozen(self):
        model = TinyModel().requires_grad_(False)
        with torch.no_grad():
            model.blocks[0].proj.weight.fill_(0.5)
        inject_lora(model, 1, 1, ("proj",), (0,))
        with torch.no_grad():
            model.blocks[0].proj.lora_a.fill_(0.2)
        base_before = model.blocks[0].proj.base.weight.detach().float().clone()
        upcast_frozen_blocks(model, (0,))
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(trainable, lr=0.1)
        inputs = torch.ones(1, 3, dtype=torch.bfloat16)
        before = model(inputs).float().square().mean()
        before.backward()
        optimizer.step()
        after = model(inputs).float().square().mean()
        self.assertLess(after.item(), before.item())
        torch.testing.assert_close(model.blocks[0].proj.base.weight, base_before, rtol=0, atol=0)
        self.assertTrue(all("lora_" in name for name, p in model.named_parameters() if p.requires_grad))
        self.assertEqual(model(inputs).dtype, torch.bfloat16)

    def test_upcasting_cannot_silently_freeze_trainable_base_parameters(self):
        with self.assertRaises(ValueError):
            FrozenFP32Linear(nn.Linear(3, 2))


if __name__ == "__main__":
    unittest.main()
