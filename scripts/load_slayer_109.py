"""Load this package's weights (model.safetensors + config.json) with a local path.

The released base loader (`SlayerLab/Slayer1B-1.09/load_slayer.py`) takes
`load(device, update)` and downloads `wagi_<update>.pt` from the checkpoints repo,
so it cannot read a packaged folder: `slayer_kv_lm.model_from_package(pkg, dev)`
raises `ValueError: Published updates: 5000 through 45000 in steps of 5000`.
This file keeps the signature the harness expects (`load(path, device)`) and
loads the local safetensors into the `r24_2` GPT class.

Tracked here as the source of truth; `scripts/fetch_slayer_109_instruct.py`
installs it as `models/Slayer1B-1.09-Instruct/load_slayer.py`, because
`slayer_kv_lm.model_from_package` imports `load_slayer` from inside the package.

Usage:
    import load_slayer
    model = load_slayer.load("path/to/Slayer1B-1.09-Instruct")   # fp32, eval, CPU
    logits, _ = model(token_ids)

`head.weight` is tied to `tok.weight` and is not stored in the safetensors; the
loader ties it before a strict state-dict load, so a shape or key mismatch stops
with an exception instead of silently loading a wrong model.
"""

import importlib.util
import json
import os
from types import SimpleNamespace

from safetensors.torch import load_file

TRAINER = "train_gpt_ref_r24_2.py"


def _trainer(path, name=TRAINER):
    spec = importlib.util.spec_from_file_location("slayer_trainer_r24_2", os.path.join(path, name))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load trainer source: {os.path.join(path, name)}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build(cfg, path, trainer_name=TRAINER):
    """Empty model with the architecture from config (no weights loaded)."""
    c = SimpleNamespace(**cfg)
    return _trainer(path, trainer_name).GPT(c.vocab, c.n_layer, c.n_embd, c.n_head, c.block, c)


def load(path, device="cpu", trainer_name=TRAINER):
    with open(os.path.join(path, "config.json"), encoding="utf-8") as handle:
        cfg = json.load(handle)
    model = build(cfg, path, trainer_name)
    state = load_file(os.path.join(path, "model.safetensors"))
    for destination, source in cfg.get("tied_weights", {}).items():
        state.setdefault(source, state[destination])
    if "head.weight" not in state and hasattr(model, "head") and "tok.weight" in state:
        state["head.weight"] = state["tok.weight"]
    model.load_state_dict(state, strict=True)
    if hasattr(model, "head") and hasattr(model, "tok"):
        model.head.weight = model.tok.weight
    return model.to(device).float().eval()
