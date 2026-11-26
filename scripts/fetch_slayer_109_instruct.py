"""Build models/Slayer1B-1.09-Instruct/ from pinned Hugging Face revisions.

Weights and the trainer code come from different repositories, and the package
must satisfy two loaders: this repo's runtime (which reads the newest
`train_gpt_ref_*.py` from the folder) and the upstream harness
(`slayer_kv_lm.model_from_package`, which imports `load_slayer` from the folder
and calls `load(path, device)`). The base loader satisfies neither — it is
`load(device, update)` and downloads `wagi_<update>.pt` — so this script installs
`scripts/load_slayer_109.py` as the package loader.

Usage:
    python scripts/fetch_slayer_109_instruct.py                 # pełna paczka (4,3 GB)
    python scripts/fetch_slayer_109_instruct.py --dry-run        # tylko plan i weryfikacja SHA
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

WEIGHTS_REPO = "SlayerLab/Slayer1B-1.09-Instruct-26.11"
WEIGHTS_REVISION = "c4170ff925ac2deb7de433173909b4b4c4b09306"
CODE_REPO = "SlayerLab/Slayer1B-1.09"
CODE_REVISION = "b5fb64ffbd2a6c8852ecb5af2be91d5dfb39e6f1"
CODE_FILES = ("train_gpt_ref_r24_2.py",)

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "models" / "Slayer1B-1.09-Instruct"
LOADER_TEMPLATE = ROOT / "scripts" / "load_slayer_109.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Zbuduj paczkę Slayer1B-1.09-Instruct z przypiętych rewizji")
    parser.add_argument("--dry-run", action="store_true", help="tylko wypisz rewizje i pliki")
    parser.add_argument("--package", default=str(PACKAGE))
    args = parser.parse_args()
    package = Path(args.package)

    plan = {
        "weights": {"repo": WEIGHTS_REPO, "revision": WEIGHTS_REVISION,
                    "files": ["config.json", "tokenizer.json", "model.safetensors"]},
        "code": {"repo": CODE_REPO, "revision": CODE_REVISION, "files": list(CODE_FILES)},
        "loader": str(LOADER_TEMPLATE.relative_to(ROOT)),
        "package": str(package),
    }
    print(json.dumps(plan, indent=1))
    if args.dry_run:
        return

    api = HfApi()
    for repo, revision in ((WEIGHTS_REPO, WEIGHTS_REVISION), (CODE_REPO, CODE_REVISION)):
        head = api.model_info(repo).sha
        if head != revision:
            print(f"uwaga: {repo} ma teraz {head}, przypięta rewizja {revision}")

    package.mkdir(parents=True, exist_ok=True)
    snapshot_download(WEIGHTS_REPO, revision=WEIGHTS_REVISION, local_dir=str(package),
                      allow_patterns=list(plan["weights"]["files"]))
    for name in CODE_FILES:
        shutil.copyfile(hf_hub_download(CODE_REPO, name, revision=CODE_REVISION), package / name)
    shutil.copyfile(LOADER_TEMPLATE, package / "load_slayer.py")

    names = plan["weights"]["files"] + list(CODE_FILES) + ["load_slayer.py"]
    provenance = {
        "package": str(package),
        "weights": {"repo": WEIGHTS_REPO, "revision": WEIGHTS_REVISION},
        "model_code": {"repo": CODE_REPO, "revision": CODE_REVISION, "files": list(CODE_FILES)},
        "loader": {"source": str(LOADER_TEMPLATE.relative_to(ROOT))},
        "files": {name: sha256(package / name) for name in names},
        "note": ("head.weight is tied to tok.weight and is not stored; load_slayer_109 ties it before a "
                 "strict load. Chat format: ChatML, <|im_start|>=1, <|im_end|>=2, BOS <|endoftext|>=0."),
    }
    (package / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print("gotowe:", {name: round((package / name).stat().st_size / 1e6, 1) for name in names})


if __name__ == "__main__":
    main()
