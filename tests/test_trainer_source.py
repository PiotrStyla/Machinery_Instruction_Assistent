from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import slayer_runtime as runtime  # noqa: E402


class TrainerSourceTest(unittest.TestCase):
    """The checkpoint is only loadable by the class that produced it.

    Slayer packages ship train_gpt_ref_r23.py (1.00-1.06) or
    train_gpt_ref_r24_2.py (1.09+); picking the older file loads weights with a
    mismatched state dict, so the newest file in the package must win.
    """

    def test_prefers_newest_trainer_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp)
            (package / "train_gpt_ref_r23.py").write_text("# r23\n", encoding="utf-8")
            (package / "train_gpt_ref_r24_2.py").write_text("# r24_2\n", encoding="utf-8")
            self.assertEqual(
                Path(runtime._trainer_source(str(package))).name,
                "train_gpt_ref_r24_2.py",
            )

    def test_single_trainer_file_is_used_as_is(self):
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp)
            (package / "train_gpt_ref_r23.py").write_text("# r23\n", encoding="utf-8")
            self.assertEqual(
                Path(runtime._trainer_source(str(package))).name,
                "train_gpt_ref_r23.py",
            )

    def test_missing_trainer_file_names_the_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError) as raised:
                runtime._trainer_source(tmp)
            self.assertIn(Path(tmp).name, str(raised.exception))

    def test_downloaded_packages_resolve(self):
        for name in ("Slayer1B-1.05", "Slayer1B-1.06", "Slayer1B-1.09-Instruct"):
            package = ROOT / "models" / name
            if not package.exists():
                continue
            with self.subTest(package=name):
                expected = (
                    "train_gpt_ref_r24_2.py"
                    if name == "Slayer1B-1.09-Instruct"
                    else "train_gpt_ref_r23.py"
                )
                self.assertEqual(Path(runtime._trainer_source(str(package))).name, expected)


if __name__ == "__main__":
    unittest.main()
