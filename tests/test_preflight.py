"""Preflight selects only the interfaces and patches needed by the chosen mode."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class PreflightTest(unittest.TestCase):
    def test_baseline_does_not_require_triage_or_r3_patch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            slime, sglang, megatron = [root / name for name in ("slime", "sglang", "megatron")]
            for path in (slime / "slime/utils", sglang, megatron, root / "hf", root / "checkpoint"):
                path.mkdir(parents=True)
            (slime / "train.py").write_text("")
            (slime / "slime/utils/arguments.py").write_text("# train-env-vars\n")
            (root / "train.jsonl").write_text('{"prompt":"test","label":"0"}\n')
            env = {key: value for key, value in os.environ.items() if not key.startswith(("TRIAGE_", "GP95_"))}
            env.update(HF_CHECKPOINT=str(root / "hf"), INITIAL_CHECKPOINT=str(root / "checkpoint"),
                       PROMPT_DATA=str(root / "train.jsonl"), SAVE_DIR=str(root / "output"))
            base = [sys.executable, str(ROOT / "scripts/preflight.py"), "--model", "qwen3-4B", "--r3", "0",
                    "--slime-root", str(slime), "--sglang-root", str(sglang), "--megatron-root", str(megatron)]
            for mode in ("bf16", "nvfp4"):
                result = subprocess.run(base + ["--mode", mode], env=env, text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
            result = subprocess.run(base + ["--mode", "triage"], env=env, text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("token-normalizer patch missing", result.stderr)
            self.assertIn("custom-loss-function-path", result.stderr)
            (slime / "slime/utils/arguments.py").write_text(
                "# train-env-vars custom-loss-function-path rollout-data-postprocess-path custom-reward-post-process-path\n")
            patch_dir = slime / "slime/backends/megatron_utils"
            patch_dir.mkdir(parents=True)
            for name in ("loss.py", "model.py"):
                (patch_dir / name).write_text('# SLIME_TRIAGE_TOKENMEAN_CUSTOM_NORMALIZER_V1\n"triage_token_weight_units"\n')
            env["TRIAGE_REPAIR_COEF"] = "0.02"
            env["TRIAGE_REPAIR_SCOPE"] = "adv_positive"
            result = subprocess.run(base + ["--mode", "triage"], env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            env["SAVE_DIR"] = env["INITIAL_CHECKPOINT"]
            result = subprocess.run(base + ["--mode", "bf16"], env=env, text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("must differ", result.stderr)


if __name__ == "__main__":
    unittest.main()
