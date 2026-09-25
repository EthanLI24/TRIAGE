"""Behavioral checks for launch modes; no Slime install or GPU is required."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


def launch(mode, **overrides):
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("TRIAGE_", "GP95_", "NVTE_", "SGLANG_"))}
    env.update(DRY_RUN="1", PATH=str(Path(sys.executable).parent) + os.pathsep + env["PATH"])
    env.update(overrides)
    return subprocess.run(["bash", str(ROOT / f"examples/train_{mode}.sh")], env=env, text=True, capture_output=True)


def command(result):
    if result.returncode:
        raise AssertionError(result.stderr)
    return shlex.split(result.stdout.strip().split("RESOLVED_TRAIN_COMMAND:", 1)[1])


class TrainingLauncherTest(unittest.TestCase):
    def test_three_modes_use_distinct_precision_and_objectives(self):
        for mode in ("bf16", "nvfp4", "triage"):
            with self.subTest(mode=mode):
                cmd = command(launch(mode))
                env = json.loads(cmd[cmd.index("--train-env-vars") + 1])
                self.assertEqual("--fp4-recipe" in cmd, mode != "bf16")
                self.assertEqual("--sglang-quantization" in cmd, mode != "bf16")
                self.assertEqual("--custom-loss-function-path" in cmd, mode == "triage")
                self.assertEqual("--use-tis" in cmd, mode == "triage")
                self.assertNotIn("--use-rollout-routing-replay", cmd)
                self.assertIn("--finetune", cmd)
                self.assertNotIn("--ckpt-step", cmd)
                if mode == "triage":
                    self.assertEqual(env["TRIAGE_REPAIR_COEF"], "0.015")
                    self.assertEqual(env["TRIAGE_REPAIR_SCOPE"], "whole")
                    self.assertEqual(env["TRIAGE_REPAIR_START_STEP"], "0")
                    self.assertEqual(env["TRIAGE_GATE_ENABLE"], "1")
                else:
                    self.assertFalse(any(k.startswith("TRIAGE_") for k in env))
                if mode == "bf16":
                    self.assertFalse(any(k.startswith(("NVTE_NVFP4", "SGLANG_NVFP4")) for k in env))
                    self.assertEqual(env["SHOULD_REPLACE_TE_GROUPLINEAR"], "0")

    def test_overrides_and_paths_with_spaces_survive_serialization(self):
        result = launch("triage", HF_CHECKPOINT="/models/model with spaces", LOAD_DIR="/models/resume path",
                        TRIAGE_REPAIR_COEF="0.02", TRIAGE_REPAIR_SCOPE="adv_positive",
                        TRIAGE_RECENTER_ENABLE="0", CKPT_STEP="299", USE_TIS="0")
        cmd = command(result)
        self.assertEqual(cmd[cmd.index("--hf-checkpoint") + 1], "/models/model with spaces")
        self.assertEqual(cmd[cmd.index("--load") + 1], "/models/resume path")
        self.assertNotIn("--finetune", cmd)
        self.assertNotIn("--custom-reward-post-process-path", cmd)
        self.assertNotIn("--use-tis", cmd)
        env = json.loads(cmd[cmd.index("--train-env-vars") + 1])
        self.assertEqual(env["TRIAGE_REPAIR_COEF"], "0.02")
        self.assertEqual(env["TRIAGE_REPAIR_SCOPE"], "adv_positive")

    def test_baselines_clear_inherited_algorithm_and_fp4_settings(self):
        for mode in ("bf16", "nvfp4"):
            cmd = command(launch(mode, TRIAGE_REPAIR_ENABLE="1", TRIAGE_REPAIR_COEF="999",
                                 SGLANG_NVFP4_PERTOKEN_SCALE="1", NVTE_NVFP4_DISABLE_RHT="1"))
            env = json.loads(cmd[cmd.index("--train-env-vars") + 1])
            self.assertFalse(any(k.startswith("TRIAGE_") for k in env))
            self.assertNotIn("--custom-loss-function-path", cmd)
            if mode == "bf16":
                self.assertNotIn("SGLANG_NVFP4_PERTOKEN_SCALE", env)

    def test_30b_uses_expert_parallelism_and_optional_routing(self):
        cmd = command(launch("triage", MODEL="qwen3-30B-A3B"))
        self.assertEqual(cmd[cmd.index("--expert-model-parallel-size") + 1], "8")
        self.assertIn("--use-rollout-routing-replay", cmd)
        self.assertNotIn("--use-rollout-routing-replay", command(launch("triage", MODEL="qwen3-30B-A3B", R3="0")))

    def test_invalid_combinations_fail_before_launch(self):
        for overrides in ({"GLOBAL_BATCH_SIZE": "255"}, {"R3": "1"}, {"CP": "2"},
                          {"TRIAGE_REPAIR_SCOPE": "adv_negative"}, {"TRIAGE_REPAIR_COEF": "nan"},
                          {"CKPT_STEP": "299"}, {"GP95_REPAIR_COEF": "0.03"},
                          {"TRIAGE_REPAIR_COEFF": "0.015"},
                          {"TRIAGE_ALIGN_COEF": "0.015"},
                          {"TRIAGE_SEG_APPLY_SCOPE": "q3"}):
            with self.subTest(overrides=overrides):
                self.assertNotEqual(launch("triage", **overrides).returncode, 0)


if __name__ == "__main__":
    unittest.main()
