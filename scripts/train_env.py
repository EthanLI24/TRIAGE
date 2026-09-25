#!/usr/bin/env python3
"""Serialize the actor environment without shell interpolation or credentials."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "policy_extension"))

BASE_KEYS = ("PYTHONPATH", "CUDA_DEVICE_MAX_CONNECTIONS", "SGLANG_MOE_RUNNER_BACKEND", "R3")
QUANT_KEYS = ("SHOULD_REPLACE_TE_GROUPLINEAR", "QAT_PARAMS", "DISABLE_BACKWARD_QUANT",
              "SGLANG_NVFP4_PERTOKEN_SCALE", "NVTE_NVFP4_DISABLE_RHT",
              "NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING", "NVTE_NVFP4_DISABLE_2D_QUANTIZATION")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("bf16", "nvfp4", "triage"))
    parser.add_argument("--recenter-enabled", action="store_true")
    args = parser.parse_args()
    values = {key: os.environ[key] for key in BASE_KEYS if key in os.environ}
    if args.mode != "bf16":
        values.update({key: os.environ[key] for key in QUANT_KEYS if key in os.environ})
    else:
        values["SHOULD_REPLACE_TE_GROUPLINEAR"] = "0"
    if args.mode == "triage":
        import triage_config
        unknown = sorted(key for key in os.environ if key.startswith("TRIAGE_") and key not in triage_config.ENV_VALUES)
        if unknown:
            parser.error(f"unknown TRIAGE settings: {unknown}; use TRIAGE_GATE_* and TRIAGE_REPAIR_* names (see docs/PAPER_AND_CODE.md)")
        legacy = sorted(key for key in os.environ if key.startswith("GP95_"))
        if legacy:
            parser.error(f"rename legacy GP95 settings to TRIAGE names: {legacy}")
        if args.recenter_enabled:
            return 0 if triage_config.RECENTER_ENABLE else 1
        values.update(triage_config.environment())
    print(json.dumps(values, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
