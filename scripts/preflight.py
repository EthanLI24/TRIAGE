#!/usr/bin/env python3
"""Check a selected training mode, local assets, and pinned runtime interfaces."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git_head(path):
    result = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def source_state(path, old, new, *, expected_new_count=1, new_contains_old=False):
    source = path.read_text()
    old_count, new_count = source.count(old), source.count(new)
    if new_count == expected_new_count and (new_contains_old or old_count == 0):
        return "patched"
    if new_count == 0 and old_count > 0:
        return "stock"
    return f"unknown(old={old_count},new={new_count})"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("bf16", "nvfp4", "triage"), default=os.environ.get("TRAINING_MODE", "triage"))
    parser.add_argument("--model", choices=("qwen3-4B", "qwen3-30B-A3B"), default=os.environ.get("MODEL", "qwen3-4B"))
    parser.add_argument("--r3", choices=("0", "1"), default=os.environ.get("R3", "0"))
    for flag, env, default in (("slime", "SLIME_ROOT", "slime"), ("sglang", "SGLANG_ROOT", "sglang"),
                               ("megatron", "MEGATRON_ROOT", "Megatron-LM"), ("minite", "MINITE_ROOT", "miniTransformer")):
        parser.add_argument(f"--{flag}-root", type=Path, default=Path(os.environ.get(env, str(REPO_ROOT / "external" / default))))
    parser.add_argument("--strict-versions", action="store_true")
    parser.add_argument("--check-runtime-imports", action="store_true")
    parser.add_argument("--expected-r3-state", choices=("stock", "patched", "either"), default="patched")
    parser.add_argument("--rollout-batch-size", type=int, default=int(os.environ.get("ROLLOUT_BATCH_SIZE", "16")))
    parser.add_argument("--samples-per-prompt", type=int, default=int(os.environ.get("N_SAMPLES_PER_PROMPT", "16")))
    parser.add_argument("--global-batch-size", type=int, default=int(os.environ.get("GLOBAL_BATCH_SIZE", "256")))
    args = parser.parse_args()
    errors, warnings = [], []
    manifest = json.loads((REPO_ROOT / "compatibility.json").read_text())
    for name, expected in manifest["overlay_sha256"].items():
        path = REPO_ROOT / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            errors.append(f"overlay missing or hash mismatch: {name}")
    if min(args.rollout_batch_size, args.samples_per_prompt, args.global_batch_size) <= 0 or args.rollout_batch_size * args.samples_per_prompt != args.global_batch_size:
        errors.append("rollout_batch_size * samples_per_prompt must equal one positive global_batch_size")
    if args.r3 == "1" and args.model == "qwen3-4B":
        errors.append("dense Qwen3-4B does not support routing replay")
    roots = {"slime": args.slime_root, "sglang": args.sglang_root, "megatron_lm": args.megatron_root}
    if args.mode != "bf16" and os.environ.get("SHOULD_REPLACE_TE_GROUPLINEAR") == "1":
        roots["mini_transformer"] = args.minite_root
    for name, root in roots.items():
        if not root.is_dir():
            errors.append(f"missing runtime directory: {root}")
            continue
        actual, expected = git_head(root), manifest["reference_stack"][name]["commit"]
        if actual != expected:
            (errors if args.strict_versions else warnings).append(f"{name} revision {actual!r}; pinned patch reference {expected}")
    arguments = args.slime_root / "slime/utils/arguments.py"
    for path in (args.slime_root / "train.py", arguments):
        if not path.is_file():
            errors.append(f"missing runtime file: {path}")
    flags = ["train-env-vars"]
    if args.mode == "triage":
        flags += ["custom-loss-function-path", "rollout-data-postprocess-path", "custom-reward-post-process-path"]
    if args.r3 == "1":
        flags.append("use-rollout-routing-replay")
    if arguments.is_file():
        source = arguments.read_text()
        errors += [f"Slime is missing --{flag}" for flag in flags if flag not in source]

    # User-supplied assets are only required for an actual launch, not command rendering.
    for key in ("HF_CHECKPOINT", "INITIAL_CHECKPOINT", "PROMPT_DATA", "SAVE_DIR"):
        value = os.environ.get(key)
        if not value:
            errors.append(f"set {key}, or use one of examples/train_*.sh")
        elif key == "INITIAL_CHECKPOINT" and os.environ.get("LOAD_DIR"):
            continue
        elif key == "PROMPT_DATA" and not Path(value).is_file():
            errors.append(f"PROMPT_DATA must be an existing JSONL file: {value}")
        elif key not in ("PROMPT_DATA", "SAVE_DIR") and not Path(value).exists():
            errors.append(f"{key} does not exist: {value}")
    load = os.environ.get("LOAD_DIR")
    if load and not Path(load).is_dir():
        errors.append(f"LOAD_DIR is not a checkpoint directory: {load}")
    destination = os.environ.get("SAVE_DIR")
    if destination:
        for source in (load, os.environ.get("INITIAL_CHECKPOINT")):
            if source and Path(source).resolve() == Path(destination).resolve():
                errors.append("SAVE_DIR must differ from the input checkpoint directory")
    if os.environ.get("EVAL_DATA") and not Path(os.environ["EVAL_DATA"]).is_file():
        errors.append("EVAL_DATA must be an existing JSONL file")

    if args.mode == "triage":
        sys.path.insert(0, str(REPO_ROOT / "policy_extension"))
        try:
            import triage_config
            if triage_config.CALIBRATION_PATH:
                from triage_calibration import load_state
                load_state(triage_config.CALIBRATION_PATH, window_size=triage_config.GATE_SIZE)
        except (ImportError, ValueError, OSError) as exc:
            errors.append(f"TRIAGE configuration: {exc}")
        marker = "SLIME_TRIAGE_TOKENMEAN_CUSTOM_NORMALIZER_V1"
        for name in ("loss.py", "model.py"):
            path = args.slime_root / "slime/backends/megatron_utils" / name
            if not path.is_file() or marker not in path.read_text():
                errors.append(f"TRIAGE token-normalizer patch missing: {path}")
            elif name == "model.py" and '"triage_token_weight_units"' not in path.read_text():
                errors.append(f"TRIAGE sample field missing: {path}")

    patch_states = {}
    if args.r3 == "1":
        targets = (
            ("slime", "patch_routing_replay_indices", args.slime_root / "slime/utils/routing_replay.py", 2, False),
            ("sglang", "patch_sglang_r3_force_return", args.sglang_root / "python/sglang/srt/managers/scheduler.py", 1, False),
            ("sglang", "patch_sglang_r3_bypassed_topk", args.sglang_root / "python/sglang/srt/layers/moe/topk.py", 1, False),
            ("slime", "patch_slime_r3_route_validation", args.slime_root / "slime/rollout/sglang_rollout.py", 1, True),
        )
        for repo, name, path, count, contains in targets:
            if not path.is_file():
                errors.append(f"missing R3 source: {path}")
                continue
            patch = load_module(name, REPO_ROOT / "patches" / repo / f"{name}.py")
            state = source_state(path, patch.OLD, patch.NEW, expected_new_count=count, new_contains_old=contains)
            patch_states[name] = state
            if state.startswith("unknown") or (args.expected_r3_state != "either" and state != args.expected_r3_state):
                errors.append(f"{name} source is {state}; expected {args.expected_r3_state}")
    if args.check_runtime_imports:
        sys.path[:0] = [str(REPO_ROOT / "policy_extension"), str(args.slime_root), str(args.sglang_root / "python"), str(args.megatron_root), str(args.minite_root)]
        names = ["torch", "slime", "sglang", "megatron.core"]
        if args.mode == "triage":
            names.append("TRIAGE")
        for name in names:
            try:
                importlib.import_module(name)
            except Exception as exc:
                errors.append(f"cannot import {name}: {type(exc).__name__}: {exc}")
    for message in warnings:
        print(f"PREFLIGHT_WARNING: {message}")
    for message in errors:
        print(f"PREFLIGHT_ERROR: {message}", file=sys.stderr)
    if errors:
        return 2
    print(f"PREFLIGHT_OK mode={args.mode} model={args.model} r3={patch_states}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
