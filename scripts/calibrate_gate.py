#!/usr/bin/env python3
"""Fit optional, frozen Gate thresholds from exported reference-window JSONL files."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "policy_extension"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--segment-size", type=int, default=64)
    parser.add_argument("--start-rollout", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--target-fpr", type=float, default=0.05)
    parser.add_argument("--quantile-deep", type=float, default=0.99)
    parser.add_argument("--quantile-full", type=float, default=0.999)
    parser.add_argument("--weight-min", type=float, default=0.1)
    args = parser.parse_args()
    from triage_calibration import calibrate
    paths = sorted({file.resolve() for path in args.input for file in (path.glob("*.jsonl") if path.is_dir() else [path])})
    if not paths:
        parser.error("no reference JSONL files found")
    def records():
        for path in paths:
            with path.open() as handle:
                for line in handle:
                    if line.strip():
                        yield json.loads(line)
    state = calibrate(records(), window_size=args.segment_size, start_rollout=args.start_rollout,
                      rounds=args.rounds, target_fpr=args.target_fpr, quantile_deep=args.quantile_deep,
                      quantile_full=args.quantile_full, weight_min=args.weight_min)
    state["source_sha256"] = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(state, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"Saved frozen Gate calibration: {args.output}")


if __name__ == "__main__":
    main()
