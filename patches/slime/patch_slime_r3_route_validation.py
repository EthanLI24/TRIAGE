#!/usr/bin/env python3
"""Fail fast if an R3 rollout omits or returns an unpopulated route cache.

The SGLang capturer allocates zero-filled buffers, so fixing only the router
field can produce shape-correct but all-zero routes if the MoE Top-K capture
hook is still bypassed.  Validate one token/layer (Top-K IDs must be distinct
and in range) before allowing rollout data into training.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


DEFAULT_TARGET = (
    Path(os.environ.get("SLIME_ROOT", Path(__file__).resolve().parents[2] / "external" / "slime"))
    / "slime/rollout/sglang_rollout.py"
)

OLD = """    if "routed_experts" in output["meta_info"]:
        sample.rollout_routed_experts = np.frombuffer(
            pybase64.b64decode(output["meta_info"]["routed_experts"].encode("ascii")),
            dtype=np.int32,
        ).reshape(
            len(sample.tokens) - 1,
            args.num_layers,
            args.moe_router_topk,
        )
"""

NEW = """    if args.use_rollout_routing_replay and "routed_experts" not in output["meta_info"]:
        raise ValueError(
            "SGLang response is missing routed_experts while "
            "use_rollout_routing_replay=True; check router field forwarding "
            "and --enable-return-routed-experts"
        )

    if "routed_experts" in output["meta_info"]:
        sample.rollout_routed_experts = np.frombuffer(
            pybase64.b64decode(output["meta_info"]["routed_experts"].encode("ascii")),
            dtype=np.int32,
        ).reshape(
            len(sample.tokens) - 1,
            args.num_layers,
            args.moe_router_topk,
        )
        # [INFIX-R3] A real Top-K row contains K distinct, in-range expert IDs.
        # This catches a zero-filled SGLang route cache immediately, before a
        # silently false routing replay can reach actor training.
        route_probe = sample.rollout_routed_experts[0, 0]
        if (
            np.unique(route_probe).size != args.moe_router_topk
            or route_probe.min() < 0
            or route_probe.max() >= args.num_experts
        ):
            raise ValueError(
                "Invalid rollout routed experts; SGLang route capture was not "
                f"populated: first_topk={route_probe.tolist()}"
            )
"""


def patch(target: Path) -> str:
    source = target.read_text()
    old_count = source.count(OLD)
    new_count = source.count(NEW)
    # NEW intentionally contains OLD as its decode prefix, so old_count stays
    # one after patching.  Detect the complete replacement first.
    if new_count == 1:
        return "already applied"
    if old_count != 1 or new_count != 0:
        raise RuntimeError(
            f"R3_ROUTE_VALIDATE: unexpected source in {target}: "
            f"old_count={old_count}, new_count={new_count}"
        )
    target.write_text(source.replace(OLD, NEW))
    if target.read_text().count(NEW) != 1:
        raise RuntimeError(f"R3_ROUTE_VALIDATE: verification failed for {target}")
    return "applied"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("target", nargs="?", type=Path, default=DEFAULT_TARGET)
    args = parser.parse_args()
    status = patch(args.target)
    print(f"R3_ROUTE_VALIDATE: {status}; fail fast on missing/empty route cache ({args.target})")


if __name__ == "__main__":
    main()
