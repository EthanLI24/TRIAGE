#!/usr/bin/env python3
"""Restore routed-experts return after sglang_router drops the request field.

The immutable image uses sglang_router 0.3.2.  Its native ``/generate`` proxy
deserializes requests into openai-protocol 1.0.0's ``GenerateRequest`` and then
serializes that typed object to the worker.  That Rust struct has no
``return_routed_experts`` member, so the flag sent by Slime is silently lost.

For an R3 server, ``enable_return_routed_experts`` is already a global server
opt-in.  Make the scheduler honor that opt-in when constructing each request,
even if the per-request bit was removed by the router.  Apply it with
``scripts/apply_patches.sh r3`` (run automatically for R3 training launches).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TARGET = (
    Path(os.environ.get("SGLANG_ROOT", REPO_ROOT / "external" / "sglang"))
    / "python/sglang/srt/managers/scheduler.py"
)

OLD = "                return_routed_experts=recv_req.return_routed_experts,\n"
NEW = """                # [INFIX-R3] sglang_router 0.3.2's typed /generate
                # schema drops return_routed_experts.  The server-level flag is
                # an explicit R3 opt-in, so use it as the authoritative fallback.
                return_routed_experts=(
                    recv_req.return_routed_experts
                    or self.server_args.enable_return_routed_experts
                ),
"""


def patch(target: Path) -> str:
    source = target.read_text()
    old_count = source.count(OLD)
    new_count = source.count(NEW)

    if old_count == 0 and new_count == 1:
        return "already applied"
    if old_count != 1 or new_count != 0:
        raise RuntimeError(
            f"R3_FORCE_RETURN_FIX: unexpected source in {target}: "
            f"old_count={old_count}, new_count={new_count}"
        )

    target.write_text(source.replace(OLD, NEW))
    patched = target.read_text()
    if patched.count(NEW) != 1 or OLD in patched:
        raise RuntimeError(f"R3_FORCE_RETURN_FIX: verification failed for {target}")
    return "applied"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("target", nargs="?", type=Path, default=DEFAULT_TARGET)
    args = parser.parse_args()
    status = patch(args.target)
    print(
        "R3_FORCE_RETURN_FIX: "
        f"{status}; server-level route return survives typed router ({args.target})"
    )


if __name__ == "__main__":
    main()
