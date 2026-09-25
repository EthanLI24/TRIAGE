#!/usr/bin/env python3
"""Make rollout routing-replay indices valid PyTorch gather/scatter indices.

SGLang serializes routed expert ids as int32 to keep the rollout payload small.
Megatron's routing replay later uses those ids as gather/scatter indices, which
PyTorch requires to be int64.  Preserve the compact pinned-CPU representation
and cast only while transferring a replay entry back to the GPU.
"""

from pathlib import Path
import os


TARGET = (
    Path(os.environ.get("SLIME_ROOT", Path(__file__).resolve().parents[2] / "external" / "slime"))
    / "slime/utils/routing_replay.py"
)
OLD = "return top_indices.to(torch.cuda.current_device())"
NEW = "return top_indices.to(torch.cuda.current_device(), dtype=torch.long)"
EXPECTED_REPLACEMENTS = 2  # pop_forward and pop_backward


def main() -> None:
    source = TARGET.read_text()
    old_count = source.count(OLD)
    new_count = source.count(NEW)

    if old_count == 0 and new_count == EXPECTED_REPLACEMENTS:
        print(f"R3_INDEX_FIX: already applied to {TARGET}")
        return
    if old_count != EXPECTED_REPLACEMENTS or new_count != 0:
        raise RuntimeError(
            f"R3_INDEX_FIX: unexpected source in {TARGET}: "
            f"old_count={old_count}, new_count={new_count}"
        )

    TARGET.write_text(source.replace(OLD, NEW))
    patched = TARGET.read_text()
    if patched.count(NEW) != EXPECTED_REPLACEMENTS or OLD in patched:
        raise RuntimeError(f"R3_INDEX_FIX: verification failed for {TARGET}")
    print(f"R3_INDEX_FIX: cast replay indices to int64 on GPU transfer ({TARGET})")


if __name__ == "__main__":
    main()
