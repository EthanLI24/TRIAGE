#!/usr/bin/env python3
"""Capture MoE routes when SGLang uses a fused, BYPASSED Top-K backend.

SGLang's ``flashinfer_trtllm`` / ``flashinfer_mxfp4`` path intentionally
returns ``BypassedTopKOutput`` so FlashInfer can fuse routing into its MoE
kernel.  In the pinned SGLang revision (see ``compatibility.json``), the
routed-experts capturer is only called by ``select_experts`` in the STANDARD
Top-K path.  Consequently ``--enable-return-routed-experts`` returns no
``routed_experts`` metadata for ModelOpt FP4 rollouts.

Keep the production FlashInfer output and MoE path unchanged.  Only when
route return is enabled, materialize the same Top-K through SGLang's existing
``select_experts`` as a side channel; that call writes the route capture cache,
and its return value is deliberately discarded.  Thus the patch costs one
extra Top-K per MoE layer only for R3 runs, without forcing native MoE or
changing the logits/hidden-state result consumed by FlashInfer.

The patch is intentionally pinned to exact source anchors and is idempotent.
It must fail closed if the pinned SGLang source changes.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TARGET = (
    Path(os.environ.get("SGLANG_ROOT", REPO_ROOT / "external" / "sglang"))
    / "python/sglang/srt/layers/moe/topk.py"
)

OLD = """        elif output_format == TopKOutputFormat.BYPASSED:
            return BypassedTopKOutput(
                hidden_states=hidden_states,
                router_logits=router_logits,
                topk_config=self.topk_config,
                num_token_non_padded=num_token_non_padded,
                expert_location_dispatch_info=expert_location_dispatch_info,
            )
"""

NEW = """        elif output_format == TopKOutputFormat.BYPASSED:
            # [INFIX-R3] flashinfer_trtllm fuses Top-K into its MoE kernel, so
            # the normal select_experts() capture site is otherwise skipped.
            # Keep the fused backend/output unchanged and materialize Top-K
            # only as a route-capture side channel when explicitly requested.
            from sglang.srt.server_args import get_global_server_args

            if get_global_server_args().enable_return_routed_experts:
                self.topk_config.torch_native = False
                with use_symmetric_memory(
                    get_tp_group(), disabled=not is_allocation_symmetric()
                ):
                    select_experts(
                        hidden_states=hidden_states,
                        layer_id=self.layer_id,
                        router_logits=router_logits,
                        topk_config=self.topk_config,
                        num_token_non_padded=num_token_non_padded,
                        expert_location_dispatch_info=expert_location_dispatch_info,
                    )

            return BypassedTopKOutput(
                hidden_states=hidden_states,
                router_logits=router_logits,
                topk_config=self.topk_config,
                num_token_non_padded=num_token_non_padded,
                expert_location_dispatch_info=expert_location_dispatch_info,
            )
"""


def patch(target: Path) -> str:
    source = target.read_text()
    old_count = source.count(OLD)
    new_count = source.count(NEW)

    if old_count == 0 and new_count == 1:
        return "already applied"
    if old_count != 1 or new_count != 0:
        raise RuntimeError(
            f"R3_BYPASS_CAPTURE_FIX: unexpected source in {target}: "
            f"old_count={old_count}, new_count={new_count}"
        )

    target.write_text(source.replace(OLD, NEW))
    patched = target.read_text()
    if patched.count(NEW) != 1 or OLD in patched:
        raise RuntimeError(
            f"R3_BYPASS_CAPTURE_FIX: verification failed for {target}"
        )
    return "applied"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("target", nargs="?", type=Path, default=DEFAULT_TARGET)
    args = parser.parse_args()
    status = patch(args.target)
    print(
        "R3_BYPASS_CAPTURE_FIX: "
        f"{status}; preserve BYPASSED FlashInfer path and add capture side channel "
        f"({args.target})"
    )


if __name__ == "__main__":
    main()
