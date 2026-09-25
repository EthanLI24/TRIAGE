# Source patches

TRIAGE ships a small set of source patches for the pinned Slime and SGLang
checkouts (see `compatibility.json` for the exact revisions). Every patch is
anchored to exact source text, is idempotent, and fails closed with an error
instead of guessing when the upstream source has changed.

Apply them with `scripts/apply_patches.sh`:

```bash
bash scripts/apply_patches.sh r3         # routing-replay (R3) patches only
bash scripts/apply_patches.sh tokenmean  # TRIAGE token-normalizer patch only
bash scripts/apply_patches.sh all        # everything
```

The training launchers apply the right subset automatically (`APPLY_PATCHES=1`
is the default): the R3 set only when `R3=1`, and the token-normalizer patch
only for the TRIAGE mode. `SLIME_ROOT`/`SGLANG_ROOT` select the checkouts and
default to `external/slime` and `external/sglang` under this repository.

## Slime patches (`patches/slime/`)

| Patch | Target file | Purpose | Idempotency marker |
|---|---|---|---|
| `patch_routing_replay_indices.py` | `slime/utils/routing_replay.py` | Casts compact int32 rollout expert IDs to `torch.long` when routing replay moves them to the GPU (2 sites) | `dtype=torch.long` replacement present at both anchors |
| `patch_slime_r3_route_validation.py` | `slime/rollout/sglang_rollout.py` | Fails fast when an R3 rollout omits `routed_experts` or returns an unpopulated (zero/duplicate/out-of-range) route cache | `[INFIX-R3]` validation block present once |
| `patch_tokenmean_custom_normalizer.py` | `slime/backends/megatron_utils/loss.py`, `slime/backends/megatron_utils/model.py` | Lets a custom loss return an exact local token normalizer and forwards the per-sample `triage_token_weight_units` field through the training DataIterator | `SLIME_TRIAGE_TOKENMEAN_CUSTOM_NORMALIZER_V1` in both files; stock files must match pinned SHA256 before patching |

## SGLang patches (`patches/sglang/`)

| Patch | Target file | Purpose | Idempotency marker |
|---|---|---|---|
| `patch_sglang_r3_force_return.py` | `python/sglang/srt/managers/scheduler.py` | Restores routed-experts return when the typed sglang_router `/generate` proxy drops the per-request `return_routed_experts` field, by honoring the server-level opt-in | `[INFIX-R3]` fallback expression present once |
| `patch_sglang_r3_bypassed_topk.py` | `python/sglang/srt/layers/moe/topk.py` | Captures MoE routes on the fused BYPASSED Top-K path (`flashinfer_trtllm`) via a side-channel `select_experts` call, without changing the FlashInfer output consumed downstream | `[INFIX-R3]` side-channel block present once |

The token-normalizer patch is additionally hash-gated: unmarked target files
must match the pinned stock SHA256 of the reference Slime revision, so stale
or hand-modified checkouts are refused rather than silently re-patched.
