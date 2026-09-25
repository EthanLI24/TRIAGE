# Reference Stack and Runtime Changes

## Reference stack

The compatibility patches target the following reference stack (pinned in
[compatibility.json](../compatibility.json); setup steps in
[ENVIRONMENT](ENVIRONMENT.md)):

| Component | Repository | Revision |
|---|---|---|
| Slime | [THUDM/slime](https://github.com/THUDM/slime) | `56eb6b8b17e0a9cf913bef1b6b72b5e3611e341b` |
| SGLang | [sgl-project/sglang](https://github.com/sgl-project/sglang) | `3dddaba34c0e266c64b6d5c4b055cc5e842636ca` |
| Megatron-LM | [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM) | `3714d81d418c9f1bca4594fc35f9e8289f652862` |
| miniTransformer | internal release; no public URL | `5fa9373427d528748b4ae205bdd6dc97720eb0ab` |
| FlashInfer | pip package | `0.6.13` |

The repository does not contain these projects or any compiled libraries. The
training image remains the source of Slime, SGLang, Megatron-LM,
miniTransformer, FlashInfer, CUTLASS, and `fp4_gemm`.

## Overlay contents

`policy_extension/TRIAGE.py` contains the token-level policy-loss
variant used by this branch. Its SHA256 is pinned in `compatibility.json`.
It exposes three Slime callbacks:

- `triage_group_recenter_reward_postprocess`
- `triage_rollout_postprocess`
- `triage_custom_loss`

The shared training runner selects BF16, NVFP4, or TRIAGE. Algorithm defaults
are defined once in `policy_extension/triage_config.py`; Gate and Repair are
activated by the TRIAGE entry point. See [TRAINING](TRAINING.md) for controls,
[METHODS](METHODS.md) for the loss, and [CALIBRATION](CALIBRATION.md) for the
optional frozen-reference Gate.

The token-normalizer adapter forwards per-sample integer weights and accumulates
one global weighted-token denominator. Repair is scaled to retain its global
valid-segment mean. This adapter is required only by the TRIAGE mode.

The four R3 source patches live in `patches/slime/` and `patches/sglang/`
(see [patches/README.md](../patches/README.md)) and make routing replay work
with the reference `flashinfer_trtllm` ModelOpt FP4 path:

1. Cast compact rollout expert IDs to `torch.long` when replay moves them to GPU.
2. Preserve routed-expert return when the typed SGLang router drops the request field.
3. Capture Top-K routes while keeping the fused BYPASSED FlashInfer MoE path.
4. Reject missing, empty, duplicated, or out-of-range route data before training.

Each patch uses exact source anchors, is idempotent, and fails instead of
guessing when the upstream source has changed.

## Architecture boundary

The loss module and R3 source patches are Python and are not tied to x86 or
ARM. Compiled miniTransformer, FlashInfer, CUTLASS, and CUDA artifacts are
architecture-specific and must come from the training image. Do not copy
compiled objects between B200/x86 and GB200/ARM environments.

## Compatibility and migration from earlier versions

The [compatibility manifest](../compatibility.json) pins the reference revisions
and overlay hashes. The R3 patches accept clean reference source or the matching
already-patched source and stop on unknown implementations. For a newer stack,
verify native route return, validation, and replay before setting
`APPLY_PATCHES=0`.

This release uses `TRIAGE_*` environment variables and `triage_*` callbacks,
metrics, and batch fields. Custom launchers must use the updated names and
`--calculate-per-token-loss`. The token-normalizer patch has a TRIAGE-specific
marker; files patched with the earlier adapter must first be restored from the
pinned Slime revision before applying this version. Legacy sequence reduction
and loss-side advantage adjustment have been removed.

Actual launches run preflight after applying the requested patches.
Set `STRICT_VERSIONS=1` on the training command to enforce exact reference
revisions. For example:

```bash
STRICT_VERSIONS=1 MODEL=qwen3-30B-A3B bash examples/train_triage.sh
```

Upstream frameworks, compiled CUDA extensions, model weights, datasets, and
credentials are not redistributed here. See [NOTICE](../NOTICE.md).
