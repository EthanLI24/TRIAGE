# Environment setup

TRIAGE is an overlay: this repository contains the policy-loss extension,
source patches, launchers, and tests, but not the training engines. Training
requires the pinned reference stack below, checked out under `external/` (or
pointed to with `*_ROOT` environment variables).

## Reference stack

Pinned revisions are recorded in [`compatibility.json`](../compatibility.json),
which `scripts/preflight.py` and `scripts/bootstrap_external.sh` both read.

| Component | Repository | Pinned revision |
|---|---|---|
| Slime | [THUDM/slime](https://github.com/THUDM/slime) | `56eb6b8b17e0a9cf913bef1b6b72b5e3611e341b` |
| SGLang | [sgl-project/sglang](https://github.com/sgl-project/sglang) | `3dddaba34c0e266c64b6d5c4b055cc5e842636ca` (v0.5.9, router 0.3.2) |
| Megatron-LM | [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM) | `3714d81d418c9f1bca4594fc35f9e8289f652862` |
| miniTransformer | internal release (see below) | `5fa9373427d528748b4ae205bdd6dc97720eb0ab` |
| FlashInfer | pip package | `0.6.13` |
| Transformer Engine | pip package | not pinned; supplied by the training environment |

**miniTransformer availability is limited.** It is an internal release and, to
our knowledge, has no public repository URL. It is required only for MoE NVFP4
modes (`SHOULD_REPLACE_TE_GROUPLINEAR=1`); dense BF16/NVFP4/TRIAGE runs on
Qwen3-4B do not need it. If you have access, place a checkout at
`external/miniTransformer` at the pinned commit above.

FlashInfer is a pip-installable package pinned at `0.6.13` in the reference
stack. The other compiled components (Transformer Engine, CUTLASS, `fp4_gemm`,
CUDA kernels) come from your training environment; this repository does not
redistribute them. NVFP4 modes additionally require Blackwell-class GPUs.

## Stack versions used for measurements

The pinned reference stack above is what the source patches anchor to and what
preflight verifies. The measurement campaigns behind the reported results ran
a nearby runtime: SGLang 0.5.10 with FlashInfer 0.6.12, recorded under
`measurement_environments` in [compatibility.json](../compatibility.json).
The pinned revisions — SGLang v0.5.9 and FlashInfer 0.6.13 — remain the
supported reference configuration; the measurement runtime is provenance, not
a replacement.

## Bootstrap

Clone the three public repositories at their pinned commits:

```bash
bash scripts/bootstrap_external.sh
```

Or do it manually:

```bash
mkdir -p external
git clone https://github.com/THUDM/slime external/slime
git -C external/slime checkout 56eb6b8b17e0a9cf913bef1b6b72b5e3611e341b
git clone https://github.com/sgl-project/sglang external/sglang
git -C external/sglang checkout 3dddaba34c0e266c64b6d5c4b055cc5e842636ca
git clone https://github.com/NVIDIA/Megatron-LM external/Megatron-LM
git -C external/Megatron-LM checkout 3714d81d418c9f1bca4594fc35f9e8289f652862
```

The expected layout afterwards:

```
external/
├── slime/            # THUDM/slime @ 56eb6b8
├── sglang/           # sgl-project/sglang @ 3dddaba
├── Megatron-LM/      # NVIDIA/Megatron-LM @ 3714d81
└── miniTransformer/  # internal release, MoE NVFP4 only
```

To use existing checkouts instead, set `SLIME_ROOT`, `SGLANG_ROOT`,
`MEGATRON_ROOT`, and `MINITE_ROOT`; the launchers, patch scripts, and preflight
all honor them.

## Patch and verify

Apply the source patches (see [patches/README.md](../patches/README.md) for
what each patch does), then run preflight:

```bash
bash scripts/apply_patches.sh all   # or: r3 / tokenmean
python3 scripts/preflight.py --mode triage
```

The patches are idempotent and fail closed on unknown source, so re-running is
safe. Preflight verifies overlay hashes, required Slime CLI flags, patch state,
and pinned revisions (`STRICT_VERSIONS=1` turns revision mismatches into
errors). The training launchers perform both steps automatically on real
launches (`APPLY_PATCHES=1` is the default); `DRY_RUN=1` skips all of it.

See [TRAINING](TRAINING.md) for launch options and the
[model and data prerequisites](TRAINING.md#models-and-data), and
[STACK_AND_CHANGES](STACK_AND_CHANGES.md) for what the overlay changes.
