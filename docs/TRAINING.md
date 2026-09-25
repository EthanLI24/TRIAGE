# Training guide

## Layout and installation

The three entry points share `examples/common.sh` and differ only by mode:

| Entry point | Learner/sampler precision | TIS default | TRIAGE loss |
|---|---|---|---|
| `examples/train_bf16.sh` | BF16 / BF16 | Off | Off |
| `examples/train_nvfp4.sh` | Native NVFP4 / ModelOpt FP4 | Off | Off |
| `examples/train_triage.sh` | Native NVFP4 / ModelOpt FP4 | On, [0, 2] | On |

Set `USE_TIS=1` on the NVFP4 entry point for the NVFP4+TIS baseline. Baseline
modes clear inherited TRIAGE settings. BF16 also clears NVFP4-only environment
switches and explicitly disables the GroupedLinear replacement.

Install the lightweight overlay in an existing compatible training environment:

```bash
python -m pip install -e .
bash scripts/run_tests.sh
```

The training engines and compiled quantization kernels are external dependencies.
Clone the pinned stack into `external/` with `bash scripts/bootstrap_external.sh`
— see [ENVIRONMENT](ENVIRONMENT.md) for the full setup, including miniTransformer
availability — and [STACK_AND_CHANGES](STACK_AND_CHANGES.md) for what the overlay
changes. The default runtime source locations are `external/slime`,
`external/sglang`, `external/Megatron-LM`, and `external/miniTransformer` under
this repository. Set `SLIME_ROOT`, `SGLANG_ROOT`, `MEGATRON_ROOT`, and
`MINITE_ROOT` to use existing installations. No installation or patching is
performed by `DRY_RUN=1`.

## Model and data paths

The default model is `qwen3-4B`, following the supplied dense-model training recipe.
Use `MODEL=qwen3-30B-A3B` for the MoE architecture. Architecture arrays live in
`configs/models/`; source-equivalent values were checked against the local Slime
model scripts. Verify tokenizer/vocabulary and model architecture when supplying
other checkpoints; arbitrary model families are not supported by these examples.

| Variable | Default / meaning |
|---|---|
| `MODEL_ROOT` | `<repository>/models` |
| `DATA_ROOT` | `<repository>/data` |
| `HF_CHECKPOINT` | `<MODEL_ROOT>/Qwen3-4B-Base` for BF16; add `-NVFP4` for NVFP4/TRIAGE |
| `INITIAL_CHECKPOINT` | `<MODEL_ROOT>/Qwen3-4B-Base_torch_dist`; `REF_LOAD` remains an alias |
| `PROMPT_DATA` | `<DATA_ROOT>/dapo-math-17k.jsonl` |
| `SAVE_DIR` | `<repository>/outputs/<model>_<mode>` |
| `LOAD_DIR` | Optional training checkpoint to resume; otherwise load the initial checkpoint with `--finetune` |
| `CKPT_STEP` | Optional explicit step; requires `LOAD_DIR` |
| `EVAL_DATA` | Optional evaluation JSONL; unset means no online evaluation |
| `WANDB_PROJECT` | Optional; unset means no W&B flags |

The [Models and data](#models-and-data) section below describes what the
checkpoints and prompt dataset must contain and how to obtain them.

30B defaults use `Qwen3-30B-A3B-Base` in model/checkpoint names. All paths can
be overridden. Paths must be visible at the same location on every Ray node.
Credentials use your existing runtime authentication; the scripts do not read
personal credential files, replace `HOME`, or print credentials in commands.
Save to a different directory from the initial/resume checkpoint.

## Models and data

Training inputs are not distributed with this repository; prepare them before
launching:

- **NVFP4 HF checkpoint (NVFP4/TRIAGE modes).** `HF_CHECKPOINT` must point to
  an NVFP4-quantized Hugging Face checkpoint. The reported runs used
  `modelopt_fp4`-quantized exports of the Qwen3 base models produced with the
  [TensorRT Model Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer).
  Quantization and conversion tooling is not yet packaged in this repository
  (roadmap item); reference checkpoints are planned but not yet available.
- **Megatron initial checkpoint.** `INITIAL_CHECKPOINT` must point to a
  Megatron `torch_dist` checkpoint converted from the HF weights (default:
  `<MODEL_ROOT>/<model>_torch_dist`).
- **Prompt data.** `PROMPT_DATA` expects JSONL with a prompt field and a label
  field (`--input-key prompt --label-key label --rm-type math`). The reported
  runs used the public
  [DAPO-Math-17K](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k)
  dataset converted to that schema.

## Run and resume

```bash
# Full command generation; no GPU, model files, Ray, or training stack required.
DRY_RUN=1 bash examples/train_triage.sh

# Use an already running Ray cluster.
HF_CHECKPOINT=/path/to/model-NVFP4 \
INITIAL_CHECKPOINT=/path/to/model_torch_dist \
PROMPT_DATA=/path/to/train.jsonl \
SAVE_DIR=./outputs/triage_run \
bash examples/train_triage.sh

# A local head can be started explicitly. Existing Ray processes are not stopped.
START_LOCAL_RAY=1 bash examples/train_bf16.sh

# Resume without changing the original checkpoint directory.
LOAD_DIR=/path/to/run_checkpoint SAVE_DIR=./outputs/resumed_run \
CKPT_STEP=299 bash examples/train_triage.sh
```

All examples accept extra Slime CLI arguments at the end. Use environment
variables for the documented controls. Passing duplicate CLI options is handled
by the installed Slime parser and is not a substitute for inspecting dry-run output.

Shared defaults: 1 node × 8 GPUs, 16 prompts × 16 responses, batch 256,
1,200 rollouts, 16,384 response tokens, TP/PP/CP=1, Adam at 1e-6,
PPO clip [0.8, 1.28], one optimizer update and weight synchronization per rollout.
CPU optimizer offload is enabled, matching the reference script; disable with
`OPTIMIZER_CPU_OFFLOAD=0` when appropriate for your hardware.

For 30B, EP defaults to 8 and R3 defaults on; dense 4B requires R3 off.
`TP`, `PP`, `CP`, `EP`, `ETP`, GPU counts, batch sizes, length budgets, LR, and
rollout count accept environment overrides. TRIAGE currently requires CP=1 and
zero entropy/reference-KL coefficients. All modes use token-level policy reduction.
Batch invariant: `ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT = GLOBAL_BATCH_SIZE`.

## Algorithm defaults and overrides

`policy_extension/triage_config.py` is the single source of truth. The TRIAGE
entry point only activates Gate and Repair. Resolved defaults and explicit overrides
are serialized as JSON and forwarded to remote actors through `--train-env-vars`.
Unknown `TRIAGE_*` keys and obsolete `GP95_*` overrides are rejected.

| Setting | Code default |
|---|---|
| Group recentering | Enabled |
| Gate segment size / scope | 64 / `whole` |
| Gate negative/severe weights | 0.3 / 0.1 |
| Gate tail full count | 2 |
| Repair coefficient | **0.015** |
| Repair scope | `whole` (segment-wide Repair) |
| Repair start step | 0 |
| Repair segment size / target / beta | 128 / -0.03 / 0.022360679775 |
| Calibration | Disabled; see [CALIBRATION](CALIBRATION.md) |

```bash
TRIAGE_REPAIR_SCOPE=adv_positive TRIAGE_GATE_SCOPE=q3 \
TRIAGE_RECENTER_ENABLE=0 TRIAGE_REPAIR_START_STEP=300 \
bash examples/train_triage.sh
```

This explicitly selects positive-advantage Repair, Q3 gating, native GRPO advantages, and a delayed
Repair start. `q2` selects Q2 Repair. The default coefficient stays 0.015 unless
`TRIAGE_REPAIR_COEF` is set. These examples are configurable training recipes,
not an assertion that every default matches every reported experiment; see
[PAPER_AND_CODE](PAPER_AND_CODE.md).

## Patches and preflight

Source patches live in [`patches/`](../patches/README.md), organized by target
repository (`patches/slime/`, `patches/sglang/`), and are applied by
`scripts/apply_patches.sh` with a patch-set argument (`r3`, `tokenmean`, or
`all`). For actual launches, `APPLY_PATCHES=1` is the default: the shared
runner applies R3 patches only when R3 is enabled, and the token-normalizer
patch only for TRIAGE. Set `APPLY_PATCHES=0` for an already prepared
environment; preflight still validates required interfaces and source state.
Exact revision matching is available with `STRICT_VERSIONS=1`. Existing Slime
patches with obsolete markers must be replaced from a clean pinned source
before applying the current adapter.

A zero reference-KL coefficient does not require loading a reference model, so
these defaults omit `--use-kl-loss` and `--ref-load`. The initial actor checkpoint
is passed through `--load`; it is distinct from a reference-policy forward pass.
Online evaluation is optional and uses the supplied script's defaults: every ten
updates, four samples, temperature 0.7, top-p 1, and 16,384 output tokens.
Those are training-monitor settings, not the five-benchmark final evaluation protocol.
