#!/usr/bin/env bash
# Shared Slime runner. Source from train_{bf16,nvfp4,triage}.sh.

triage_train() {
  if [[ "${1:-}" == --help ]]; then
    cat <<'HELP'
Usage: bash examples/train_{bf16,nvfp4,triage}.sh [extra Slime arguments]

DRY_RUN=1 prints the complete command without checking/starting the GPU stack.
MODEL=qwen3-4B (default) or qwen3-30B-A3B
SLIME_ROOT, SGLANG_ROOT, MEGATRON_ROOT, MINITE_ROOT: installed source trees
HF_CHECKPOINT, INITIAL_CHECKPOINT, PROMPT_DATA, SAVE_DIR: model/data/output paths
LOAD_DIR and optional CKPT_STEP: resume an existing training run
START_LOCAL_RAY=1: start a local head; otherwise use an existing Ray cluster
USE_TIS=0|1, R3=0|1, OPTIMIZER_CPU_OFFLOAD=0|1
TRIAGE_*: optional algorithm overrides, resolved by policy_extension/triage_config.py
HELP
    return
  fi
  local mode="${TRAINING_MODE:?Select a training configuration}" model="${MODEL:-qwen3-4B}"
  case "$mode" in bf16|nvfp4|triage) ;; *) echo "Invalid TRAINING_MODE: $mode" >&2; return 2;; esac
  local model_name moe=0
  case "$model" in
    qwen3-4B) model_name=Qwen3-4B-Base ;;
    qwen3-30B-A3B) model_name=Qwen3-30B-A3B-Base; moe=1 ;;
    *) echo "MODEL must be qwen3-4B or qwen3-30B-A3B" >&2; return 2 ;;
  esac
  local precision=nvfp4
  [[ "$mode" != bf16 ]] || precision=bf16
  export TRAINING_MODE="$mode" MODEL="$model"
  export SLIME_ROOT="${SLIME_ROOT:-$REPO_ROOT/external/slime}"
  export SGLANG_ROOT="${SGLANG_ROOT:-$REPO_ROOT/external/sglang}"
  export MEGATRON_ROOT="${MEGATRON_ROOT:-$REPO_ROOT/external/Megatron-LM}"
  export MINITE_ROOT="${MINITE_ROOT:-$REPO_ROOT/external/miniTransformer}"
  local data_root="${DATA_ROOT:-$REPO_ROOT/data}" model_root="${MODEL_ROOT:-$REPO_ROOT/models}"
  local hf_default="$model_root/$model_name"
  [[ "$precision" != nvfp4 ]] || hf_default="$hf_default-NVFP4"
  export HF_CHECKPOINT="${HF_CHECKPOINT:-$hf_default}"
  export INITIAL_CHECKPOINT="${INITIAL_CHECKPOINT:-${REF_LOAD:-$model_root/${model_name}_torch_dist}}"
  export PROMPT_DATA="${PROMPT_DATA:-$data_root/dapo-math-17k.jsonl}"
  local run_name="${RUN_NAME:-${model}_${mode}}"
  export SAVE_DIR="${SAVE_DIR:-$REPO_ROOT/outputs/$run_name}"
  export ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-1}"
  export ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-8}"
  export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-16}"
  export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-16}"
  export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"
  export R3="${R3:-$moe}"
  local tis_default=0
  [[ "$mode" != triage ]] || tis_default=1
  export USE_TIS="${USE_TIS:-$tis_default}"
  local dry_run="${DRY_RUN:-0}" local_ray="${START_LOCAL_RAY:-0}"
  local cpu_offload="${OPTIMIZER_CPU_OFFLOAD:-1}" apply_patches="${APPLY_PATCHES:-1}"
  local key value
  for value in "$dry_run" "$local_ray" "$cpu_offload" "$apply_patches" "$USE_TIS" "$R3"; do
    [[ "$value" == 0 || "$value" == 1 ]] || { echo "Boolean switches must be 0 or 1" >&2; return 2; }
  done
  for key in ACTOR_NUM_NODES ACTOR_NUM_GPUS_PER_NODE ROLLOUT_BATCH_SIZE N_SAMPLES_PER_PROMPT GLOBAL_BATCH_SIZE; do
    [[ "${!key}" =~ ^[1-9][0-9]*$ ]] || { echo "$key must be a positive integer" >&2; return 2; }
  done
  (( ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT == GLOBAL_BATCH_SIZE )) || {
    echo "rollout batch size * samples per prompt must equal global batch size" >&2; return 2;
  }
  [[ "$local_ray" != 1 || "$ACTOR_NUM_NODES" == 1 ]] || { echo "Local Ray requires one node" >&2; return 2; }
  [[ "$moe" == 1 || "$R3" == 0 ]] || { echo "Dense Qwen3-4B requires R3=0" >&2; return 2; }
  [[ "${CP:-1}" == 1 || "$mode" != triage ]] || { echo "TRIAGE requires CP=1" >&2; return 2; }

  local quant_keys=(SHOULD_REPLACE_TE_GROUPLINEAR QAT_PARAMS DISABLE_BACKWARD_QUANT
    SGLANG_NVFP4_PERTOKEN_SCALE NVTE_NVFP4_DISABLE_RHT
    NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING NVTE_NVFP4_DISABLE_2D_QUANTIZATION)
  if [[ "$precision" == nvfp4 ]]; then
    export SHOULD_REPLACE_TE_GROUPLINEAR="${SHOULD_REPLACE_TE_GROUPLINEAR:-$moe}"
    export QAT_PARAMS="${QAT_PARAMS:-8}" DISABLE_BACKWARD_QUANT="${DISABLE_BACKWARD_QUANT:-0}"
    export SGLANG_NVFP4_PERTOKEN_SCALE="${SGLANG_NVFP4_PERTOKEN_SCALE:-1}"
    export NVTE_NVFP4_DISABLE_RHT="${NVTE_NVFP4_DISABLE_RHT:-0}"
    export NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING="${NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING:-0}"
    export NVTE_NVFP4_DISABLE_2D_QUANTIZATION="${NVTE_NVFP4_DISABLE_2D_QUANTIZATION:-1}"
    export SGLANG_MOE_RUNNER_BACKEND="${SGLANG_MOE_RUNNER_BACKEND:-flashinfer_trtllm}"
  else
    for key in "${quant_keys[@]}"; do unset "$key"; done
    export SHOULD_REPLACE_TE_GROUPLINEAR=0
    export SGLANG_MOE_RUNNER_BACKEND="${SGLANG_MOE_RUNNER_BACKEND:-triton}"
  fi
  if [[ "$mode" == triage ]]; then
    # Only activation switches belong here; all tuning defaults live in Python.
    export TRIAGE_GATE_ENABLE="${TRIAGE_GATE_ENABLE:-1}"
    export TRIAGE_REPAIR_ENABLE="${TRIAGE_REPAIR_ENABLE:-1}"
  else
    for key in ${!TRIAGE_@}; do unset "$key"; done
  fi
  export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
  # Resolve explicit relative paths once on the driver before remote actors use them.
  for key in SLIME_ROOT SGLANG_ROOT MEGATRON_ROOT MINITE_ROOT HF_CHECKPOINT INITIAL_CHECKPOINT PROMPT_DATA SAVE_DIR LOAD_DIR EVAL_DATA TRIAGE_CALIBRATION_PATH TRIAGE_CALIBRATION_EXPORT_DIR; do
    if [[ -n "${!key:-}" ]]; then
      export "$key=$(python3 -c 'import os, sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "${!key}")"
    fi
  done
  export PYTHONPATH="$REPO_ROOT/policy_extension:$MEGATRON_ROOT:$MINITE_ROOT:$SLIME_ROOT:$SGLANG_ROOT/python:${PYTHONPATH:-}"
  local train_env
  train_env="$(python3 "$REPO_ROOT/scripts/train_env.py" "$mode")"

  local -a MODEL_ARGS=()
  source "$REPO_ROOT/configs/models/$model.sh"
  local -a args=(--actor-num-nodes "$ACTOR_NUM_NODES"
    --actor-num-gpus-per-node "$ACTOR_NUM_GPUS_PER_NODE"
    --num-gpus-per-node "$ACTOR_NUM_GPUS_PER_NODE" --colocate
    --hf-checkpoint "$HF_CHECKPOINT" --load "${LOAD_DIR:-$INITIAL_CHECKPOINT}"
    --save "$SAVE_DIR" --save-interval "${SAVE_INTERVAL:-100}" --ckpt-format torch_dist
    --prompt-data "$PROMPT_DATA" --input-key "${INPUT_KEY:-prompt}" --label-key "${LABEL_KEY:-label}"
    --apply-chat-template --rollout-shuffle --rm-type "${RM_TYPE:-math}"
    --seed "${TRAIN_SEED:-42}" --rollout-seed "${ROLLOUT_SEED:-42}"
    --num-rollout "${NUM_ROLLOUT:-1200}" --rollout-batch-size "$ROLLOUT_BATCH_SIZE"
    --n-samples-per-prompt "$N_SAMPLES_PER_PROMPT" --global-batch-size "$GLOBAL_BATCH_SIZE"
    --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN:-16384}"
    --rollout-temperature "${ROLLOUT_TEMPERATURE:-1.0}" --balance-data
    --num-steps-per-rollout 1 --update-weights-interval 1
    --tensor-model-parallel-size "${TP:-1}" --sequence-parallel
    --pipeline-model-parallel-size "${PP:-1}" --context-parallel-size "${CP:-1}"
    --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
    --use-dynamic-batch-size --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-16384}"
    --advantage-estimator grpo --entropy-coef 0.0 --eps-clip 0.2 --eps-clip-high 0.28
    --calculate-per-token-loss --optimizer adam --lr "${LR:-1e-6}" --lr-decay-style constant
    --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 --use-distributed-optimizer
    --rollout-num-gpus-per-engine 1 --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC:-0.7}"
    --sglang-moe-runner-backend "$SGLANG_MOE_RUNNER_BACKEND"
    --sglang-cuda-graph-bs 1 2 4 8 16 24 32 40 48 56 64 72 80 88 96
    --enable-piecewise-cuda-graph --use-slime-router --log-probs-chunk-size 2048
    --transformer-impl transformer_engine --bf16
    --attention-dropout 0.0 --hidden-dropout 0.0 --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32 --attention-backend flash)
  if [[ -z "${LOAD_DIR:-}" ]]; then args+=(--finetune); fi
  if [[ -n "${CKPT_STEP:-}" ]]; then
    [[ -n "${LOAD_DIR:-}" && "$CKPT_STEP" =~ ^[0-9]+$ ]] || { echo "CKPT_STEP requires LOAD_DIR and a non-negative integer" >&2; return 2; }
    args+=(--ckpt-step "$CKPT_STEP")
  fi
  if [[ "$moe" == 1 ]]; then args+=(--expert-model-parallel-size "${EP:-8}" --expert-tensor-parallel-size "${ETP:-1}"); fi
  if [[ "$precision" == nvfp4 ]]; then args+=(--fp4-format e2m1 --fp4-recipe nvfp4 --sglang-quantization modelopt_fp4); fi
  if [[ "$USE_TIS" == 1 ]]; then args+=(--use-tis --tis-clip "${TIS_CLIP_HIGH:-2.0}" --tis-clip-low "${TIS_CLIP_LOW:-0.0}"); fi
  if [[ "$R3" == 1 ]]; then args+=(--use-rollout-routing-replay); fi
  if [[ "$cpu_offload" == 1 ]]; then args+=(--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer); fi
  if [[ "$mode" == triage ]]; then
    args+=(--loss-type custom_loss --custom-loss-function-path TRIAGE.triage_custom_loss
      --rollout-data-postprocess-path TRIAGE.triage_rollout_postprocess)
    if python3 "$REPO_ROOT/scripts/train_env.py" triage --recenter-enabled; then
      args+=(--custom-reward-post-process-path TRIAGE.triage_group_recenter_reward_postprocess)
    fi
  fi
  if [[ -n "${EVAL_DATA:-}" ]]; then
    args+=(--eval-interval "${EVAL_INTERVAL:-10}" --eval-prompt-data "${EVAL_NAME:-aime24}" "$EVAL_DATA"
      --eval-input-key prompt --eval-label-key label --eval-temperature "${EVAL_TEMPERATURE:-0.7}"
      --eval-top-p "${EVAL_TOP_P:-1.0}" --eval-max-response-len "${EVAL_MAX_RESPONSE_LEN:-16384}"
      --n-samples-per-eval-prompt "${N_SAMPLES_PER_EVAL_PROMPT:-4}" --log-passrate)
  fi
  if [[ -n "${WANDB_PROJECT:-}" ]]; then
    args+=(--use-wandb --wandb-project "$WANDB_PROJECT" --wandb-group "$run_name" --wandb-mode "${WANDB_MODE:-online}")
    [[ -z "${WANDB_ENTITY:-}" ]] || args+=(--wandb-team "$WANDB_ENTITY")
  fi
  local -a command=(python3 "$SLIME_ROOT/train.py" "${MODEL_ARGS[@]}" "${args[@]}" --train-env-vars "$train_env" "$@")
  if [[ "$dry_run" == 1 ]]; then
    printf 'RESOLVED_TRAIN_COMMAND:'; printf ' %q' "${command[@]}"; printf '\n'
    return
  fi
  if [[ "$apply_patches" == 1 ]]; then
    [[ "$R3" != 1 ]] || bash "$REPO_ROOT/scripts/apply_patches.sh" r3
    [[ "$mode" != triage ]] || bash "$REPO_ROOT/scripts/apply_patches.sh" tokenmean
  fi
  local -a preflight=(--mode "$mode" --model "$model" --r3 "$R3")
  [[ "${STRICT_VERSIONS:-0}" != 1 ]] || preflight+=(--strict-versions)
  python3 "$REPO_ROOT/scripts/preflight.py" "${preflight[@]}"
  mkdir -p "$SAVE_DIR"
  if [[ "$local_ray" == 1 ]]; then
    ray start --head --node-ip-address "${MASTER_ADDR:-127.0.0.1}" --num-gpus "$ACTOR_NUM_GPUS_PER_NODE" --disable-usage-stats
  fi
  exec "${command[@]}"
}
