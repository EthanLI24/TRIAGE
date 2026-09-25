"""TRIAGE policy stability: group-recentered P70, Gate weighting, and Repair.

The reward hook adjusts advantages before DP balancing. The rollout hook
materializes detached token weights and global update metadata before training.
The custom loss returns a numerator, metrics, and integer weighted-token
normalizer for the supplied Slime adapter. Context parallel size must be 1,
and each rollout must contain exactly one complete optimizer update.
"""

import json
import logging

import torch

logger = logging.getLogger(__name__)

from triage_config import (
    RECENTER_ENABLE,
    ADV_QUANTILE,
    GATE_ENABLE,
    REPAIR_ENABLE,
    TOKEN_WEIGHT_SCALE,
    TOKEN_WEIGHT_KEY,
    ADV_ALPHA,
    ADV_MIN,
    ADV_MAX,
    GATE_SIZE,
    GATE_NEG_DELTA,
    GATE_NEG_W,
    GATE_SEV_DELTA,
    GATE_SEV_W,
    GATE_BAD_DELTA,
    GATE_BAD_COUNT_FULL,
    GATE_NEG_ADV_MAX,
    GATE_SCOPES,
    GATE_SCOPE,
    REPAIR_SIZE,
    REPAIR_COEF,
    REPAIR_START_STEP,
    REPAIR_HUBER_BETA,
    REPAIR_TARGET,
    REPAIR_SCOPES,
    REPAIR_SCOPE,
    CALIBRATION_PATH,
    CALIBRATION_EXPORT_DIR,
)


CALIBRATION_STATE = None
if CALIBRATION_PATH:
    from triage_calibration import load_state
    CALIBRATION_STATE = load_state(CALIBRATION_PATH, window_size=GATE_SIZE)


# Set once per update by triage_rollout_postprocess.  A module-level channel is appropriate here:
# the postprocess hook and custom loss run serially in the same Megatron actor process, and the
# current production rollout has exactly one global training batch per update.
_GLOBAL_LENGTH_REFS = None
_LAST_RECENTER_METRICS = None


def _group_recenter_advantages(
    raw_rewards: list[float],
    active_lengths: list[float],
    group_indices: list[int],
    *,
    expected_group_size: int,
    std_normalization: bool,
) -> tuple[list[float], dict[str, float]]:
    """Apply negative-only length damping, then restore each GRPO group's zero mean.

    This runs before sequence-length balancing scatters a prompt group across DP ranks and
    token-budgeted micro-batches.  ``group_indices`` are explicit identities from ``Sample``;
    input order is deliberately irrelevant.
    """
    if not (len(raw_rewards) == len(active_lengths) == len(group_indices)):
        raise RuntimeError(
            "triage recenter received mismatched rewards/lengths/groups: "
            f"{len(raw_rewards)}, {len(active_lengths)}, {len(group_indices)}"
        )
    if not raw_rewards:
        raise RuntimeError("triage recenter received an empty update")
    if expected_group_size <= 1:
        raise RuntimeError(f"triage recenter requires GRPO group size > 1, got {expected_group_size}")

    rewards = torch.tensor(raw_rewards, dtype=torch.float32, device="cpu")
    lengths = torch.tensor(active_lengths, dtype=torch.float32, device="cpu")
    if not torch.isfinite(rewards).all():
        raise RuntimeError("triage recenter received non-finite raw rewards")
    if not torch.isfinite(lengths).all() or (lengths <= 0).any():
        raise RuntimeError("triage recenter requires every sample to have a finite active length > 0")

    groups: dict[int, list[int]] = {}
    for position, group_index in enumerate(group_indices):
        groups.setdefault(group_index, []).append(position)
    bad_sizes = {
        group_index: len(indices)
        for group_index, indices in groups.items()
        if len(indices) != expected_group_size
    }
    if bad_sizes:
        raise RuntimeError(
            "triage recenter found incomplete prompt groups; "
            f"expected {expected_group_size} samples per group, got {bad_sizes}"
        )

    # Reproduce Slime's default GRPO reward normalization exactly, but by explicit group
    # identity rather than relying on contiguous reshape order.
    baseline = torch.empty_like(rewards)
    baseline_group_means = []
    baseline_group_stds = []
    for indices in groups.values():
        idx = torch.tensor(indices, dtype=torch.long)
        values = rewards[idx]
        values = values - values.mean()
        if std_normalization:
            values = values / (values.std() + 1e-6)
        baseline[idx] = values
        baseline_group_means.append(values.mean())
        baseline_group_stds.append(values.std(unbiased=False))

    adv_ref = float(torch.quantile(lengths, ADV_QUANTILE).item())
    if not adv_ref > 0:
        raise RuntimeError(f"triage recenter global P70 reference must be positive, got {adv_ref}")
    candidate_scale = torch.pow(torch.clamp(adv_ref / lengths, max=1.0), ADV_ALPHA).clamp(ADV_MIN, ADV_MAX)
    if not torch.isfinite(candidate_scale).all():
        raise RuntimeError("triage recenter produced non-finite candidate scales")

    applied_scale = torch.where(baseline < 0, candidate_scale, torch.ones_like(candidate_scale))
    candidate = baseline * applied_scale
    recentered = torch.empty_like(candidate)
    candidate_group_means = []
    post_group_means = []
    post_group_stds = []
    for indices in groups.values():
        idx = torch.tensor(indices, dtype=torch.long)
        values = candidate[idx]
        candidate_group_means.append(values.mean())
        values = values - values.mean()
        recentered[idx] = values
        post_group_means.append(values.mean())
        post_group_stds.append(values.std(unbiased=False))

    def _max_abs(values: list[torch.Tensor]) -> float:
        return float(torch.stack(values).abs().max().item())

    metrics = {
        "recenter_enabled": 1.0,
        "adv_ref_global": adv_ref,
        "candidate_scale_mean": float(candidate_scale.mean().item()),
        "candidate_scale_min": float(candidate_scale.min().item()),
        "applied_scale_mean": float(applied_scale.mean().item()),
        "scaled_seq_fraction": float((candidate_scale < 0.999999).float().mean().item()),
        "scaled_negative_fraction": float(((baseline < 0) & (candidate_scale < 0.999999)).float().mean().item()),
        "baseline_group_mean_maxabs": _max_abs(baseline_group_means),
        # "pre" is the old, sign-asymmetric candidate immediately before recentering.
        "pre_group_mean_maxabs": _max_abs(candidate_group_means),
        "post_group_mean_maxabs": _max_abs(post_group_means),
        "baseline_group_std_mean": float(torch.stack(baseline_group_stds).mean().item()),
        "post_group_std_mean": float(torch.stack(post_group_stds).mean().item()),
        "group_count": float(len(groups)),
        "group_size": float(expected_group_size),
        "global_n": float(len(raw_rewards)),
    }
    if not torch.isfinite(recentered).all() or not all(torch.isfinite(torch.tensor(v)) for v in metrics.values()):
        raise RuntimeError("triage recenter produced non-finite rewards or diagnostics")
    return recentered.tolist(), metrics


def _sample_active_length(sample) -> float:
    if getattr(sample, "remove_sample", False):
        raise RuntimeError("triage recenter does not accept fully removed/masked samples")
    response_length = int(sample.response_length)
    if response_length <= 0:
        raise RuntimeError(f"triage recenter requires response_length > 0, got {response_length}")
    loss_mask = getattr(sample, "loss_mask", None)
    if loss_mask is None:
        return float(response_length)
    if len(loss_mask) != response_length:
        raise RuntimeError(
            f"triage recenter loss-mask length {len(loss_mask)} != response length {response_length}"
        )
    if isinstance(loss_mask, torch.Tensor):
        return float(loss_mask.detach().float().sum().item())
    return float(sum(loss_mask))


def _emit_recenter_metrics(metrics: dict[str, float]) -> None:
    logger.info("TRIAGE_RECENTER_METRICS %s", json.dumps(metrics, sort_keys=True))
    # This hook executes once on the RolloutManager (the W&B primary process).  Keep logging
    # best-effort so a telemetry outage can never invalidate an otherwise valid optimizer step.
    try:
        import wandb

        if wandb.run is not None:
            wandb.log({f"triage_recenter/{key}": value for key, value in metrics.items()})
    except Exception as exc:  # pragma: no cover - only exercised with a live W&B backend
        logger.warning("failed to upload TRIAGE recenter metrics: %s", exc)


def triage_group_recenter_reward_postprocess(args, samples):
    """Slime reward hook: default GRPO normalization -> neg-only P70 -> group recenter.

    The hook intentionally requires exactly one complete optimizer update.  It must run in
    RolloutManager before ``balance_data`` splits prompt siblings across DP ranks.  No standard
    deviation restoration is applied after recentering; that is a separate future ablation.
    """
    global _LAST_RECENTER_METRICS
    _LAST_RECENTER_METRICS = None

    if not RECENTER_ENABLE:
        raise RuntimeError("triage recenter hook was wired while TRIAGE_RECENTER_ENABLE=0")
    if args.advantage_estimator not in {"grpo", "gspo"}:
        raise RuntimeError(f"triage recenter requires GRPO/GSPO, got {args.advantage_estimator}")
    if not args.rewards_normalization:
        raise RuntimeError("triage recenter requires Slime rewards_normalization")
    if not isinstance(samples, list) or not samples or isinstance(samples[0], list):
        raise RuntimeError("triage recenter expects the complete flattened Sample list")

    group_size = int(args.n_samples_per_prompt)
    expected_global_n = int(args.global_batch_size)
    generated_global_n = int(args.rollout_batch_size) * group_size
    if expected_global_n != generated_global_n or len(samples) != expected_global_n:
        raise RuntimeError(
            "triage recenter requires one complete update: "
            f"len(samples)={len(samples)}, global_batch_size={expected_global_n}, "
            f"rollout_batch_size*n_samples={generated_global_n}"
        )

    sample_indices = [getattr(sample, "index", None) for sample in samples]
    if any(index is None for index in sample_indices) or len(set(sample_indices)) != len(sample_indices):
        raise RuntimeError("triage recenter requires unique, non-null sample indices")
    group_indices = [getattr(sample, "group_index", None) for sample in samples]
    if any(group_index is None for group_index in group_indices):
        raise RuntimeError("triage recenter found a sample with missing group_index")
    if len(set(group_indices)) != int(args.rollout_batch_size):
        raise RuntimeError(
            "triage recenter group count mismatch: "
            f"found {len(set(group_indices))}, expected rollout_batch_size={args.rollout_batch_size}"
        )

    raw_rewards = [float(sample.get_reward_value(args)) for sample in samples]
    active_lengths = [_sample_active_length(sample) for sample in samples]
    horizon = float(args.rollout_max_response_len)
    if horizon <= 0 or max(active_lengths) > horizon + 1e-6:
        raise RuntimeError(
            f"triage recenter active length exceeds invalid horizon: max={max(active_lengths)}, horizon={horizon}"
        )

    rewards, metrics = _group_recenter_advantages(
        raw_rewards,
        active_lengths,
        group_indices,
        expected_group_size=group_size,
        std_normalization=bool(args.grpo_std_normalization),
    )
    _LAST_RECENTER_METRICS = metrics
    _emit_recenter_metrics(metrics)
    return raw_rewards, rewards


def _rollout_length_metadata(rollout_data) -> tuple[list[float], int, int]:
    """Return active lengths, total samples, and valid REPAIR_SIZE segments."""
    response_lengths = rollout_data.get("response_lengths", [])
    loss_masks = rollout_data.get("loss_masks")
    if loss_masks is None:
        lengths = [float(v) for v in response_lengths if int(v) > 0]
        valid_segments = sum((int(v) + REPAIR_SIZE - 1) // REPAIR_SIZE for v in response_lengths if int(v) > 0)
        return lengths, len(response_lengths), valid_segments
    if len(loss_masks) != len(response_lengths):
        raise RuntimeError(
            "triage rollout postprocess received mismatched response_lengths/loss_masks: "
            f"{len(response_lengths)} vs {len(loss_masks)}"
        )

    lengths = []
    valid_segments = 0
    for response_length, loss_mask in zip(response_lengths, loss_masks, strict=True):
        if isinstance(loss_mask, torch.Tensor):
            flat_mask = loss_mask.detach().float().reshape(-1)[: int(response_length)].cpu()
        else:
            flat_mask = torch.as_tensor(loss_mask, dtype=torch.float32).reshape(-1)[: int(response_length)]
        length = float(flat_mask.sum().item())
        # A fully masked response does not contribute to active-length metadata.
        if length > 0 and int(response_length) > 0:
            lengths.append(length)
        for start in range(0, int(response_length), REPAIR_SIZE):
            if flat_mask[start : start + REPAIR_SIZE].sum().item() > 0:
                valid_segments += 1
    return lengths, len(response_lengths), valid_segments


def _aggregate_length_metadata(
    gathered: list[dict],
    *,
    expected_global_batch: int,
    horizon: float,
    rollout_id: int,
) -> dict:
    """Aggregate rank metadata; kept separate to test arbitrary DP partitions on CPU."""
    all_lengths = [value for rank_data in gathered for value in rank_data["lengths"]]
    global_total_n = sum(int(rank_data["total_n"]) for rank_data in gathered)
    global_valid_segments = sum(int(rank_data["valid_segments"]) for rank_data in gathered)
    dp_world_size = len(gathered)

    if not all_lengths:
        raise RuntimeError("triage rollout postprocess found no loss-contributing responses")
    if global_total_n != expected_global_batch:
        raise RuntimeError(
            "triage metadata requires exactly one complete update: "
            f"gathered {global_total_n} samples ({len(all_lengths)} active) across DP={dp_world_size}, "
            f"expected global_batch_size={expected_global_batch}. "
            "Move reference computation to per-step metadata before using multiple steps/rollout."
        )
    if global_valid_segments <= 0:
        raise RuntimeError("triage rollout postprocess found no valid Repair segments")
    if horizon <= 0:
        raise RuntimeError(f"rollout_max_response_len must be positive, got {horizon}")

    max_length = max(all_lengths)
    if max_length > horizon + 1e-6:
        raise RuntimeError(f"effective response length {max_length} exceeds horizon {horizon}")

    result = {
        "rollout_id": int(rollout_id),
        "global_n": int(global_total_n),
        "active_n": len(all_lengths),
        "global_valid_segments": int(global_valid_segments),
        "dp_world_size": int(dp_world_size),
        "horizon": float(horizon),
    }
    weighted_metadata = [
        (rank_data.get("token_count"), rank_data.get("token_weight_units"))
        for rank_data in gathered
    ]
    if any(token_count is not None or weight_units is not None for token_count, weight_units in weighted_metadata):
        if any(token_count is None or weight_units is None for token_count, weight_units in weighted_metadata):
            raise RuntimeError("triage token-level metadata must be present on every DP rank")
        global_token_count = sum(int(token_count) for token_count, _ in weighted_metadata)
        global_token_weight_units = sum(int(weight_units) for _, weight_units in weighted_metadata)
        if global_token_count <= 0 or global_token_weight_units <= 0:
            raise RuntimeError(
                "triage token-level normalization requires positive global token counts, got "
                f"tokens={global_token_count}, weight_units={global_token_weight_units}"
            )
        if global_token_weight_units > TOKEN_WEIGHT_SCALE * global_token_count:
            raise RuntimeError(
                "triage weighted-token normalizer exceeds the unweighted token count: "
                f"units={global_token_weight_units}, tokens={global_token_count}"
            )
        result.update(
            {
                "global_token_count": int(global_token_count),
                "global_token_weight_units": int(global_token_weight_units),
            }
        )
    return result


def triage_rollout_postprocess(args, rollout_id, rollout_data) -> None:
    """Collect weighted-token and segment counts for one complete DP-wide update."""
    global _GLOBAL_LENGTH_REFS
    _GLOBAL_LENGTH_REFS = None  # never leak a stale reference after a failed/skipped update

    import torch.distributed as dist
    from megatron.core import mpu

    local_lengths, local_total_n, local_valid_segments = _rollout_length_metadata(rollout_data)
    local_metadata = {
        "lengths": local_lengths,
        "total_n": local_total_n,
        "valid_segments": local_valid_segments,
    }
    token_weight_units, local_token_count, local_weight_units = _materialize_token_weight_units(
        rollout_data
    )
    # DataIterator keeps a reference to rollout_data, so this detached per-sample
    # field follows the exact same balancing and micro-batch partition as the data.
    rollout_data[TOKEN_WEIGHT_KEY] = token_weight_units
    local_metadata.update(
        {
            "token_count": int(local_token_count),
            "token_weight_units": int(local_weight_units),
        }
    )
    if not dist.is_available() or not dist.is_initialized():
        gathered = [local_metadata]
        dp_world_size = 1
    else:
        group = mpu.get_data_parallel_group_gloo(with_context_parallel=True)
        dp_world_size = dist.get_world_size(group=group)
        gathered = [None] * dp_world_size
        dist.all_gather_object(gathered, local_metadata, group=group)

    expected_global_batch = int(rollout_data.get("dynamic_global_batch_size", args.global_batch_size))
    horizon = float(args.rollout_max_response_len)
    _GLOBAL_LENGTH_REFS = _aggregate_length_metadata(
        gathered,
        expected_global_batch=expected_global_batch,
        horizon=horizon,
        rollout_id=rollout_id,
    )

    if CALIBRATION_EXPORT_DIR:
        from triage_calibration import export_reference
        distributed = dist.is_available() and dist.is_initialized()
        # TP/PP replicas share samples; export each DP shard only once.
        writer = not distributed or (
            mpu.get_tensor_model_parallel_rank() == 0
            and mpu.get_pipeline_model_parallel_rank() == 0
        )
        if writer:
            rank = dist.get_rank(group=group) if distributed else 0
            export_reference(CALIBRATION_EXPORT_DIR, rollout_id, rank, rollout_data)


def _require_global_length_refs(horizon: float | None = None) -> dict:
    if _GLOBAL_LENGTH_REFS is None:
        raise RuntimeError(
            "global update metadata is unset. Add "
            "--rollout-data-postprocess-path TRIAGE.triage_rollout_postprocess; "
            "local micro-batch metadata fallback is intentionally disabled."
        )
    refs = _GLOBAL_LENGTH_REFS
    if horizon is not None and abs(float(refs["horizon"]) - float(horizon)) > 1e-6:
        raise RuntimeError(
            f"stale/mismatched triage horizon: hook={refs['horizon']} loss={horizon}"
        )
    return refs


def _pad_rows(concat_1d, resp_lens, T):
    """Concat over response tokens (CP=1) -> padded [n_seq, T], zero-padded."""
    out = concat_1d.new_zeros((len(resp_lens), T))
    off = 0
    for i, L in enumerate(resp_lens):
        if L > 0:
            out[i, :L] = concat_1d[off:off + L]
        off += L
    return out


def _one_sided_pseudo_huber(seg_delta, *, target: float, beta: float):
    """Threshold penalty plus its positive force magnitude.

    Unlike ``clamp(relu(target-delta)**2, max=cap)``, the force approaches ``beta``
    smoothly for severe negative mismatch instead of becoming exactly zero.
    """
    gap = torch.relu(target - seg_delta)
    scaled_gap = gap / beta
    root = torch.sqrt(1.0 + scaled_gap.square())
    penalty = beta**2 * (root - 1.0)
    force = gap / root
    return penalty, force, gap


def _effective_repair_coef(
    *, rollout_id: int, enabled: bool, start_step: int, coef: float
) -> float:
    """Return the scheduled Repair coefficient for this rollout update."""
    if rollout_id < 0:
        raise ValueError(f"rollout_id must be non-negative, got {rollout_id}")
    if start_step < 0:
        raise ValueError(f"Repair start_step must be non-negative, got {start_step}")
    if coef < 0:
        raise ValueError(f"Repair coef must be non-negative, got {coef}")
    return float(coef) if enabled and rollout_id >= start_step else 0.0


def _scoped_repair_segment_delta(
    segment_delta_live,
    segment_adv_detached,
    segment_mask,
    segment_mask_f,
    *,
    grad_scope: str,
):
    """Preserve the segment-wide Repair forward statistic while scoping its gradient.

    ``adv_positive`` keeps whole-segment backward support only for a positive
    detached segment-mean advantage. ``q2``
    keeps only negative-delta tokens in positive-advantage segments. Non-selected
    live deltas are replaced by value-identical detached copies, so segment
    values, triggers, penalties, and denominators remain unchanged.
    """
    if grad_scope not in REPAIR_SCOPES:
        raise ValueError(
            f"Repair grad_scope must be one of {sorted(REPAIR_SCOPES)}, "
            f"got {grad_scope!r}"
        )
    if not (
        segment_delta_live.shape
        == segment_adv_detached.shape
        == segment_mask.shape
        == segment_mask_f.shape
    ):
        raise RuntimeError(
            "Repair scope requires matching delta/advantage/mask shapes, got "
            f"{tuple(segment_delta_live.shape)}, {tuple(segment_adv_detached.shape)}, "
            f"{tuple(segment_mask.shape)}, {tuple(segment_mask_f.shape)}"
        )

    token_count = segment_mask_f.sum(dim=-1)
    denominator = token_count.clamp_min(1.0)
    valid = token_count > 0
    segment_advantage = (
        segment_adv_detached.detach() * segment_mask_f
    ).sum(dim=-1) / denominator

    if grad_scope == "whole":
        grad_mask = segment_mask
    elif grad_scope == "adv_positive":
        grad_mask = segment_mask & (segment_advantage > 0).unsqueeze(-1)
    else:
        grad_mask = (
            segment_mask
            & (segment_advantage > 0).unsqueeze(-1)
            & (segment_delta_live.detach() < 0)
        )

    scoped_delta = torch.where(
        grad_mask,
        segment_delta_live,
        segment_delta_live.detach(),
    )
    segment_delta = (scoped_delta * segment_mask_f).sum(dim=-1) / denominator
    return segment_delta, grad_mask, valid


def _apply_stock_tis(pg_loss, train_old_log_probs, rollout_log_probs, *, lower: float, upper: float):
    """Apply Slime's stock token-level TIS correction.

    This intentionally mirrors ``vanilla_tis_function`` in
    ``slime.backends.megatron_utils.loss``: the behavior-policy ratio is
    ``exp(train_old_logp - rollout_logp)`` and is clipped before multiplying
    the token policy-gradient loss.
    """
    if lower < 0 or upper <= 0 or lower > upper:
        raise ValueError(f"invalid stock TIS bounds: lower={lower}, upper={upper}")
    if not (pg_loss.shape == train_old_log_probs.shape == rollout_log_probs.shape):
        raise RuntimeError(
            "stock TIS requires matching pg/train/rollout shapes, got "
            f"{tuple(pg_loss.shape)}, {tuple(train_old_log_probs.shape)}, "
            f"{tuple(rollout_log_probs.shape)}"
        )
    tis = torch.exp(train_old_log_probs - rollout_log_probs)
    tis_abs = (tis - 1).abs()
    tis_weights = torch.clamp(tis, min=lower, max=upper)
    tis_clipfrac = (tis_weights != tis).to(pg_loss.dtype)
    return pg_loss * tis_weights, tis, tis_clipfrac, tis_abs


def _gate_weights(
    rawdelta_det_pad,
    det_adv,
    response_mask,
    response_mask_f,
    *,
    apply_scope: str,
):
    """Apply the Gate deep-token count ramp with whole-segment or Q3-only application.

    The mean-delta branch retains the legacy hard ``1/0.3/0.1`` tiers.  The
    tail branch linearly reaches the severe weight when ``GATE_BAD_COUNT_FULL``
    tokens cross ``GATE_BAD_DELTA``; with the production defaults this gives
    ``k=0/1/>=2 -> 1/0.55/0.1``.
    """
    if apply_scope not in GATE_SCOPES:
        raise ValueError(
            "segment apply_scope must be one of "
            f"{sorted(GATE_SCOPES)}, got {apply_scope!r}"
        )
    if not (
        rawdelta_det_pad.shape
        == det_adv.shape
        == response_mask.shape
        == response_mask_f.shape
    ):
        raise RuntimeError(
            "Gate segment gate requires matching delta/advantage/mask shapes, got "
            f"{tuple(rawdelta_det_pad.shape)}, {tuple(det_adv.shape)}, "
            f"{tuple(response_mask.shape)}, {tuple(response_mask_f.shape)}"
        )

    pg_weight = torch.ones_like(rawdelta_det_pad)
    width = rawdelta_det_pad.shape[-1]
    for s in range(0, width, GATE_SIZE):
        e = min(s + GATE_SIZE, width)
        m, mf = response_mask[:, s:e], response_mask_f[:, s:e]
        tc = mf.sum(dim=-1)
        denom = tc.clamp_min(1.0)
        valid = tc > 0
        segment_delta = rawdelta_det_pad[:, s:e]
        seg_delta = (segment_delta * mf).sum(dim=-1) / denom
        seg_adv = (det_adv[:, s:e] * mf).sum(dim=-1) / denom
        seg_badcount = (
            ((segment_delta < GATE_BAD_DELTA) & m).to(mf.dtype)
        ).sum(dim=-1)
        eligible = valid & (seg_adv < GATE_NEG_ADV_MAX)
        neg = eligible & (seg_delta < GATE_NEG_DELTA)
        mean_sev = eligible & (seg_delta < GATE_SEV_DELTA)

        mean_w = torch.ones_like(seg_delta)
        mean_w = torch.where(neg, torch.full_like(mean_w, GATE_NEG_W), mean_w)
        mean_w = torch.where(mean_sev, torch.full_like(mean_w, GATE_SEV_W), mean_w)

        tail_strength = (seg_badcount / float(GATE_BAD_COUNT_FULL)).clamp(0.0, 1.0)
        tail_w = 1.0 + (GATE_SEV_W - 1.0) * tail_strength
        tail_w = torch.where(eligible, tail_w, torch.ones_like(tail_w))
        w = torch.minimum(mean_w, tail_w)

        apply_mask = m if apply_scope == "whole" else m & (segment_delta < 0)
        pg_weight[:, s:e] = torch.where(
            apply_mask,
            w.unsqueeze(-1),
            pg_weight[:, s:e],
        )
    return pg_weight


@torch.no_grad()
def _materialize_token_weight_units(rollout_data):
    """Freeze Gate weights from the update's current-policy log-prob pass.

    Slime computes ``rollout_data['log_probs']`` with the actor immediately
    before advantages and this hook, without changing model parameters before
    the training forward.  Gate's gate is detached by definition, so storing the
    resulting integer weight units gives every later micro-batch an exact,
    partition-invariant denominator without adding a second model forward.
    """
    required = ("response_lengths", "loss_masks", "log_probs", "rollout_log_probs", "advantages")
    missing = [key for key in required if rollout_data.get(key) is None]
    if missing:
        raise RuntimeError(f"triage token-level postprocess is missing fields: {missing}")

    fields = [rollout_data[key] for key in required]
    count = len(fields[0])
    if count == 0 or any(len(values) != count for values in fields[1:]):
        raise RuntimeError(
            "triage token-level postprocess received empty or misaligned per-sample fields"
        )

    result = []
    local_token_count = 0
    local_weight_units = 0
    for index in range(count):
        response_length = int(rollout_data["response_lengths"][index])
        if response_length <= 0:
            raise RuntimeError(f"triage token-level response {index} has invalid length {response_length}")

        current = rollout_data["log_probs"][index].detach().reshape(-1)[:response_length]
        rollout = rollout_data["rollout_log_probs"][index].detach().to(current.device).reshape(-1)[:response_length]
        advantage = rollout_data["advantages"][index].detach().to(current.device).reshape(-1)[:response_length]
        mask = rollout_data["loss_masks"][index].detach().to(current.device).reshape(-1)[:response_length]
        if not (current.numel() == rollout.numel() == advantage.numel() == mask.numel() == response_length):
            raise RuntimeError(
                f"triage token-level response {index} has inconsistent token fields for length {response_length}"
            )
        mask_bool = mask > 0
        if not mask_bool.any():
            raise RuntimeError(f"triage token-level response {index} has no valid loss tokens")

        if GATE_ENABLE:
            gate = _gate_weights
            extra = {}
            if CALIBRATION_STATE is not None:
                from triage_calibration import segment_weights
                gate = segment_weights
                extra["state"] = CALIBRATION_STATE
            weights = gate(
                (current - rollout).float().unsqueeze(0),
                advantage.float().unsqueeze(0),
                mask_bool.unsqueeze(0),
                mask_bool.float().unsqueeze(0),
                apply_scope=GATE_SCOPE,
                **extra,
            ).squeeze(0)
        else:
            weights = torch.ones(response_length, dtype=torch.float32, device=current.device)

        units = torch.where(
            mask_bool,
            torch.round(weights * TOKEN_WEIGHT_SCALE).to(torch.int32),
            torch.zeros_like(weights, dtype=torch.int32),
        )
        valid_units = units[mask_bool]
        if not torch.all((valid_units >= 1) & (valid_units <= TOKEN_WEIGHT_SCALE)):
            raise RuntimeError(f"triage token-level response {index} produced unsupported Gate weights")
        result.append(units.cpu())
        local_token_count += int(mask_bool.sum().item())
        local_weight_units += int(units.sum().item())

    if local_token_count <= 0 or local_weight_units <= 0:
        raise RuntimeError(
            "triage token-level postprocess produced a non-positive local normalizer: "
            f"tokens={local_token_count}, weight_units={local_weight_units}"
        )
    return result, local_token_count, local_weight_units


def triage_custom_loss(args, batch, logits, sum_of_sample_mean):
    """Return (loss numerator, metric numerators, local weighted-token units)."""
    from slime.backends.megatron_utils.loss import get_log_probs_and_entropy
    from slime.utils.ppo_utils import compute_approx_kl, compute_policy_loss

    horizon = float(args.rollout_max_response_len)
    refs = _require_global_length_refs(horizon)
    rollout_id = int(refs["rollout_id"])
    repair_schedule_active = rollout_id >= REPAIR_START_STEP
    repair_effective_coef = _effective_repair_coef(
        rollout_id=rollout_id,
        enabled=REPAIR_ENABLE,
        start_step=REPAIR_START_STEP,
        coef=REPAIR_COEF,
    )
    if not bool(getattr(args, "calculate_per_token_loss", False)):
        raise RuntimeError("triage_custom_loss requires --calculate-per-token-loss")
    if float(args.entropy_coef) != 0.0:
        raise RuntimeError("the exact token-level adapter currently requires entropy_coef=0")
    if bool(args.use_kl_loss) and float(args.kl_loss_coef) != 0.0:
        raise RuntimeError("the exact token-level adapter currently requires kl_loss_coef=0")
    for key in ("global_token_count", "global_token_weight_units"):
        if key not in refs:
            raise RuntimeError(f"triage token-level global reference is missing {key}")
    resp_lens = [int(x) for x in batch["response_lengths"]]
    n, T = len(resp_lens), (max(resp_lens) if resp_lens else 1)

    advantages = torch.cat(batch["advantages"], dim=0)
    train_old_log_probs = torch.cat(batch["log_probs"], dim=0)      # train/old policy logp
    rollout_log_probs = torch.cat(batch["rollout_log_probs"], dim=0)
    _, lpe = get_log_probs_and_entropy(
        logits, args=args, unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=batch["total_lengths"], response_lengths=batch["response_lengths"],
        with_entropy=True, max_seq_lens=batch.get("max_seq_lens", None),
    )
    log_probs = torch.cat(lpe["log_probs"], dim=0)                  # current policy logp (from logits)
    entropy = torch.cat(lpe["entropy"], dim=0)
    device, dtype = log_probs.device, log_probs.dtype
    raw_delta = log_probs - rollout_log_probs

    # Pad response tokens to [n, T] for segment operations.
    response_mask_f = _pad_rows(
        torch.cat([batch["loss_masks"][i].to(device=device, dtype=dtype).reshape(-1)[:resp_lens[i]]
                   for i in range(n)], dim=0) if n else log_probs.new_zeros(0),
        resp_lens, T)
    response_mask = response_mask_f > 0
    adv_pad = _pad_rows(advantages.to(dtype), resp_lens, T)
    # Gating decisions are detached; Repair must use the live current-policy logp.
    rawdelta_live_pad = _pad_rows(raw_delta.to(dtype), resp_lens, T)
    det_adv = adv_pad.detach()
    ppo_old_log_probs = rollout_log_probs if getattr(args, "use_rollout_logprobs", False) else train_old_log_probs
    oldlp_pad = _pad_rows(ppo_old_log_probs.to(dtype), resp_lens, T)
    newlp_pad = _pad_rows(log_probs.to(dtype), resp_lens, T)

    # ---- PPO clipped loss (slime compute_policy_loss) ----
    pg_loss_pad, clipfrac_pad = compute_policy_loss(oldlp_pad - newlp_pad, adv_pad, args.eps_clip, args.eps_clip_high)

    # ---- Slime stock token-level TIS (train-old behavior correction) ----
    tis_on = bool(getattr(args, "use_tis", False))
    if getattr(args, "get_mismatch_metrics", False):
        raise RuntimeError(
            "triage_custom_loss does not yet support --get-mismatch-metrics/custom rejection masks"
        )
    if tis_on and getattr(args, "custom_tis_function_path", None) is not None:
        raise RuntimeError(
            "triage_custom_loss currently supports stock vanilla TIS only; "
            "remove --custom-tis-function-path"
        )
    tis = torch.ones_like(train_old_log_probs)
    tis_clipfrac = torch.zeros_like(train_old_log_probs)
    tis_abs = torch.zeros_like(train_old_log_probs)
    ois = torch.ones_like(train_old_log_probs)
    if tis_on:
        # Stock applies TIS to the flattened response-token loss.  Flattening before the
        # correction also guarantees that padding never contributes to diagnostics.
        pg_loss_flat = torch.cat(
            [pg_loss_pad[i, :L] for i, L in enumerate(resp_lens)], dim=0
        )
        pg_loss_flat, tis, tis_clipfrac, tis_abs = _apply_stock_tis(
            pg_loss_flat,
            train_old_log_probs,
            rollout_log_probs,
            lower=float(args.tis_clip_low),
            upper=float(args.tis_clip),
        )
        pg_loss_pad = _pad_rows(pg_loss_flat, resp_lens, T)
        # Same ``(-ppo_kl).exp()`` diagnostic as stock policy_loss_function.
        ois = torch.exp(log_probs - ppo_old_log_probs)

    # ---- Gate detached segment weights ----
    # The gate was materialized by triage_rollout_postprocess
    # from Slime's immediately preceding current-policy forward.  Carrying those
    # exact units with the sample lets Megatron aggregate one global denominator
    # across arbitrary DP and micro-batch partitions.
    token_weight_units = batch.get(TOKEN_WEIGHT_KEY)
    if token_weight_units is None or len(token_weight_units) != n:
        raise RuntimeError(
            f"triage token-level batch is missing {TOKEN_WEIGHT_KEY} for {n} responses"
        )
    flat_units = torch.cat(
        [
            units.to(device=device, dtype=torch.int32).reshape(-1)[: resp_lens[i]]
            for i, units in enumerate(token_weight_units)
        ],
        dim=0,
    ) if n else torch.zeros(0, device=device, dtype=torch.int32)
    token_weight_units_pad = _pad_rows(flat_units, resp_lens, T)
    valid_units = token_weight_units_pad[response_mask]
    if not torch.all((valid_units >= 1) & (valid_units <= TOKEN_WEIGHT_SCALE)):
        raise RuntimeError("triage token-level batch contains unsupported Gate weight units")
    if torch.any(token_weight_units_pad[~response_mask] != 0):
        raise RuntimeError("triage token-level batch assigns weight to a masked token")
    pg_weight = token_weight_units_pad.to(dtype) / float(TOKEN_WEIGHT_SCALE)
    # Global weighted-token policy numerator.
    # Units are 100*m*w, so numerator/normalizer is exactly
    # sum(m*w*token_loss) / sum(m*w); the common factor 100 cancels.
    custom_normalizer = token_weight_units_pad.sum(dtype=torch.int32).detach()
    if custom_normalizer.item() <= 0:
        raise RuntimeError("triage token-level micro-batch has a non-positive normalizer")
    pg_loss = (pg_loss_pad * token_weight_units_pad.to(dtype)).sum()

    # ---- entropy + KL (concat + passed sum_of_sample_mean; not seg-gated; matches policy_loss_function) ----
    entropy_loss = sum_of_sample_mean(entropy)
    loss = pg_loss - args.entropy_coef * entropy_loss
    kl_loss = log_probs.new_zeros(())
    if args.use_kl_loss:
        ref_log_probs = torch.cat(batch["ref_log_probs"], dim=0)
        kl_loss = sum_of_sample_mean(compute_approx_kl(log_probs, ref_log_probs, kl_loss_type=args.kl_loss_type))
        loss = loss + args.kl_loss_coef * kl_loss

    # ---- threshold Repair: all segments below target, no percentile/top-k selection ----
    # Pseudo-Huber smoothly bounds |d penalty / d seg_delta| by REPAIR_HUBER_BETA while retaining
    # non-zero force on the worst mismatch segments.  A hard clamp would make those gradients 0.
    repair_num = rawdelta_live_pad.new_zeros(n)
    repair_cnt = rawdelta_live_pad.new_zeros(n)
    repair_active_cnt = rawdelta_live_pad.new_zeros(n)
    repair_delta_sum = rawdelta_live_pad.new_zeros(n)
    repair_force_sum = rawdelta_live_pad.new_zeros(n)
    repair_scope_fraction_sum = rawdelta_live_pad.new_zeros(n)
    repair_scoped_force_sum = rawdelta_live_pad.new_zeros(n)
    for s in range(0, T, REPAIR_SIZE):
        e = min(s + REPAIR_SIZE, T)
        m = response_mask[:, s:e]
        mf = response_mask_f[:, s:e]
        tc = mf.sum(dim=-1)
        seg_delta, repair_grad_mask, valid_bool = _scoped_repair_segment_delta(
            rawdelta_live_pad[:, s:e],
            det_adv[:, s:e],
            m,
            mf,
            grad_scope=REPAIR_SCOPE,
        )
        valid = valid_bool.to(dtype)
        pen, force, gap = _one_sided_pseudo_huber(
            seg_delta, target=REPAIR_TARGET, beta=REPAIR_HUBER_BETA
        )
        scope_fraction = (
            repair_grad_mask.to(dtype).sum(dim=-1) / tc.clamp_min(1.0)
        ) * valid
        active_bool = valid_bool & (gap.detach() > 0)
        scope_selected = valid_bool & (scope_fraction.detach() > 0)
        scope_selected_f = scope_selected.to(dtype)
        repair_num = repair_num + pen * scope_selected_f
        repair_cnt = repair_cnt + valid
        repair_active_cnt = repair_active_cnt + (active_bool & scope_selected).to(dtype)
        repair_delta_sum = repair_delta_sum + seg_delta.detach() * scope_selected_f
        repair_force_sum = repair_force_sum + force.detach() * scope_selected_f
        repair_scope_fraction_sum = repair_scope_fraction_sum + scope_fraction.detach()
        repair_scoped_force_sum = repair_scoped_force_sum + (
            force.detach() * scope_fraction.detach()
        )
    # Megatron will divide the accumulated loss by global_token_weight_units.
    # Multiplying local penalty sums by U/S therefore preserves the exact
    # global Repair segment mean: sum(P_local * U/S) / U = sum(P) / S.
    repair_outer_scale = float(refs["global_token_weight_units"]) / float(
        refs["global_valid_segments"]
    )
    repair_loss = repair_num.sum() * repair_outer_scale
    if repair_effective_coef > 0:
        loss = loss + repair_effective_coef * repair_loss

    if log_probs.numel() == 0:
        loss = loss + 0 * logits.sum()

    global_units = float(refs["global_token_weight_units"])
    global_tokens = float(refs["global_token_count"])
    global_responses = float(refs["global_n"])
    global_segments = float(refs["global_valid_segments"])
    local_units = custom_normalizer.to(dtype)
    token_metric_scale = global_units / global_tokens
    response_metric_scale = global_units / global_responses
    segment_metric_scale = global_units / global_segments

    def constant_metric(value):
        return local_units * float(value)

    def token_sum_metric(value):
        return value * token_metric_scale

    reported = {
        # Megatron divides every accumulated metric by the same custom
        # weighted-token normalizer.  Each numerator below is therefore
        # explicitly adapted to its intended token/response/segment mean.
        "loss": loss.clone().detach(),
        "pg_loss": pg_loss.clone().detach(),
        "entropy_loss": token_sum_metric(entropy_loss).clone().detach(),
        "ppo_kl": token_sum_metric(
            sum_of_sample_mean(ppo_old_log_probs - log_probs)
        ).clone().detach(),
        "kl_loss": token_sum_metric(kl_loss).clone().detach(),
        "pg_clipfrac": (
            (clipfrac_pad * response_mask_f).sum() * token_metric_scale
        ).clone().detach(),
        "train_rollout_logprob_abs_diff": token_sum_metric(
            sum_of_sample_mean((train_old_log_probs - rollout_log_probs).abs())
        ).clone().detach(),
        "triage_repair_loss": repair_loss.clone().detach(),
        "triage_repair_contribution": (
            repair_effective_coef * repair_loss
        ).clone().detach(),
        "triage_repair_active_segment_fraction": (
            repair_active_cnt.sum() * segment_metric_scale
        ).clone().detach(),
        "triage_repair_valid_segments_per_seq": (
            repair_cnt.sum() * response_metric_scale
        ).clone().detach(),
        "triage_repair_mean_delta": (
            repair_delta_sum.sum() * segment_metric_scale
        ).clone().detach(),
        "triage_repair_force_mean": (
            repair_force_sum.sum() * segment_metric_scale
        ).clone().detach(),
        "triage_repair_enabled": constant_metric(REPAIR_ENABLE),
        "triage_repair_coef": constant_metric(REPAIR_COEF),
        "triage_repair_start_step": constant_metric(REPAIR_START_STEP),
        "triage_repair_current_step": constant_metric(rollout_id),
        "triage_repair_schedule_active": constant_metric(
            REPAIR_ENABLE and repair_schedule_active
        ),
        "triage_repair_effective_coef": constant_metric(repair_effective_coef),
        "triage_repair_huber_beta": constant_metric(REPAIR_HUBER_BETA),
        "triage_repair_grad_scope_whole": constant_metric(REPAIR_SCOPE == "whole"),
        "triage_repair_grad_scope_adv_positive": constant_metric(
            REPAIR_SCOPE == "adv_positive"
        ),
        "triage_repair_grad_scope_q2": constant_metric(REPAIR_SCOPE == "q2"),
        "triage_repair_scope_token_fraction": (
            repair_scope_fraction_sum.sum() * segment_metric_scale
        ).clone().detach(),
        "triage_repair_scoped_force_mean": (
            repair_scoped_force_sum.sum() * segment_metric_scale
        ).clone().detach(),
        "triage_repair_scoped_force_contribution": (
            repair_effective_coef
            * repair_scoped_force_sum.sum()
            * segment_metric_scale
        ).clone().detach(),
        "triage_gate_weight_mean": (
            (pg_weight * response_mask_f).sum() * token_metric_scale
        ).clone().detach(),
        "triage_gate_enabled": constant_metric(GATE_ENABLE),
        "triage_gate_calibrated": constant_metric(GATE_ENABLE and CALIBRATION_STATE is not None),
        "triage_gate_scope_q3": constant_metric(GATE_ENABLE and GATE_SCOPE == "q3"),
        "triage_gate_scope_whole": constant_metric(GATE_ENABLE and GATE_SCOPE == "whole"),
        "triage_gate_bad_count_full": constant_metric(GATE_BAD_COUNT_FULL),
        "triage_token_weight_mean": constant_metric(
            global_units / (TOKEN_WEIGHT_SCALE * global_tokens)
        ),
        "triage_global_token_count": constant_metric(global_tokens),
        "triage_global_token_weight_units": constant_metric(global_units),
        "triage_tis_enabled": constant_metric(tis_on),
        "triage_response_horizon": constant_metric(horizon),
        "triage_global_n": constant_metric(refs["global_n"]),
        "triage_active_n": constant_metric(refs["active_n"]),
        "triage_global_valid_segments": constant_metric(refs["global_valid_segments"]),
        "triage_dp_world_size": constant_metric(refs["dp_world_size"]),
    }
    if tis_on:
        reported.update(
            {
                "ois": token_sum_metric(sum_of_sample_mean(ois)).clone().detach(),
                "tis": token_sum_metric(sum_of_sample_mean(tis)).clone().detach(),
                "tis_clipfrac": token_sum_metric(
                    sum_of_sample_mean(tis_clipfrac)
                ).clone().detach(),
                "tis_abs": token_sum_metric(sum_of_sample_mean(tis_abs)).clone().detach(),
            }
        )
    return loss, reported, custom_normalizer
