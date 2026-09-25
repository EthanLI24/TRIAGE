"""Optional frozen-reference Gate calibration, independent of Slime and Megatron.

The shared empirical tail-fraction search follows the earlier absolute-risk
segment gate. This adapter retains calibration and continuous segment weights;
it does not enable the historical PG-budget or Repair variants.
"""
from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np
import torch

ALGORITHM = "triage_gate_reference_v1"


def higher_quantile(values, quantile):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("quantiles require nonempty finite values")
    index = min(math.ceil(quantile * (values.size - 1)), values.size - 1)
    return float(np.partition(values, index)[index])


def calibrate_shared_feature_quantile(x1, x2, target_fpr):
    """Largest shared empirical tail fraction with union rate <= target_fpr."""
    n = len(x1)
    order1, order2 = np.sort(x1), np.sort(x2)

    def evaluate(k):
        index = min(math.ceil((1 - k / n) * (n - 1)), n - 1)
        t1, t2 = float(order1[index]), float(order2[index])
        union = float(np.mean((x1 > t1) | (x2 > t2)))
        return t1, t2, union

    low, high = 0, n
    while low < high:
        mid = (low + high + 1) // 2
        if evaluate(mid)[2] <= target_fpr + 1e-12:
            low = mid
        else:
            high = mid - 1
    return low / n, *evaluate(low)


def calibrate(records, *, window_size=64, start_rollout=0, rounds=20,
              target_fpr=0.05, quantile_deep=0.99, quantile_full=0.999, weight_min=0.1):
    if window_size <= 0 or rounds <= 0 or start_rollout < 0:
        raise ValueError("segment size/rounds must be positive and start rollout non-negative")
    if not (0 < target_fpr < 1 and 0 < quantile_deep < 1 and 0 < quantile_full < 1):
        raise ValueError("calibration probabilities must be strictly between zero and one")
    if not 0.01 <= weight_min <= 1:
        raise ValueError("weight_min must be in [0.01, 1] for integer-hundredth weights")
    negative_depths, segments, seen, ids = [], [], set(), set()
    for record in records:
        rollout = int(record["rollout_id"])
        if not start_rollout <= rollout < start_rollout + rounds:
            continue
        identity = (rollout, str(record["response_id"]))
        if identity in seen:
            raise ValueError(f"duplicate reference response: {identity}")
        seen.add(identity)
        ids.add(rollout)
        delta = np.asarray(record["delta"], dtype=np.float64)
        mask = np.asarray(record.get("loss_mask", np.ones_like(delta)), dtype=np.float64)
        advantage = np.asarray(record["advantage"], dtype=np.float64)
        if advantage.ndim == 0:
            advantage = np.full_like(delta, advantage)
        if delta.ndim != 1 or not (delta.shape == mask.shape == advantage.shape):
            raise ValueError("delta, advantage, and mask must have matching one-dimensional shapes")
        if not (np.isfinite(delta).all() and np.isfinite(advantage).all() and np.isin(mask, [0, 1]).all()):
            raise ValueError("reference values must be finite and masks binary")
        valid = mask > 0
        negative_depths.extend((-delta[valid & (delta < 0)]).tolist())
        for start in range(0, len(delta), window_size):
            active = valid[start:start + window_size]
            if active.any() and advantage[start:start + window_size][active].mean() < 0:
                segments.append(delta[start:start + window_size][active])
    if ids != set(range(start_rollout, start_rollout + rounds)):
        raise ValueError("reference input must cover every rollout in the requested window")
    if not segments or not negative_depths:
        raise ValueError("reference window needs negative-advantage segments and negative gaps")
    tau_deep = higher_quantile(negative_depths, quantile_deep)
    x1 = np.asarray([-segment.mean() for segment in segments])
    x2 = np.asarray([np.mean(segment < -tau_deep) for segment in segments])
    pf, t1, t2, _ = calibrate_shared_feature_quantile(x1, x2, target_fpr)
    features = {}
    hit = np.zeros(len(segments), dtype=bool)
    for name, values, threshold in (("mean_depth", x1, t1), ("deep_fraction", x2, t2)):
        full = higher_quantile(values, quantile_full)
        active = full > threshold
        features[name] = {"reference": threshold, "full": full, "active": active}
        if active:
            hit |= values > threshold
    state = {"schema_version": 1, "algorithm": ALGORITHM, "window_size": window_size,
             "weight_min": weight_min, "tau_deep": tau_deep, "features": features,
             "reference": {"rollout_ids": sorted(ids), "responses": len(seen), "eligible_segments": len(segments),
                           "target_fpr": target_fpr, "empirical_union_rate": float(hit.mean()),
                           "shared_tail_fraction": pf, "quantile_deep": quantile_deep,
                           "quantile_full": quantile_full, "quantile_interpolation": "higher"}}
    validate_state(state, window_size=window_size)
    return state


def validate_state(state, *, window_size):
    if state.get("schema_version") != 1 or state.get("algorithm") not in {ALGORITHM, "triage_m2_reference_v1"}:
        raise ValueError("unsupported Gate calibration schema/algorithm")
    if state.get("window_size") != window_size:
        raise ValueError("calibrated segment size differs from TRIAGE_GATE_SIZE")
    weight_min, tau = float(state["weight_min"]), float(state["tau_deep"])
    if not (math.isfinite(weight_min) and 0.01 <= weight_min <= 1 and math.isfinite(tau) and tau > 0):
        raise ValueError("invalid calibration weight_min or negative-depth threshold")
    active = False
    for name in ("mean_depth", "deep_fraction"):
        feature = state["features"][name]
        ref, full = float(feature["reference"]), float(feature["full"])
        if not (math.isfinite(ref) and math.isfinite(full)) or type(feature["active"]) is not bool:
            raise ValueError("invalid calibrated feature")
        if feature["active"] != (full > ref):
            raise ValueError("calibration feature activation disagrees with thresholds")
        active |= feature["active"]
    if not active:
        raise ValueError("both calibration features are degenerate; use a representative reference window")


def load_state(path, *, window_size):
    state = json.loads(Path(path).read_text())
    validate_state(state, window_size=window_size)
    state["algorithm"] = ALGORITHM
    return state


@torch.no_grad()
def segment_weights(delta, advantage, mask, mask_f, *, state, apply_scope):
    """Continuous calibrated Gate weights; Repair and the outer reducer are unchanged."""
    if apply_scope not in {"whole", "q3"}:
        raise ValueError("unsupported calibrated Gate apply scope")
    if not (delta.shape == advantage.shape == mask.shape == mask_f.shape):
        raise ValueError("calibrated Gate tensors must have matching shapes")
    result = torch.ones_like(delta)
    for start in range(0, delta.shape[-1], state["window_size"]):
        stop = start + state["window_size"]
        d, a, m, mf = delta[:, start:stop], advantage[:, start:stop], mask[:, start:stop], mask_f[:, start:stop]
        counts = mf.sum(-1)
        denominator = counts.clamp_min(1)
        eligible = (counts > 0) & ((a * mf).sum(-1) / denominator < 0)
        features = {"mean_depth": -(d * mf).sum(-1) / denominator,
                    "deep_fraction": ((d < -state["tau_deep"]) & m).to(d.dtype).sum(-1) / denominator}
        exceedance = torch.zeros_like(counts)
        for name, value in features.items():
            threshold = state["features"][name]
            if threshold["active"]:
                excess = (value - threshold["reference"]) / (threshold["full"] - threshold["reference"])
                exceedance = torch.maximum(exceedance, excess)
        weights = 1 - (1 - state["weight_min"]) * exceedance.clamp(0, 1)
        selected = m & eligible.unsqueeze(-1)
        if apply_scope == "q3":
            selected &= d < 0
        result[:, start:stop] = torch.where(selected, weights.unsqueeze(-1), 1.0)
    return result


def export_reference(directory, rollout_id, rank, rollout_data):
    """One JSONL shard per DP rank/update; never overwrite prior reference data."""
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"rollout_{int(rollout_id):06d}_rank_{int(rank):04d}.jsonl"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=target, prefix=".reference-", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            for index, length in enumerate(rollout_data["response_lengths"]):
                current = torch.as_tensor(rollout_data["log_probs"][index]).detach().float().cpu()[:length]
                rollout = torch.as_tensor(rollout_data["rollout_log_probs"][index]).detach().float().cpu()[:length]
                advantage = torch.as_tensor(rollout_data["advantages"][index]).detach().float().cpu()[:length]
                masks = rollout_data.get("loss_masks")
                mask = torch.ones(length) if masks is None else torch.as_tensor(masks[index]).detach().float().cpu()[:length]
                handle.write(json.dumps({"rollout_id": int(rollout_id), "response_id": f"{rank}:{index}",
                                         "delta": (current - rollout).tolist(), "advantage": advantage.tolist(),
                                         "loss_mask": mask.tolist()}, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        # Publish the completed shard atomically without replacing an existing file.
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
