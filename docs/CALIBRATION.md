# Optional Gate reference-window calibration

Calibration is disabled by default. The standard Gate path retains fixed mean-gap
tiers and the deep-token count ramp. Setting `TRIAGE_CALIBRATION_PATH` selects
an alternative continuous gate with frozen reference thresholds. Repair, its scope,
and the global weighted-token reduction are unchanged.

This optional path adapts an earlier development variant of the segment
gate's calibration: negative-depth quantiles and an empirical shared-tail
search over mean-depth and deep-token fraction. Historical
PG-budget allocation, automatic online recalibration, and separate Repair
variants are not enabled by this adapter.

## 1. Export a complete reference window

Choose representative, stable early updates for the model and runtime being
calibrated. The following example exports twenty updates without advantage balancing, Gate, or Repair
interventions. It still uses the TRIAGE adapter to collect metadata and retains
TIS; it is not a complete reproduction of the reported training experiments.

```bash
TRIAGE_RECENTER_ENABLE=0 TRIAGE_GATE_ENABLE=0 TRIAGE_REPAIR_ENABLE=0 \
TRIAGE_CALIBRATION_EXPORT_DIR=/shared/calibration/reference \
NUM_ROLLOUT=20 SAVE_DIR=/shared/checkpoints/reference \
bash examples/train_triage.sh
```

Each DP rank writes one `rollout_<id>_rank_<rank>.jsonl` shard. Collection is an
explicit I/O operation and is not suitable for throughput measurements. It stores
gaps, advantages, and masks, but no prompts, responses, or credentials. Existing
shards are never overwritten. Start a new directory when re-running a reference
experiment. Include all DP ranks from each selected update when fitting.

## 2. Fit and freeze thresholds

```bash
python scripts/calibrate_gate.py \
  --input /shared/calibration/reference \
  --output /shared/calibration/gate.json \
  --segment-size 64 --start-rollout 0 --rounds 20
```

Default calibration parameters are target empirical union rate 0.05, negative-
depth quantile 0.99, full-strength quantile 0.999, and minimum weight 0.1.
The search uses exact empirical `higher` quantiles. It finds the largest shared
tail fraction whose union trigger rate on the reference window is at most the
target. This is an empirical reference-window property, not a guaranteed
false-positive rate on future data. Degenerate features are disabled; a window
with no usable feature is rejected.

The JSON records thresholds, reference IDs/counts, parameters, measured reference
trigger rate, and input-file SHA-256 hashes. The output is created exclusively;
choose another filename to recalibrate. Fitting currently retains reference
statistics in CPU memory; ensure sufficient memory for the chosen window.

An external exporter can supply one JSON object per response with `rollout_id`,
unique `response_id` within the update, `delta` (learner minus sampler log-prob),
`advantage` (scalar or token array), and optional binary `loss_mask`.
Do not mix checkpoints, model families, runtime variants, or incomplete DP shards.

## 3. Use the saved profile

```bash
TRIAGE_CALIBRATION_PATH=/shared/calibration/gate.json \
TRIAGE_GATE_SCOPE=q3 \
bash examples/train_triage.sh
```

The same absolute JSON path must be accessible on all training nodes. The profile
is loaded and validated once when the loss module is imported; no per-update
quantile fitting occurs. Segment size must match the profile. Eligibility stays
negative segment-mean advantage; `whole`/`q3` select the token support as usual.

For each active feature, excess above its reference threshold is normalized by
the distance to the full-strength threshold. The maximum excess is clipped to
[0, 1] and interpolates the segment weight from 1 to `weight_min`. The existing
weighted-token adapter stores integer hundredths, so continuous weights are
rounded to a 0.01 grid before numerator/denominator accumulation. Static mean/tail thresholds are not used while a calibration profile is active;
`triage_gate_calibrated` identifies this mode in the training metrics. The default
fixed-threshold gate and calibrated gate should be reported as distinct variants.
