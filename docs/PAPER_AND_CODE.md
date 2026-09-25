# Paper correspondence and reproduction notes

How this repository maps to the accompanying paper, **TRIAGE: Direction-Aware
Mismatch Stabilization of Native NVFP4 Reinforcement Learning**, and where the
code intentionally differs from the objective as written there.

## Implementation map

| Component | Entry point | Documentation |
|---|---|---|
| Gate | `_gate_weights` | [METHODS](METHODS.md) |
| Repair | `_scoped_repair_segment_delta`, `_one_sided_pseudo_huber` | [METHODS](METHODS.md) |
| Update normalization | `triage_rollout_postprocess`, `triage_custom_loss` | reduction difference below |
| Optional advantage balancing | `triage_group_recenter_reward_postprocess` | extra implementation option |
| Optional calibrated Gate | `triage_calibration.segment_weights` | [CALIBRATION](CALIBRATION.md) |

## Defaults and experiment settings

The reusable defaults retain Gate scope `whole`, Repair scope `whole`, and
advantage balancing on. To match the reported configuration — directional Gate,
positive-advantage Repair support, and native GRPO advantages:

```bash
TRIAGE_GATE_SCOPE=q3 TRIAGE_REPAIR_SCOPE=adv_positive \
TRIAGE_RECENTER_ENABLE=0 bash examples/train_triage.sh
```

Repair coefficient **0.015** and start index **0** match the reported
configuration. The code retains `beta = 0.022360679775` (equivalently √0.0005);
write-ups that round beta to 0.022 refer to the same value.

## Loss-reduction difference

The paper's objective computes a weighted mean within each response, then takes
the outer response mean. This branch retains **global weighted-token
normalization** over the complete optimizer update, spanning all DP ranks and
micro-batches. The two reductions are not identical, and scope selection does
not remove the difference. The calibrated Gate is optional and is not the
fixed-threshold Gate used in the reported main experiments.

## Results and reproducibility

[benchmarks/results/paper.json](../benchmarks/results/paper.json) records the
nine final-checkpoint accuracy rows, the natural-generation timing runs, the
paired learner-overhead measurement, and the standalone/long-tail serving
protocols; the tables in [benchmarks](../benchmarks/README.md) are generated
from it. The overview, training-dynamics, and ablation figures are the original
vector assets, with provenance recorded in
[assets/figures](../assets/figures/README.md). The learner–sampler mismatch
study is documented self-contained in [CHARACTERIZATION](CHARACTERIZATION.md),
with aggregates under
[benchmarks/results/characterization/](../benchmarks/results/characterization/).

The ablation continuations branch all four variants from the full-TRIAGE
checkpoint at step 300, so the ablated TIS continuation is not the same
experiment as TIS trained from the beginning in the training-dynamics curves.
Numeric per-variant ablation summaries and a distributable final-checkpoint
evaluation runner are tracked under
[Planned measurements](../benchmarks/README.md#planned-measurements); curves
are not converted into guessed values.

The timing runs use Qwen3-30B-A3B-Base on 8×B300 with 20,480-token responses,
R3 capture off, and exclude evaluation and checkpoint saving; the reusable
training defaults differ. See [benchmark protocols](../benchmarks/README.md)
and [validation](VALIDATION.md).

## Naming migration

| Previous setting | Current setting |
|---|---|
| `TRIAGE_SEG_APPLY_SCOPE` | `TRIAGE_GATE_SCOPE` |
| `TRIAGE_SEG_GATE_SIZE` | `TRIAGE_GATE_SIZE` |
| Other `TRIAGE_SEG_*` | `TRIAGE_GATE_*` |
| `TRIAGE_ALIGN_GRAD_SCOPE` | `TRIAGE_REPAIR_SCOPE` |
| Other `TRIAGE_ALIGN_*` | `TRIAGE_REPAIR_*` |

Metrics use `triage_gate_*` and `triage_repair_*`. The calibration command is
`scripts/calibrate_gate.py`. Previously generated calibration profiles remain
loadable; new profiles use the Gate identifier. Update old launch environments;
unknown names are rejected. The old single-purpose 30B wrapper and activation
config have been removed; use `MODEL=qwen3-30B-A3B bash examples/train_triage.sh`.
