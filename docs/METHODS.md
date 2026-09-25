# Gate and Repair

TRIAGE combines a detached **Gate** for the policy-gradient loss with an
additive **Repair** objective. Both use the learner–sampler log-probability gap
`delta = learner_log_prob - rollout_log_prob`. PPO still uses its old-policy
ratio, and TIS uses the train-old/sampler ratio; these quantities are distinct.
The implementation is in [`TRIAGE.py`](../policy_extension/TRIAGE.py).

## Gate

Split each response into non-overlapping 64-token segments, retaining the final
short segment and excluding masked positions. A segment is eligible when its
masked mean advantage is negative. For an eligible segment:

| Mean gap | Mean-tier weight |
|---|---:|
| `delta_mean < -1.5` | 0.1 |
| `-1.5 <= delta_mean < -0.5` | 0.3 |
| `delta_mean >= -0.5` | 1.0 |

Let `k` be the number of valid tokens with `delta < -6`. The tail weight is
`1 - 0.9 * min(k / 2, 1)`, giving 1.0, 0.55, and 0.1 for zero, one, and at least
two deep-tail tokens. The segment weight is the minimum of mean-tier and tail
weights. Non-negative-advantage segments retain weight one.

`TRIAGE_GATE_SCOPE` chooses where this segment weight applies:

- `whole`: all valid tokens in an eligible segment;
- `q3`: only its negative-gap tokens, preserving the other tokens' raw weight one.

The reported experiments use `q3`. Gate decisions are detached and materialized
before micro-batching. Integer hundredths preserve the fixed-gate weights
exactly and allow the runtime to accumulate an integer weighted-token
normalizer.

This branch computes

$$
L_{\mathrm{PG}} =
\frac{\sum_{i,t}m_{i,t}w_{i,t}\ell^{\mathrm{PPO/TIS}}_{i,t}}
     {\sum_{i,t}m_{i,t}w_{i,t}}.
$$

The sum is over the complete optimizer update, including all DP ranks and
micro-batches. An alternative reduction normalizes within each response before
taking the response mean; these reductions are different, and the difference is
documented in [PAPER_AND_CODE](PAPER_AND_CODE.md).

## Repair

Repair uses 128-token segments. For a valid segment with masked mean gap $d_R$,

$$
q_R=\max(\tau-d_R,0),\qquad
\phi_\beta(q_R)=\beta^2\left(\sqrt{1+(q_R/\beta)^2}-1\right).
$$

Defaults: `tau = -0.03`, `beta = 0.022360679775`, coefficient `0.015`, and
inclusive starting rollout index `0`. The derivative is bounded by `beta` and
remains nonzero for severe violations. The code retains the exact value above
(equivalently √0.0005); write-ups that round beta to 0.022 refer to the same
value.

| `TRIAGE_REPAIR_SCOPE` | Gradient support |
|---|---|
| `whole` | All valid tokens in an active Repair segment |
| `adv_positive` | All valid tokens if segment-mean advantage is positive |
| `q2` | Negative-gap tokens if segment-mean advantage is positive |

The reported experiments use `adv_positive`. Scope selection preserves the
whole-segment gap statistic by detaching unselected token contributions.
Penalties are included only for segments with selected tokens. The denominator
counts **all valid Repair segments**, including unselected and inactive
segments.

The loss adapts the local Repair numerator to the global weighted-token outer
normalizer, so accumulation recovers one global valid-segment mean. The Repair
coefficient is zero before `TRIAGE_REPAIR_START_STEP` and active from the boundary
onward. Diagnostics are still computed before activation.

## Optional advantage balancing

`TRIAGE_RECENTER_ENABLE` controls the reward hook. It computes a whole-update
P70 active-length reference, damps long-response negative advantages with exponent
0.25 and scale range [0.70, 1.0], then restores each prompt group's zero mean.
It uses explicit group IDs before DP balancing. Set it to `0` for native GRPO
advantages, as in the reported-configuration example in [TRAINING](TRAINING.md).

## Optional Gate calibration

[CALIBRATION](CALIBRATION.md) describes the frozen-reference continuous Gate.
It replaces the fixed mean/tail thresholds only when a profile is explicitly
selected; it does not change Repair or the outer loss reduction.

All defaults live in [`triage_config.py`](../policy_extension/triage_config.py).
Training actors receive the same resolved values through `--train-env-vars`.
