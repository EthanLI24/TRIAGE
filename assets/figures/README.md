# Figure assets

Each figure ships as a vector PDF plus a PNG preview rendered at 2,200 px on the
longest edge. Markdown pages embed the PNG; grab the PDF when you need a
resolution-independent copy.

## Top-level figures

| Asset | Files | Shown in |
|---|---|---|
| `triage-overview` | `.png` / `.pdf` | README overview |
| `training-dynamics` | `.png` / `.pdf` | README results: entropy, reward, and learner–sampler gap over training |
| `ablation` | `.png` / `.pdf` | README results: Gate/Repair ablations branching from the step-300 checkpoint |

## Characterization (`characterization/`)

Diagnostic figures from the mismatch characterization study; see
[docs/CHARACTERIZATION.md](../../docs/CHARACTERIZATION.md) for the full analysis
and [benchmarks/results/characterization/](../../benchmarks/results/characterization/)
for the underlying data.

| Asset | Contents |
|---|---|
| `directional-asymmetry.png` | Directional mass split and negative-tail depth across healthy → terminal windows |
| `segment-localization.png` | Concentration of amplifying-tail tokens in top segments; response means hiding local tails |
| `response-mean-distributions.png` | Per-response mean-gap magnitude distributions by window |
| `randomization-controls.png` | Observed local-structure statistics against matched null / expectation |
| `full-trajectory.png` | Full training trajectories with the analysis windows in context |

`assets/logo.svg` is the TRIAGE project logo.

Result tables elsewhere in the repository are generated from
[benchmarks/results](../../benchmarks/results) by `scripts/render_results.py`;
do not hand-edit generated blocks.
