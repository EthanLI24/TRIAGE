# Advantage×gap heatmap gallery

Representative two-dimensional histograms of the joint (advantage A,
learner–sampler gap δ) distribution, from the characterization runs
documented in [CHARACTERIZATION](CHARACTERIZATION.md). In every panel the
horizontal axis is δ on a symmetric-log scale (linear threshold 0.05) and the
vertical axis is the discrete GRPO advantage level; the four corner
percentages are the quadrant shares (top-left A>0, δ<0; top-right A>0, δ>0;
bottom-left A<0, δ<0; bottom-right A<0, δ>0). Mass with δ=0 stays in the
total but is not assigned to any corner, so the corners need not sum to 100%.

Two metrics are shown:

- **tokenmass** — share of the window's valid completion tokens with A≠0 in
  each bin;
- **advmass** — share of the window's total raw |A| weight in each bin.

Color uses one shared LogNorm scale per metric across both models and all
windows (zero cells in flat gray); exact `vmin`/`vmax` are recorded in
[shared_color_scales.json](../assets/figures/heatmaps/shared_color_scales.json).
The 4B panels show the vanilla learner–sampler gap; the 30B panels show the
routing-replay residual. These are different estimands — compare shapes
within a model, not absolute intensities across models.

| Panel | Window (updates) | Metric | One-line summary |
|---|---|---|---|
| [4b_tokenmass_1_50](../assets/figures/heatmaps/4b_tokenmass_1_50.png) | 4B, 1–50 | tokenmass | Early 4B token distribution across the sign quadrants |
| [4b_advmass_1_50](../assets/figures/heatmaps/4b_advmass_1_50.png) | 4B, 1–50 | advmass | Early 4B advantage-weighted mass, near-symmetric at healthy stage |
| [4b_tokenmass_251_324](../assets/figures/heatmaps/4b_tokenmass_251_324.png) | 4B, 251–324 | tokenmass | Terminal-regime 4B tokens with a deepened negative-gap tail |
| [4b_advmass_251_324](../assets/figures/heatmaps/4b_advmass_251_324.png) | 4B, 251–324 | advmass | Terminal 4B: bottom-left (A<0, δ<0) quadrant dominates at 28.96% |
| [30b_tokenmass_601_724](../assets/figures/heatmaps/30b_tokenmass_601_724.png) | 30B, 601–724 | tokenmass | Terminal-regime 30B (routing-replay residual) token distribution |
| [30b_advmass_601_724](../assets/figures/heatmaps/30b_advmass_601_724.png) | 30B, 601–724 | advmass | Terminal 30B: bottom-left quadrant at 25.06% of raw \|A\| weight |

Color bars: [tokenmass](../assets/figures/heatmaps/colorbar_tokenmass.png) ·
[advmass](../assets/figures/heatmaps/colorbar_advmass.png)

The complete source grid covers six windows for 4B (1–50 through 251–324;
50 updates each, except the 74-update final window) and seven windows for
30B (1–100 through 601–724; 100 updates each, except the 124-update final
window), two metrics per window. [panel_index.csv](../assets/figures/heatmaps/panel_index.csv)
is the verbatim index of that source grid (its `basename` column is relative
to the source layout; the six panels mirrored here are flattened into
`assets/figures/heatmaps/`). Panels carry no titles or per-panel colorbars by
design; captions live in this page.
