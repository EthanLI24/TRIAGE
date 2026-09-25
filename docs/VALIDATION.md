# Validation status

This is a living document describing how the repository is validated: what the
automated checks cover, where they run, and what still requires GPU hardware.

## Continuous integration

Every push and pull request runs the CPU checks on GitHub Actions
(`ubuntu-latest`, Python 3.11):

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .
python -c "import TRIAGE, triage_config, triage_calibration"
bash scripts/run_tests.sh
python scripts/render_results.py --check
# plus bash -n over examples/*.sh, scripts/*.sh, and configs/models/*.sh
```

PyTorch is installed from the CPU wheel index first so the package install
treats the `torch` requirement as satisfied without downloading the CUDA
build; the import check verifies the editable install exposes the extension
modules without `PYTHONPATH`.

The CPU test suite mocks the Slime/Megatron interfaces. Coverage includes:

- Loss values, gradients, masks, and partition-invariant token/Repair normalization.
- Gate weights, Repair scopes, and inclusive activation boundaries.
- Frozen calibration, reference-data validation, non-overwriting export, and loading
  profiles produced by the previous calibration command.
- BF16/NVFP4/TRIAGE command construction, actor-environment forwarding, and rejection
  of obsolete configuration names.
- Mode-aware preflight, patch application/idempotence/rejection, overlay hashes,
  and the repository layout contract.

The suite is developed on macOS ARM64 (Python 3.12, PyTorch 2.14.0, NumPy 2.5.3)
and enforced in CI, so these checks must stay green on every change.

## Result tables

All Markdown result tables in [benchmarks/README.md](../benchmarks/README.md) are
generated from the versioned JSON data in `benchmarks/results/` by
`scripts/render_results.py`. CI runs the script with `--check`, so the tables
cannot drift from the recorded data. Measurement protocols (hardware, batch
composition, warm-up, aggregation, repetition counts) are documented alongside
the tables.

## GPU validation status

The published performance values were measured on the pinned training stack
(recorded in [compatibility.json](../compatibility.json) and
[ENVIRONMENT](ENVIRONMENT.md)) on B300 nodes. The automated tests are CPU-only:
numerical and preflight tests mock the Slime/Megatron interfaces, and launcher
tests generate commands without starting Ray. End-to-end GPU training is not
part of CI, because it requires the pinned Slime/SGLang/Megatron stack and
Blackwell-class hardware.

Planned: repeatable end-to-end validation recipes on the pinned stack, including
final-checkpoint evaluation runs for the published benchmark configurations.

Before relying on a local build, inspect the resolved command, then validate at
least one rollout and learner forward/backward update on the target hardware:

```bash
DRY_RUN=1 bash examples/train_triage.sh
```

Actual launches apply only the requested patches and run preflight.
`STRICT_VERSIONS=1` enforces pinned source revisions. On multiple nodes, model,
data, repository, and calibration paths must be accessible at identical locations.
Keep reference export, evaluation, and saving outside timing experiments unless
explicitly included by the measurement protocol.

See [TRAINING](TRAINING.md), [CALIBRATION](CALIBRATION.md), and
[benchmark protocols](../benchmarks/README.md) for the supported workflows.
