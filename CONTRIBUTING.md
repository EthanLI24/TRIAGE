# Contributing

Thanks for your interest in TRIAGE. This file describes the contribution
workflow and what is currently in scope.

Install the local package in a Python 3.10+ environment with PyTorch and NumPy:

```bash
python -m pip install -e .
bash scripts/run_tests.sh
python scripts/render_results.py --check
git diff --check
```

CPU tests mock the Slime/Megatron interfaces. GPU integration must be checked in
the target training image; record its revisions, hardware, mode, and command.
Do not describe a dry-run or mocked test as successful GPU training.

## What we welcome / what's currently out of scope

| What we welcome | What's currently out of scope |
|---|---|
| Bug reports with a reproducer and environment details | GPU end-to-end reproduction without the pinned stack (see [compatibility.json](compatibility.json)) |
| Fixes and new cases for the CPU test suite | Supporting upstream versions that diverge from the pinned Slime/SGLang/Megatron revisions |
| Documentation corrections and clarifications | Landing breaking changes to public launcher/config interfaces without discussion |
| New calibration reference data with documented collection protocols | Redistributing model weights, datasets, or third-party runtime sources |

## Guidelines

- Keep shared launch behavior in `examples/common.sh`; mode files only select modes.
- Keep algorithm defaults in `policy_extension/triage_config.py`. Environment
  overrides must reach remote actors through the resolved training environment.
- Add behavior-focused tests for masking, gradients, distributed normalization,
  calibration state, or command construction when changing those interfaces.
- Update `compatibility.json` hashes when intentionally changing a pinned overlay
  file. Upstream patch anchors must remain idempotent and reject unknown source.
- Keep generated outputs, local models, datasets, caches, and credentials outside
  version control. Use repository-relative defaults or documented environment paths.
- Add measured results to `benchmarks/results/` with units, workload, aggregation,
  sample count, and source. Missing values stay `null`; a different measurement
  protocol is a separate result, not an interchangeable replacement.

## License

TRIAGE's own code is licensed under the Apache License, Version 2.0; see
[LICENSE](LICENSE). By contributing, you agree that your contributions are
submitted under the same license. [NOTICE](NOTICE.md) describes the included
code and external components. Do not infer an upstream license for this overlay
or bundle third-party runtime sources without their own notices.
