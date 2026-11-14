#!/usr/bin/env bash
# Defaults: TIS, Q3 Gate, positive-advantage Repair from step 0; recentering off.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TRAINING_MODE=triage
source "$REPO_ROOT/examples/common.sh"
triage_train "$@"
