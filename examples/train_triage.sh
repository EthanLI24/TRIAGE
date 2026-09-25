#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TRAINING_MODE=triage
source "$REPO_ROOT/examples/common.sh"
triage_train "$@"
