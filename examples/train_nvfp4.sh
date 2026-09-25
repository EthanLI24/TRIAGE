#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TRAINING_MODE=nvfp4
source "$REPO_ROOT/examples/common.sh"
triage_train "$@"
