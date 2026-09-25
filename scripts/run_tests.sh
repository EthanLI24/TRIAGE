#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$REPO_ROOT/policy_extension:${PYTHONPATH:-}"
python3 -m unittest discover -s "$REPO_ROOT/tests" -p 'test_*.py' -v
