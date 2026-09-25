#!/usr/bin/env bash
# Apply the TRIAGE source patches to the pinned Slime/SGLang checkouts.
#
# Usage: bash scripts/apply_patches.sh [r3|tokenmean|all]...   (default: all)
#
# SLIME_ROOT and SGLANG_ROOT select the target checkouts and default to
# external/slime and external/sglang under this repository, matching
# scripts/preflight.py and examples/common.sh.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SLIME_ROOT="${SLIME_ROOT:-$REPO_ROOT/external/slime}"
SGLANG_ROOT="${SGLANG_ROOT:-$REPO_ROOT/external/sglang}"

require_dir() {
  if [[ ! -d "$1" ]]; then
    echo "Missing $2 checkout: $1" >&2
    echo "Clone the pinned stack with scripts/bootstrap_external.sh, or set $2 to an existing checkout." >&2
    exit 1
  fi
}

apply_r3() {
  require_dir "$SLIME_ROOT" SLIME_ROOT
  require_dir "$SGLANG_ROOT" SGLANG_ROOT
  python3 - \
    "$REPO_ROOT/patches/slime/patch_routing_replay_indices.py" \
    "$SLIME_ROOT/slime/utils/routing_replay.py" <<'PY'
import importlib.util
import pathlib
import sys

patcher_path = pathlib.Path(sys.argv[1])
target_path = pathlib.Path(sys.argv[2])
spec = importlib.util.spec_from_file_location("patch_routing_replay_indices", patcher_path)
if spec is None or spec.loader is None:
    raise SystemExit(f"cannot load {patcher_path}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.TARGET = target_path
module.main()
PY
  python3 "$REPO_ROOT/patches/sglang/patch_sglang_r3_force_return.py" \
    "$SGLANG_ROOT/python/sglang/srt/managers/scheduler.py"
  python3 "$REPO_ROOT/patches/sglang/patch_sglang_r3_bypassed_topk.py" \
    "$SGLANG_ROOT/python/sglang/srt/layers/moe/topk.py"
  python3 "$REPO_ROOT/patches/slime/patch_slime_r3_route_validation.py" \
    "$SLIME_ROOT/slime/rollout/sglang_rollout.py"
  echo "R3_PATCHES_OK slime=$SLIME_ROOT sglang=$SGLANG_ROOT"
}

apply_tokenmean() {
  require_dir "$SLIME_ROOT" SLIME_ROOT
  python3 "$REPO_ROOT/patches/slime/patch_tokenmean_custom_normalizer.py" \
    --slime-root "$SLIME_ROOT"
}

if [[ $# -eq 0 ]]; then
  set -- all
fi
for patch_set in "$@"; do
  case "$patch_set" in
    r3) apply_r3 ;;
    tokenmean) apply_tokenmean ;;
    all) apply_r3; apply_tokenmean ;;
    *) echo "Unknown patch set: $patch_set (expected r3, tokenmean, or all)" >&2; exit 2 ;;
  esac
done
