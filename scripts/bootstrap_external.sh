#!/usr/bin/env bash
# Clone the pinned public reference stack into external/.
#
# Reads repository URLs and pinned commits from compatibility.json, so the
# manifest stays the single source of truth. Safe to re-run: checkouts that
# already sit at the pinned commit are left untouched.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
EXTERNAL_DIR="$REPO_ROOT/external"
MANIFEST="$REPO_ROOT/compatibility.json"
mkdir -p "$EXTERNAL_DIR"

# key<TAB>url<TAB>commit<TAB>directory for every public repo in the manifest.
# Empty fields print as '-' because `read` with IFS=$'\t' collapses
# consecutive tabs (tab is IFS whitespace), which would shift the remaining
# fields left; the loop below maps '-' back to an empty string.
ENTRIES="$(python3 - "$MANIFEST" <<'PY'
import json
import sys

def row(*fields):
    print("\t".join(field if field else "-" for field in fields))

stack = json.loads(open(sys.argv[1]).read())["reference_stack"]
directories = {"slime": "slime", "sglang": "sglang", "megatron_lm": "Megatron-LM"}
for key, directory in directories.items():
    entry = stack[key]
    row(key, entry["url"], entry["commit"], directory)
internal = stack.get("mini_transformer", {})
row("mini_transformer", internal.get("url") or "", internal.get("commit", ""), "miniTransformer")
PY
)"

failures=0
while IFS=$'\t' read -r key url commit directory; do
  # '-' is the manifest printer's placeholder for an empty field.
  [[ "$url" != "-" ]] || url=""
  [[ "$commit" != "-" ]] || commit=""
  dest="$EXTERNAL_DIR/$directory"
  if [[ -z "$url" ]]; then
    cat <<EOF
NOTE: $key is an internal release with no public repository URL.
  It is required only for MoE NVFP4 modes (SHOULD_REPLACE_TE_GROUPLINEAR=1).
  Place a checkout at:
    $dest
  pinned to commit ${commit:-unknown}.
EOF
    continue
  fi
  if [[ -d "$dest/.git" ]]; then
    current="$(git -C "$dest" rev-parse HEAD)"
    if [[ "$current" == "$commit" ]]; then
      echo "OK: $directory already at pinned commit $commit"
      continue
    fi
    echo "Updating $directory from $current to pinned $commit"
  elif [[ -e "$dest" ]]; then
    echo "ERROR: $dest exists but is not a git checkout; move it aside or fix it manually." >&2
    failures=1
    continue
  else
    echo "Cloning $url into $dest"
    git clone "$url" "$dest"
  fi
  if ! git -C "$dest" checkout "$commit"; then
    echo "ERROR: could not check out $commit in $dest (dirty tree or missing object)." >&2
    failures=1
  fi
done <<< "$ENTRIES"

if [[ "$failures" != 0 ]]; then
  echo "Bootstrap incomplete; see errors above." >&2
  exit 1
fi
cat <<'EOF'
Bootstrap done. Next: point the launchers at your local assets, e.g.
  export MODEL_ROOT=/path/to/models
  export PROMPT_DATA=/path/to/dapo-math-17k.jsonl
then inspect the resolved command (patching and preflight run automatically
on a real launch):
  DRY_RUN=1 bash examples/train_triage.sh
EOF
