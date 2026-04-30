#!/usr/bin/env bash
# Minimal rerun after changing the chat model (taxonomy stays frozen).
# 1) Backs up existing coded tasks
# 2) Full O*NET recode (do not pass --resume)
# 3) Rebuilds comparison + coverage artifacts
#
# Usage:
#   export OPENAI_API_KEY=...
#   bash scripts/rerun_after_chat_model_change.sh
#
# Optional: WORKERS=20 bash scripts/rerun_after_chat_model_change.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -z "${OPENAI_API_KEY:-}" && -z "${OPENAI_API:-}" ]]; then
  echo "Set OPENAI_API_KEY (or OPENAI_API) first."
  exit 1
fi

WORKERS="${WORKERS:-10}"
stamp="$(date +%Y%m%d_%H%M%S)"
if [[ -f data/derived/onet_tasks_coded.parquet ]]; then
  cp -v data/derived/onet_tasks_coded.parquet \
    "data/derived/onet_tasks_coded.parquet.bak_${stamp}"
fi

python3 scripts/code_onet_tasks.py --workers "${WORKERS}"
python3 scripts/build_comparison.py
python3 scripts/plot_coverage.py

echo "Done. Pin the exact chat model snapshot in a new file under docs/freeze_log/ if results feed the paper."
