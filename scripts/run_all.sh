#!/usr/bin/env bash
# Run the full DamageNet-VGG16 pipeline from the project root.
# Usage: scripts/run_all.sh [--debug] [--skip-data]
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -x .venv/bin/python ]; then PY=.venv/bin/python; else PY=python; fi
export HF_HUB_DISABLE_PROGRESS_BARS=1
"$PY" -m src.run_all "$@"
