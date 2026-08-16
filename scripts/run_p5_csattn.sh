#!/bin/bash
set -euo pipefail
cd /workspace
unset MODAL_TOKEN_ID MODAL_TOKEN_SECRET
export MODAL_PROFILE=uelmann
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1
ARGS=("$@")
echo "[run] $(date -u +%Y-%m-%dT%H:%M:%SZ) starting modal run btcb_phase5_pipeline.py profile=$MODAL_PROFILE args=${ARGS[*]:-}"
exec modal run --detach btcb_phase5_pipeline.py "${ARGS[@]}"
