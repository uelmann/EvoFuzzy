#!/usr/bin/env bash
# Wait ONLY for last_official_summary.json (written at end of FT), then test.
set -euo pipefail
cd /workspace
LOG=/tmp/overnight_base_pipeline.log
exec > >(tee -a "$LOG") 2>&1

echo "=== $(date -u) overnight watcher v2 start ==="
SUM_MARK="official_runs_base/last_official_summary.json"

n=0
while true; do
  n=$((n+1))
  if modal volume ls kronos-crypto-data "$SUM_MARK" >/dev/null 2>&1; then
    echo "$(date -u) FOUND FT summary — training complete"
    break
  fi
  if (( n % 5 == 0 )); then
    echo "$(date -u) waiting for FT summary (poll #$n)"
    rg -n "\[pred\] epoch .* val=|Predictor done|RemoteError|App completed" /tmp/modal_official_base_ft_resume2.log 2>/dev/null | tail -3 || true
    modal app list 2>/dev/null | head -8 || true
  fi
  sleep 120
done

echo "=== $(date -u) download FT summary ==="
modal volume get kronos-crypto-data official_runs_base/last_official_summary.json \
  kronos_signal/last_official_base.json --force
cp -f kronos_signal/last_official_base.json /opt/cursor/artifacts/last_official_base.json
python3 - <<'PY'
import json
from pathlib import Path
d=json.loads(Path('kronos_signal/last_official_base.json').read_text())
print('dataset', d.get('dataset_meta'))
print('train keys', list((d.get('train') or {}).keys()))
print('roc', d.get('topk_roc_baseline'))
print('paths', d.get('paths'))
PY

echo "=== $(date -u) official_bt scores 90/10 ==="
modal run kronos_signal/modal_app.py --mode official_bt --predictor-size base --signal mean --lookback 90 --pred-len 10 \
  2>&1 | tee /tmp/modal_base_bt.log | tail -n 100

echo "=== $(date -u) fetch scores + panel ==="
modal volume get kronos-crypto-data official_runs_base/ft_prediction_scores_lb90_h10.pkl \
  /opt/cursor/artifacts/crypto_data/base_ft_prediction_scores_lb90_h10.pkl --force
modal volume get kronos-crypto-data official_runs_base/ft_prediction_scores.pkl \
  /opt/cursor/artifacts/crypto_data/base_ft_prediction_scores.pkl --force || true
modal volume get kronos-crypto-data official_runs_base/last_official_ft_bt_lb90_h10.json \
  kronos_signal/last_official_ft_bt_base_lb90_h10.json --force || true
modal volume get kronos-crypto-data official_runs_base/processed_datasets/full_panel.pkl \
  /opt/cursor/artifacts/crypto_data/full_panel_base.pkl --force

SCORE_PKL=/opt/cursor/artifacts/crypto_data/base_ft_prediction_scores_lb90_h10.pkl
[[ -f $SCORE_PKL ]] || SCORE_PKL=/opt/cursor/artifacts/crypto_data/base_ft_prediction_scores.pkl

echo "=== $(date -u) L3S3 ==="
ZS_ARG=()
[[ -f /opt/cursor/artifacts/crypto_data/zs_prediction_scores_lb90_h10.pkl ]] && \
  ZS_ARG=(--zs-scores /opt/cursor/artifacts/crypto_data/zs_prediction_scores_lb90_h10.pkl)

python3 -m kronos_signal.run_ft_l3s3 \
  --scores "$SCORE_PKL" \
  "${ZS_ARG[@]}" \
  --panel-pkl /opt/cursor/artifacts/crypto_data/full_panel_base.pkl \
  --long-n 3 --short-n 3 --lookback 90 --pred-len 10 --roc-window 10 \
  --signals last,mean \
  --start 2024-07-01 --end 2026-08-08 \
  --out kronos_signal/last_ft_base_l3s3.json \
  --plot kronos_signal/ft_base_l3s3_equity.png

cp -f kronos_signal/ft_base_l3s3_equity.png /opt/cursor/artifacts/ft_base_l3s3_equity.png
cp -f kronos_signal/last_ft_base_l3s3.json /opt/cursor/artifacts/last_ft_base_l3s3.json

echo "=== $(date -u) git ==="
git add kronos_signal/last_official_base.json \
  kronos_signal/last_ft_base_l3s3.json \
  kronos_signal/ft_base_l3s3_equity.png \
  kronos_signal/last_official_ft_bt_base_lb90_h10.json \
  scripts/overnight_base_pipeline.sh \
  2>/dev/null || true
git status --short | head -30
git commit -m "$(cat <<'EOF'
Complete Kronos-base FT on full KuCoin 2016+ panel and L3/S3 test.

Train 2016-2022; OOS L3/S3 lookback=90 hold=10 vs ROC (and ZS if present).
EOF
)" || echo "nothing to commit"
git push -u origin cursor/kronos-btc-daily-signal-a189 || true

pkill -f 'modal run' 2>/dev/null || true
echo "=== $(date -u) overnight pipeline DONE ==="
