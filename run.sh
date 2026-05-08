#!/bin/bash
# Sahl UFC Predictor - One-Click Bootstrap
# Runs: install deps → build features → train model → start server
# Usage:  ./run.sh           (synthetic data, fast)
#         ./run.sh --real    (scrape ufcstats.com, ~1 hour)

set -e
cd "$(dirname "$0")"

echo "=== Sahl UFC Predictor ==="
echo "Installing dependencies..."
pip install -q -r requirements.txt --break-system-packages 2>/dev/null || \
  pip install -q -r requirements.txt

cd backend

if [ "$1" = "--real" ]; then
  echo
  echo "[1/3] Scraping ufcstats.com (last 10 years, ~1 hour)..."
  python scraper.py --output ../data/ufc_fights_raw.csv --years 10
  echo
  echo "[2/3] Building features..."
  python features.py
else
  echo
  echo "[1/2] Generating synthetic data + features..."
  python features.py --synthetic
fi

echo
echo "[Training] XGBoost + calibration + SHAP..."
python train.py

echo
echo "==============================================="
echo "  Sahl UFC Predictor is starting at:"
echo "    http://localhost:8000"
echo "  Press Ctrl+C to stop."
echo "==============================================="
echo
python server.py
