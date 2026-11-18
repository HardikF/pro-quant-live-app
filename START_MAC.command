#!/bin/bash
cd "$(dirname "$0")"
echo "============================================"
echo " ProQuant One-Click Setup & Daily Scanner"
echo "============================================"

if ! command -v python3 &> /dev/null; then
  echo "[ERROR] Python 3 not found. Install from https://www.python.org/downloads/"
  read -p "Press Enter to exit..."
  exit 1
fi

if [ ! -f ".env" ]; then
  cp .env.example .env
  echo "[OK] Created default .env configuration (Capital: Rs 5,00,000)."
fi

echo "[1/3] Installing required packages..."
python3 -m pip install --quiet -r requirements.txt

echo "[2/3] Running today's 4-engine market scan..."
python3 run_daily.py

echo "[3/3] Opening your Groww Call Cards in browser..."
LATEST_HTML=$(ls -t reports/daily_calls_*.html 2>/dev/null | head -n 1)
if [ -n "$LATEST_HTML" ]; then
  open "$LATEST_HTML"
fi

echo "Done! Review the browser report before placing any manual order on Groww."
read -p "Press Enter to close..."
