# ProQuant v2 — Automated Opportunity Scanner & Groww Order Ticket Generator

A hands-free **alert & recommendation system** that scans the market using 4 engines (**Quantitative**, **Fundamental**, **News/Event**, and **Risk Management**), verifies each stock's **2-year historical backtest edge**, and gives you a plain-English **Groww Order Ticket** so you can place the trade yourself in seconds.

## Quick Start (Zero API Keys Needed)
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Configure your capital (optional Telegram alerts):**
   Copy `.env.example` to `.env` and set `PROQUANT_CAPITAL` (e.g. `500000`).
3. **Run a single scan now:**
   ```bash
   python run_daily.py
   ```
   Open `reports/daily_calls_YYYY-MM-DD.html` in your browser to see your **Groww Order Tickets** and **2Y Backtest Edge**.
4. **Or run the continuous opportunity watcher during NSE hours (09:15–15:30 IST):**
   ```bash
   python watch_market.py
   ```
   Checks every 15 minutes and sends a Telegram alert only when a fresh `BUY` or `STRONG BUY` opportunity triggers.
5. **Or use the visual dashboard:**
   ```bash
   streamlit run app.py
   ```
