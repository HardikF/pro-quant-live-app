import argparse
import dataclasses
import datetime as dt
import os
import urllib.parse
import urllib.request
from typing import Dict, List

import pandas as pd

from backtester import BacktestStats, StrategyBacktester
from config import EngineConfig
from engines import FundamentalEngine, NewsEngine, QuantEngine, RiskEngine, TradeCall


def send_telegram_alerts(calls: List[TradeCall], cfg: EngineConfig) -> None:
    if not cfg.telegram_bot_token or not cfg.telegram_chat_id:
        return
    buys = [c for c in calls if c.action in ("STRONG BUY", "BUY")]
    if not buys:
        return
    lines = [f"*ProQuant Opportunity Alert ({dt.datetime.now().strftime('%d %b %H:%M')})*"]
    for c in buys[:5]:
        lines.append(
            f"\n*{c.action}: {c.symbol}* (Score {c.composite_score}/100)\n"
            f"Groww Ticket -> BUY {c.qty} shares @ ₹{c.entry_price}\n"
            f"GTT Stop-Loss: ₹{c.stop_loss} | GTT Target: ₹{c.target_1}\n"
            f"When: {c.when_to_enter} | {c.horizon}\n"
            f"Why: {c.why}"
        )
    msg = "\n".join(lines)
    url = f"https://api.telegram.org/bot{cfg.telegram_bot_token}/sendMessage"
    data = urllib.parse.urlencode(
        {"chat_id": cfg.telegram_chat_id, "text": msg, "parse_mode": "Markdown"}
    ).encode()
    try:
        urllib.request.urlopen(urllib.request.Request(url, data=data), timeout=8)
        print("[OK] Sent Telegram alert.")
    except Exception as exc:
        print(f"[WARN] Telegram alert failed: {exc}")


def write_html_report(calls: List[TradeCall], bt_map: Dict[str, BacktestStats], out_path: str) -> None:
    cards = []
    for c in calls:
        badge_color = (
            "#10b981" if "STRONG BUY" in c.action else
            "#34d399" if c.action == "BUY" else
            "#f59e0b" if "WATCH" in c.action else "#f43f5e"
        )
        bt = bt_map.get(c.symbol)
        bt_html = (
            f"<div class='bt'>2Y Backtest Edge: <strong>{bt.edge_label}</strong> · "
            f"Win Rate {bt.win_rate_pct}% · Profit Factor {bt.profit_factor} · "
            f"Max DD {bt.max_drawdown_pct}% ({bt.total_trades} trades)</div>"
            if bt and bt.total_trades > 0
            else "<div class='bt'>2Y Backtest: Run live mode (`python run_daily.py`) for 2Y historical edge stats.</div>"
        )
        warn_html = (
            "".join(f"<div class='warn'>⚠ {w}</div>" for w in c.risk_warnings)
            if c.risk_warnings else ""
        )
        groww_box = (
            f"<div class='groww'><strong>Groww Order Ticket (Manual Execution):</strong> "
            f"Buy <b>{c.qty}</b> shares of <b>{c.symbol.replace('.NS','')}</b> (Delivery/CNC or Intraday/MIS) "
            f"around <b>₹{c.entry_price:,.2f}</b> · Set GTT Stop-Loss at <b>₹{c.stop_loss:,.2f}</b> "
            f"&amp; GTT Target at <b>₹{c.target_1:,.2f}</b></div>"
            if c.qty > 0 else ""
        )
        cards.append(f"""
        <div class="card">
          <div class="card-top">
            <div>
              <span class="symbol">{c.symbol}</span>
              <span class="name">{c.company_name} · {c.sector}</span>
            </div>
            <span class="badge" style="background:{badge_color}22;color:{badge_color};border:1px solid {badge_color}55;">
              {c.action} · {c.composite_score}/100
            </span>
          </div>
          <p class="why"><strong>Why:</strong> {c.why}</p>
          {bt_html}
          <div class="grid">
            <div><span>When</span><strong>{c.when_to_enter}</strong></div>
            <div><span>How Much</span><strong>{c.qty} shares (₹{c.capital_required:,.0f} · {c.portfolio_pct}%)</strong></div>
            <div><span>Stop Loss</span><strong style="color:#fb7185;">₹{c.stop_loss:,.2f} (Max Loss ₹{c.max_loss_inr:,.0f})</strong></div>
            <div><span>Targets</span><strong style="color:#34d399;">T1 ₹{c.target_1:,.2f} / T2 ₹{c.target_2:,.2f}</strong></div>
          </div>
          {groww_box}
          {warn_html}
        </div>
        """)

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><title>ProQuant Opportunity &amp; Groww Order Cards</title>
<style>
  body {{ font-family: system-ui, sans-serif; background:#020617; color:#e2e8f0; margin:0; padding:24px; }}
  .wrap {{ max-width:980px; margin:auto; }}
  h1 {{ font-size:22px; margin-bottom:4px; }}
  .sub {{ color:#94a3b8; font-size:13px; margin-bottom:20px; }}
  .card {{ background:#0f172a; border:1px solid #1e293b; border-radius:12px; padding:16px; margin-bottom:14px; }}
  .card-top {{ display:flex; justify-content:space-between; align-items:center; gap:12px; flex-wrap:wrap; }}
  .symbol {{ font-size:17px; font-weight:700; color:#fff; margin-right:8px; }}
  .name {{ font-size:12px; color:#94a3b8; }}
  .badge {{ font-size:12px; font-weight:700; padding:4px 10px; border-radius:999px; }}
  .why {{ font-size:13px; color:#cbd5e1; line-height:1.5; margin:10px 0 6px; }}
  .bt {{ font-size:12px; color:#38bdf8; margin-bottom:10px; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:10px; background:#020617; padding:12px; border-radius:8px; }}
  .grid span {{ display:block; font-size:11px; color:#64748b; margin-bottom:2px; }}
  .grid strong {{ font-size:13px; }}
  .groww {{ margin-top:10px; padding:10px 12px; border-radius:8px; background:#10b98115; border:1px solid #10b98140; font-size:12px; color:#a7f3d0; }}
  .warn {{ margin-top:8px; font-size:12px; color:#fbbf24; }}
</style></head>
<body><div class="wrap">
  <h1>ProQuant Opportunity Scanner &amp; Groww Order Cards</h1>
  <div class="sub">Generated: {dt.datetime.now().strftime('%Y-%m-%d %H:%M')} · Review each call &amp; backtest edge before placing manual orders on Groww</div>
  {''.join(cards)}
</div></body></html>"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)


def run_pipeline(csv_path: str = "", offline_news: bool = False, notify: bool = True) -> List[TradeCall]:
    cfg = EngineConfig()
    q_eng = QuantEngine()
    f_eng = FundamentalEngine()
    n_eng = NewsEngine()
    r_eng = RiskEngine(cfg)
    bt_eng = StrategyBacktester()

    analyzed_items = []
    bt_map: Dict[str, BacktestStats] = {}

    if csv_path:
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            q_sig = q_eng.analyze_from_row(row)
            f_sig = f_eng.analyze_from_row(row)
            headlines = [str(row["headline"])] if "headline" in row and pd.notna(row["headline"]) else []
            n_sig = n_eng.analyze(
                q_sig.symbol,
                f_sig.company_name,
                offline_headlines=headlines if offline_news else None,
            )
            analyzed_items.append((q_sig, f_sig, n_sig))
    else:
        for sym in cfg.watchlist:
            q_sig = q_eng.analyze_from_yfinance(sym)
            f_sig = f_eng.analyze_from_yfinance(sym)
            if q_sig is None or f_sig is None:
                print(f"[SKIP] Could not fetch live data for {sym}")
                continue
            n_sig = n_eng.analyze(sym, f_sig.company_name)
            bt_map[q_sig.symbol] = bt_eng.run_yfinance(sym, period="2y", rr_ratio=cfg.target_rr_ratio)
            analyzed_items.append((q_sig, f_sig, n_sig))

    calls = r_eng.generate_calls(analyzed_items)

    os.makedirs("reports", exist_ok=True)
    stamp = dt.date.today().isoformat()
    csv_out = os.path.join("reports", f"daily_calls_{stamp}.csv")
    html_out = os.path.join("reports", f"daily_calls_{stamp}.html")

    pd.DataFrame([dataclasses.asdict(c) for c in calls]).to_csv(csv_out, index=False)
    write_html_report(calls, bt_map, html_out)

    print(f"\n=== ProQuant Opportunity Calls ({stamp}) ===")
    for c in calls:
        print(
            f"[{c.action}] {c.symbol} | Score: {c.composite_score}/100 | "
            f"Entry: ₹{c.entry_price} | SL: ₹{c.stop_loss} | T1: ₹{c.target_1} | Qty: {c.qty}"
        )
    print(f"\nSaved: {csv_out} and {html_out}")

    if notify:
        send_telegram_alerts(calls, cfg)

    return calls


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ProQuant 4-Engine Opportunity & Groww Call Generator")
    parser.add_argument("--csv", default="", help="Path to daily CSV snapshot (optional fallback)")
    parser.add_argument("--offline-news", action="store_true", help="Use CSV headline column instead of live RSS")
    args = parser.parse_args()
    run_pipeline(csv_path=args.csv, offline_news=args.offline_news)
