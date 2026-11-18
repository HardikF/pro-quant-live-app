import datetime as dt
import json
import os
from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple

import pandas as pd
import streamlit as st

from backtester import BacktestStats, StrategyBacktester
from config import EngineConfig
from engines import FundamentalEngine, FundamentalSignal, NewsEngine, NewsSignal, QuantEngine, QuantSignal

st.set_page_config(page_title="ProQuant Live — Intraday, Delivery & Tracker", layout="wide")

TRACKER_FILE = "data/active_delivery_trades.json"


@dataclass
class StrategyCall:
    symbol: str
    company_name: str
    sector: str
    category: str          # INTRADAY or DELIVERY
    action: str            # STRONG BUY / BUY / WATCH / AVOID
    score: float
    entry_price: float
    stop_loss: float
    target_1: float
    target_2: float
    rr_label: str
    why: str
    warnings: List[str]
    win_rate_pct: float = 0.0
    profit_factor: float = 0.0
    edge_label: str = "N/A"


def load_tracked_trades() -> List[dict]:
    if not os.path.exists(TRACKER_FILE):
        return []
    try:
        with open(TRACKER_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_tracked_trades(trades: List[dict]) -> None:
    os.makedirs(os.path.dirname(TRACKER_FILE), exist_ok=True)
    with open(TRACKER_FILE, "w", encoding="utf-8") as f:
        json.dump(trades, f, indent=2)


def build_intraday_call(q: QuantSignal, f: FundamentalSignal, n: NewsSignal, bt: Optional[BacktestStats]) -> StrategyCall:
    """Strict intraday filter: requires momentum alignment, volume, and clean news."""
    score = round(q.score * 0.70 + n.score * 0.30, 1)
    warnings = ["Intraday trades carry high risk of rapid loss — always use a strict Stop-Loss and square off before 3:15 PM."]

    if n.event_risk_flag:
        warnings.append("Major news/event headline detected — high gap/whipsaw risk.")
    if q.volume_ratio < 1.1:
        warnings.append("Below-average intraday volume — breakout may fail.")

    if score >= 74 and q.ema9 > q.ema21 and n.score >= 45 and not n.event_risk_flag:
        action = "STRONG BUY"
    elif score >= 64 and q.ema9 > q.ema21 and n.score >= 42:
        action = "BUY"
    elif score >= 48:
        action = "WATCH"
    else:
        action = "AVOID"

    entry = q.last_price
    # Tight intraday stop-loss: max(0.6% of price, 0.75 * ATR14)
    sl_dist = max(entry * 0.006, min(q.atr14 * 0.75, entry * 0.015))
    sl = round(entry - sl_dist, 2)
    t1 = round(entry + sl_dist * 1.5, 2)
    t2 = round(entry + sl_dist * 2.5, 2)

    why = f"Quant {q.score}/100 ({q.reasons[0]}); {q.orb_status}; News {n.sentiment_label} ({n.score}/100)."
    return StrategyCall(
        symbol=q.symbol,
        company_name=f.company_name,
        sector=f.sector,
        category="INTRADAY",
        action=action,
        score=score,
        entry_price=entry,
        stop_loss=sl,
        target_1=t1,
        target_2=t2,
        rr_label="1 : 1.5 (T1) / 1 : 2.5 (T2)",
        why=why,
        warnings=warnings,
        win_rate_pct=bt.win_rate_pct if bt else 0.0,
        profit_factor=bt.profit_factor if bt else 0.0,
        edge_label=bt.edge_label if bt else "N/A",
    )


def build_delivery_call(q: QuantSignal, f: FundamentalSignal, n: NewsSignal, bt: Optional[BacktestStats]) -> StrategyCall:
    """Delivery/Swing filter: prioritizes fundamentals, uptrend above EMA50, and avoids falling knives."""
    score = round(f.score * 0.45 + q.score * 0.35 + n.score * 0.20, 1)
    warnings = []

    falling_knife = q.last_price < q.ema50 and q.ema9 < q.ema21
    if falling_knife:
        warnings.append("Downtrend warning (below EMA 21 & 50) — high risk of capital getting stuck.")
    if n.score <= 40:
        warnings.append("Negative news sentiment — wait for news clarity before delivery entry.")
    if n.event_risk_flag:
        warnings.append("Event/earnings headline active — expect short-term volatility.")

    if score >= 70 and not falling_knife and f.score >= 60 and n.score >= 45:
        action = "STRONG BUY"
    elif score >= 60 and not falling_knife and f.score >= 52 and n.score >= 40:
        action = "BUY"
    elif score >= 48:
        action = "WATCH"
    else:
        action = "AVOID"

    entry = q.last_price
    # Wider delivery stop-loss: max(3.5% of price, 1.8 * ATR14), capped at 8% so capital doesn't get trapped
    sl_dist = min(max(entry * 0.035, q.atr14 * 1.8), entry * 0.08)
    sl = round(entry - sl_dist, 2)
    t1 = round(entry + sl_dist * 2.0, 2)
    t2 = round(entry + sl_dist * 3.0, 2)

    why = (
        f"Fundamentals {f.score}/100 ({f.reasons[0]}); "
        f"Trend {q.score}/100 ({q.reasons[0]}); "
        f"News {n.sentiment_label} ({n.score}/100)."
    )
    return StrategyCall(
        symbol=q.symbol,
        company_name=f.company_name,
        sector=f.sector,
        category="DELIVERY",
        action=action,
        score=score,
        entry_price=entry,
        stop_loss=sl,
        target_1=t1,
        target_2=t2,
        rr_label="1 : 2.0 (T1) / 1 : 3.0 (T2)",
        why=why,
        warnings=warnings,
        win_rate_pct=bt.win_rate_pct if bt else 0.0,
        profit_factor=bt.profit_factor if bt else 0.0,
        edge_label=bt.edge_label if bt else "N/A",
    )


def evaluate_tracked_delivery(trade: dict, q: QuantSignal, f: FundamentalSignal, n: NewsSignal) -> Tuple[str, str, float, Optional[float]]:
    """
    Re-evaluates an open delivery trade against live price, trend, and fresh news.
    Returns: (status_signal, reason_text, pnl_pct, suggested_trailing_sl)
    """
    entry = float(trade["entry_price"])
    sl = float(trade["stop_loss"])
    t1 = float(trade["target_1"])
    ltp = q.last_price
    pnl_pct = round(((ltp - entry) / entry) * 100.0, 2) if entry > 0 else 0.0

    if ltp <= sl:
        return ("EXIT NOW — STOP-LOSS HIT", f"Price (₹{ltp}) breached your Stop-Loss (₹{sl}). Exit to prevent deeper capital block.", pnl_pct, None)

    if ltp >= t1:
        trail_sl = round(max(entry * 1.01, ltp - q.atr14 * 1.2), 2)
        return ("BOOK PARTIAL / TRAIL SL", f"Target 1 (₹{t1}) achieved (+{pnl_pct}%). Book partial profit or move Stop-Loss up to ₹{trail_sl}.", pnl_pct, trail_sl)

    if n.score <= 36:
        return ("EXIT EARLY — NEGATIVE NEWS", f"News sentiment turned negative ({n.score}/100). Headlines: {'; '.join(n.top_headlines[:2])}", pnl_pct, None)

    if ltp < q.ema50 and q.ema9 < q.ema21 and pnl_pct < -2.0:
        return ("REVIEW / EXIT — TREND BROKE", f"Stock slipped below EMA 21 & 50 (LTP ₹{ltp}, P&L {pnl_pct}%). Dynamics weakened before SL.", pnl_pct, None)

    if pnl_pct >= 3.5:
        trail_sl = round(max(sl, entry * 1.005), 2)
        return ("HOLD & TRAIL SL TO BREAKEVEN", f"Up +{pnl_pct}% with intact trend. Consider moving Stop-Loss from ₹{sl} to ₹{trail_sl} to make trade risk-free.", pnl_pct, trail_sl)

    return ("HOLD — THESIS INTACT", f"Price ₹{ltp} ({pnl_pct:+.2f}%), Trend & News ({n.sentiment_label}) remain within plan. Keep SL at ₹{sl}.", pnl_pct, None)


@st.cache_data(ttl=600, show_spinner=False)
def scan_watchlist(symbols: Tuple[str, ...]):
    q_eng, f_eng, n_eng, bt_eng = QuantEngine(), FundamentalEngine(), NewsEngine(), StrategyBacktester()
    intraday_calls, delivery_calls = [], []
    raw_map = {}

    for sym in symbols:
        q = q_eng.analyze_from_yfinance(sym)
        f = f_eng.analyze_from_yfinance(sym)
        if q is None or f is None:
            continue
        n = n_eng.analyze(sym, f.company_name)
        bt = bt_eng.run_yfinance(sym, period="2y", rr_ratio=2.0)
        raw_map[sym] = (q, f, n)
        intraday_calls.append(build_intraday_call(q, f, n, bt))
        delivery_calls.append(build_delivery_call(q, f, n, bt))

    intraday_calls.sort(key=lambda c: c.score, reverse=True)
    delivery_calls.sort(key=lambda c: c.score, reverse=True)
    return intraday_calls, delivery_calls, raw_map


def render_call_card(c: StrategyCall, allow_track: bool = False):
    color = "green" if "BUY" in c.action else ("orange" if c.action == "WATCH" else "red")
    with st.container(border=True):
        top = st.columns([3, 1])
        top[0].markdown(f"### {c.symbol.replace('.NS', '')} — :{color}[{c.action}]")
        top[0].caption(f"{c.company_name} · {c.sector} · Score: **{c.score}/100** · R:R `{c.rr_label}`")
        if allow_track and top[1].button("Track Delivery", key=f"track_{c.symbol}_{c.category}"):
            trades = load_tracked_trades()
            trades = [t for t in trades if t["symbol"] != c.symbol]
            trades.append({
                "symbol": c.symbol,
                "company_name": c.company_name,
                "entry_price": c.entry_price,
                "stop_loss": c.stop_loss,
                "target_1": c.target_1,
                "target_2": c.target_2,
                "added_on": dt.date.today().isoformat(),
            })
            save_tracked_trades(trades)
            st.success(f"Added {c.symbol} to Active Delivery Tracker!")

        cols = st.columns(4)
        cols[0].metric("Entry Price", f"₹{c.entry_price:,.2f}")
        cols[1].metric("Strategy Stop-Loss", f"₹{c.stop_loss:,.2f}")
        cols[2].metric("Target 1", f"₹{c.target_1:,.2f}")
        cols[3].metric("Target 2", f"₹{c.target_2:,.2f}")

        st.write(f"**Why:** {c.why}")
        if c.edge_label != "N/A":
            st.caption(f"2Y Backtest Check: **{c.edge_label}** · Win Rate {c.win_rate_pct}% · Profit Factor {c.profit_factor}")
        for w in c.warnings:
            st.warning(w)


# --- UI Header ---
st.title("ProQuant Live — Intraday, Delivery & Active Trade Tracker")
st.info(
    "**Important Risk Disclosure:** No automated scanner or strategy is 100% error-free or guaranteed to make money. "
    "Use this tool for research, paper-trading, and disciplined Stop-Loss planning—not as guaranteed financial advice. "
    "You place and manage every order manually on Groww."
)

cfg = EngineConfig()
with st.sidebar:
    st.header("Scanner Controls")
    custom_tickers = st.text_area(
        "NSE Watchlist (comma-separated, .NS suffix)",
        value=", ".join(cfg.watchlist),
        height=110,
    )
    if st.button("Refresh Live Market Scan", type="primary"):
        st.cache_data.clear()

symbols = tuple(s.strip().upper() for s in custom_tickers.split(",") if s.strip())

with st.spinner("Scanning live NSE prices, fundamentals & news..."):
    intraday_calls, delivery_calls, raw_map = scan_watchlist(symbols)

tab_delivery, tab_intraday, tab_tracker = st.tabs([
    f"Delivery / Swing Calls ({len(delivery_calls)})",
    f"Intraday Calls ({len(intraday_calls)})",
    f"Active Delivery Tracker ({len(load_tracked_trades())})",
])

with tab_delivery:
    st.subheader("Delivery / Positional Calls (Multi-Day to Multi-Week)")
    st.caption("Filters out falling knives (below EMA 21 & 50) and weak fundamentals. Click 'Track Delivery' on any stock you buy so the tracker monitors news & trend changes for you.")
    if not delivery_calls:
        st.warning("No live data returned. Check internet connection or ticker symbols.")
    for call in delivery_calls:
        render_call_card(call, allow_track=True)

with tab_intraday:
    st.subheader("Intraday Calls (Same-Day Only — High Risk)")
    st.caption("Strict momentum + volume + clean-news filter with tight Stop-Loss. Never carry a losing intraday trade overnight.")
    if not intraday_calls:
        st.warning("No live data returned. Check internet connection or ticker symbols.")
    for call in intraday_calls:
        render_call_card(call, allow_track=False)

with tab_tracker:
    st.subheader("Active Delivery Trade Tracker (Live Re-Evaluation)")
    st.caption("Continuously checks your open delivery trades for Stop-Loss hits, Target hits, Trend breaks, or Negative News so your capital doesn't stay trapped.")

    with st.expander("+ Manually Log a Delivery Trade You Bought on Groww"):
        with st.form("add_manual_trade"):
            c1, c2, c3, c4 = st.columns(4)
            m_sym = c1.text_input("NSE Symbol (e.g. SBIN.NS)").strip().upper()
            m_entry = c2.number_input("Your Buy Price (₹)", min_value=1.0, value=1000.0)
            m_sl = c3.number_input("Stop-Loss (₹)", min_value=1.0, value=950.0)
            m_t1 = c4.number_input("Target 1 (₹)", min_value=1.0, value=1100.0)
            if st.form_submit_button("Add to Tracker") and m_sym:
                if not m_sym.endswith(".NS") and not m_sym.endswith(".BO"):
                    m_sym += ".NS"
                trades = load_tracked_trades()
                trades = [t for t in trades if t["symbol"] != m_sym]
                trades.append({
                    "symbol": m_sym,
                    "company_name": m_sym.replace(".NS", ""),
                    "entry_price": m_entry,
                    "stop_loss": m_sl,
                    "target_1": m_t1,
                    "target_2": round(m_t1 * 1.05, 2),
                    "added_on": dt.date.today().isoformat(),
                })
                save_tracked_trades(trades)
                st.rerun()

    tracked = load_tracked_trades()
    if not tracked:
        st.info("No delivery trades being tracked yet. Click **Track Delivery** on any Delivery Call or add one manually above.")
    else:
        q_eng, f_eng, n_eng = QuantEngine(), FundamentalEngine(), NewsEngine()
        for idx, tr in enumerate(tracked):
            sym = tr["symbol"]
            if sym in raw_map:
                q, f, n = raw_map[sym]
            else:
                q = q_eng.analyze_from_yfinance(sym)
                f = f_eng.analyze_from_yfinance(sym)
                n = n_eng.analyze(sym, f.company_name if f else sym) if q else None

            with st.container(border=True):
                if q is None or f is None or n is None:
                    st.warning(f"{sym}: Could not fetch live data right now.")
                    continue
                status, reason, pnl_pct, trail_sl = evaluate_tracked_delivery(tr, q, f, n)
                badge_col = "red" if "EXIT" in status else ("green" if "BOOK" in status or "TRAIL" in status else "blue")
                head = st.columns([4, 1])
                head[0].markdown(f"### {sym.replace('.NS', '')} — :{badge_col}[{status}]")
                head[0].caption(f"Tracked since {tr.get('added_on', '—')} · Live Price: ₹{q.last_price:,.2f} ({pnl_pct:+.2f}%)")
                if head[1].button("Remove", key=f"rm_{sym}_{idx}"):
                    updated = [t for t in tracked if t["symbol"] != sym]
                    save_tracked_trades(updated)
                    st.rerun()

                mcols = st.columns(4)
                mcols[0].metric("Your Entry", f"₹{tr['entry_price']:,.2f}")
                mcols[1].metric("Live LTP", f"₹{q.last_price:,.2f}", f"{pnl_pct:+.2f}%")
                mcols[2].metric("Active Stop-Loss", f"₹{tr['stop_loss']:,.2f}")
                mcols[3].metric("Target 1", f"₹{tr['target_1']:,.2f}")
                st.write(f"**Action Guidance:** {reason}")
                if trail_sl:
                    st.success(f"Suggested New Stop-Loss: ₹{trail_sl:,.2f} (update your Groww GTT order)")
