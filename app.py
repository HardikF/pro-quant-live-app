import concurrent.futures
import datetime as dt
import json
import math
import os
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="ProQuant NSE — Full Market Screener & Tracker", layout="wide")

TRACKER_FILE = "active_delivery_trades.json"

NIFTY_50 = [
    "ADANIENT.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS", "AXISBANK.NS",
    "BAJAJ-AUTO.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", "BEL.NS", "BHARTIARTL.NS",
    "BPCL.NS", "BRITANNIA.NS", "CIPLA.NS", "COALINDIA.NS", "DRREDDY.NS",
    "EICHERMOT.NS", "GRASIM.NS", "HCLTECH.NS", "HDFCBANK.NS", "HDFCLIFE.NS",
    "HEROMOTOCO.NS", "HINDALCO.NS", "HINDUNILVR.NS", "ICICIBANK.NS", "INDUSINDBK.NS",
    "INFY.NS", "ITC.NS", "JSWSTEEL.NS", "KOTAKBANK.NS", "LT.NS",
    "M&M.NS", "MARUTI.NS", "NESTLEIND.NS", "NTPC.NS", "ONGC.NS",
    "POWERGRID.NS", "RELIANCE.NS", "SBILIFE.NS", "SBIN.NS", "SHRIRAMFIN.NS",
    "SUNPHARMA.NS", "TATACONSUM.NS", "TATAMOTORS.NS", "TATASTEEL.NS", "TCS.NS",
    "TECHM.NS", "TITAN.NS", "TRENT.NS", "ULTRACEMCO.NS", "WIPRO.NS",
]

NIFTY_NEXT_50_AND_MIDCAPS = [
    "ABB.NS", "ADANIENSOL.NS", "ADANIGREEN.NS", "ADANIPOWER.NS", "AMBUJACEM.NS",
    "ATGL.NS", "BAJAJHLDNG.NS", "BANKBARODA.NS", "BOSCHLTD.NS", "CANBK.NS",
    "CGPOWER.NS", "CHOLAFIN.NS", "COLPAL.NS", "DABUR.NS", "DIVISLAB.NS",
    "DLF.NS", "DMART.NS", "GAIL.NS", "GODREJCP.NS", "HAL.NS",
    "HAVELLS.NS", "ICICIGI.NS", "ICICIPRULI.NS", "INDIGO.NS", "IOC.NS",
    "IRCTC.NS", "IRFC.NS", "JINDALSTEL.NS", "JIOFIN.NS", "LICI.NS",
    "LODHA.NS", "LTIM.NS", "MARICO.NS", "MOTHERSON.NS", "NAUKRI.NS",
    "NHPC.NS", "PFC.NS", "PIDILITIND.NS", "PNB.NS", "POLYCAB.NS",
    "RECLTD.NS", "SBICARD.NS", "SHREECEM.NS", "SIEMENS.NS", "SRF.NS",
    "TATAPOWER.NS", "TORNTPHARM.NS", "TVSMOTOR.NS", "UNIONBANK.NS", "UNITDSPR.NS",
    "VBL.NS", "VEDL.NS", "ZOMATO.NS", "ZYDUSLIFE.NS", "AUROPHARMA.NS",
    "BHEL.NS", "COFORGE.NS", "CUMMINSIND.NS", "DIXON.NS", "FEDERALBNK.NS",
    "HDFCAMC.NS", "IDFCFIRSTB.NS", "INDUSTOWER.NS", "LUPIN.NS", "MANKIND.NS",
    "MAXHEALTH.NS", "MPHASIS.NS", "MUTHOOTFIN.NS", "OBEROIRLTY.NS", "OFSS.NS",
    "PAGEIND.NS", "PERSISTENT.NS", "PETRONET.NS", "PIIND.NS", "SUZLON.NS",
]

UNIVERSE_PRESETS = {
    "Broad NSE Liquid (125 Stocks — Recommended)": NIFTY_50 + NIFTY_NEXT_50_AND_MIDCAPS,
    "NIFTY 50 Only (50 Large-Caps)": NIFTY_50,
    "NIFTY Next 50 & Top Midcaps (75 Stocks)": NIFTY_NEXT_50_AND_MIDCAPS,
}

POSITIVE_KEYWORDS = {
    "beat": 12, "surge": 10, "upgrade": 12, "outperform": 10, "order win": 14,
    "contract": 10, "expansion": 8, "dividend": 8, "buyback": 12, "profit": 8,
    "record": 10, "approval": 10, "partnership": 8, "rally": 8, "growth": 8,
}

NEGATIVE_KEYWORDS = {
    "miss": -12, "downgrade": -12, "underperform": -10, "sebi": -15, "probe": -15,
    "penalty": -12, "fraud": -20, "loss": -10, "slump": -10, "lawsuit": -12,
    "resign": -10, "default": -18, "recall": -10, "warning": -10, "cut": -8,
}

EVENT_RISK_KEYWORDS = ["earnings", "results", "sebi", "rbi", "investigation", "merger", "split"]


@dataclass
class QuantSignal:
    symbol: str
    last_price: float
    prev_close: float
    change_pct: float
    ema9: float
    ema21: float
    ema50: float
    rsi14: float
    atr14: float
    volume_ratio: float
    score: float
    reasons: List[str]


@dataclass
class FundamentalSignal:
    symbol: str
    company_name: str
    sector: str
    pe_ratio: float
    roe_pct: float
    debt_to_equity: float
    div_yield_pct: float
    revenue_growth_pct: float
    score: float
    reasons: List[str]


@dataclass
class NewsSignal:
    symbol: str
    score: float
    sentiment_label: str
    event_risk_flag: bool
    top_headlines: List[str]


@dataclass
class BacktestStats:
    total_trades: int
    win_rate_pct: float
    profit_factor: float
    net_return_pct: float
    max_drawdown_pct: float
    edge_label: str


@dataclass
class StrategyCall:
    symbol: str
    company_name: str
    sector: str
    category: str
    action: str
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


def _safe_float(val, default: float = 0.0) -> float:
    try:
        f = float(val)
        return f if math.isfinite(f) else default
    except Exception:
        return default


def _clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    if "Close" not in cols or "High" not in cols or "Low" not in cols:
        return pd.DataFrame()
    clean = df[cols].copy()
    for c in cols:
        clean[c] = pd.to_numeric(clean[c], errors="coerce")
    clean = clean.dropna(subset=["Close", "High", "Low"])
    clean = clean[(clean["Close"] > 0) & (clean["High"] > 0) & (clean["Low"] > 0)]
    return clean


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(series: pd.Series, period: int = 14) -> float:
    if len(series) < period + 1:
        return 50.0
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = (-delta.clip(upper=0)).rolling(window=period).mean()
    last_gain = _safe_float(gain.iloc[-1], 0.0)
    last_loss = _safe_float(loss.iloc[-1], 0.0)
    if last_loss == 0.0:
        return 100.0 if last_gain > 0 else 50.0
    rs = last_gain / last_loss
    return float(100.0 - (100.0 / (1.0 + rs)))


def _atr(df: pd.DataFrame, period: int = 14) -> float:
    if len(df) < 2:
        return max(_safe_float(df["High"].iloc[-1] - df["Low"].iloc[-1]), _safe_float(df["Close"].iloc[-1]) * 0.015)
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    val = _safe_float(tr.rolling(window=min(period, len(tr))).mean().iloc[-1], 0.0)
    if val <= 0:
        val = _safe_float(df["Close"].iloc[-1]) * 0.015
    return val


def analyze_quant(symbol: str, daily: pd.DataFrame) -> Optional[QuantSignal]:
    df = _clean_ohlcv(daily)
    if len(df) < 25:
        return None

    close = df["Close"]
    last_p = _safe_float(close.iloc[-1], 0.0)
    if last_p <= 0:
        return None
    prev_c = _safe_float(close.iloc[-2], last_p)
    change_pct = ((last_p - prev_c) / prev_c * 100.0) if prev_c > 0 else 0.0

    ema9 = _safe_float(_ema(close, 9).iloc[-1], last_p)
    ema21 = _safe_float(_ema(close, 21).iloc[-1], last_p)
    ema50 = _safe_float(_ema(close, 50).iloc[-1], last_p)
    rsi14 = _rsi(close, 14)
    atr14 = _atr(df, 14)

    if "Volume" in df.columns:
        avg_vol20 = _safe_float(df["Volume"].tail(20).mean(), 0.0)
        cur_vol = _safe_float(df["Volume"].iloc[-1], 0.0)
        vol_ratio = (cur_vol / avg_vol20) if avg_vol20 > 0 else 1.0
    else:
        vol_ratio = 1.0

    score = 50.0
    reasons = []
    if ema9 > ema21 > ema50 and last_p > ema9:
        score += 20
        reasons.append("Strong bullish trend (Price > EMA 9 > 21 > 50)")
    elif ema9 > ema21:
        score += 11
        reasons.append("Short-term bullish crossover (EMA 9 > 21)")
    elif ema9 < ema21 < ema50:
        score -= 16
        reasons.append("Bearish downtrend (EMA 9 < 21 < 50)")

    if 45 <= rsi14 <= 65:
        score += 12
        reasons.append(f"Healthy momentum RSI ({rsi14:.1f})")
    elif rsi14 < 34:
        score += 13
        reasons.append(f"Oversold RSI ({rsi14:.1f}) rebound setup")
    elif rsi14 > 74:
        score -= 14
        reasons.append(f"Overbought RSI ({rsi14:.1f}) — avoid chasing")

    if vol_ratio >= 1.35:
        score += 12
        reasons.append(f"Institutional volume surge ({vol_ratio:.1f}x 20D avg)")
    elif vol_ratio >= 1.1:
        score += 5
        reasons.append(f"Above-average volume ({vol_ratio:.1f}x)")

    if change_pct > 0.5 and change_pct < 6.0:
        score += 6
        reasons.append(f"Positive price action (+{change_pct:.1f}%)")
    elif change_pct <= -3.0:
        score -= 8
        reasons.append(f"Sharp daily drop ({change_pct:.1f}%)")

    return QuantSignal(
        symbol=symbol.upper(),
        last_price=round(last_p, 2),
        prev_close=round(prev_c, 2),
        change_pct=round(change_pct, 2),
        ema9=round(ema9, 2),
        ema21=round(ema21, 2),
        ema50=round(ema50, 2),
        rsi14=round(rsi14, 1),
        atr14=round(atr14, 2),
        volume_ratio=round(vol_ratio, 2),
        score=float(np.clip(round(score, 1), 5, 98)),
        reasons=reasons or ["Neutral technical consolidation"],
    )


def analyze_fundamentals(symbol: str, info: dict) -> FundamentalSignal:
    name = info.get("shortName") or info.get("longName") or symbol.replace(".NS", "")
    sector = info.get("sector") or "NSE Equity"
    pe = _safe_float(info.get("trailingPE") or info.get("forwardPE"), 0.0)
    roe = _safe_float(info.get("returnOnEquity"), 0.0) * 100.0
    de = _safe_float(info.get("debtToEquity"), 0.0)
    div_y = _safe_float(info.get("dividendYield"), 0.0) * 100.0
    rev_g = _safe_float(info.get("revenueGrowth"), 0.0) * 100.0

    score = 50.0
    reasons = []
    if 0 < pe <= 18:
        score += 18
        reasons.append(f"Attractive P/E ({pe:.1f}x)")
    elif 18 < pe <= 28:
        score += 10
        reasons.append(f"Reasonable P/E ({pe:.1f}x)")
    elif pe > 50:
        score -= 14
        reasons.append(f"Expensive P/E ({pe:.1f}x)")

    if roe >= 18:
        score += 15
        reasons.append(f"Strong ROE ({roe:.1f}%)")
    elif roe >= 12:
        score += 8
        reasons.append(f"Healthy ROE ({roe:.1f}%)")
    elif 0 < roe < 8:
        score -= 8
        reasons.append(f"Weak ROE ({roe:.1f}%)")

    norm_de = de / 100.0 if de > 10 else de
    if 0 <= norm_de <= 0.6:
        score += 8
        reasons.append("Low debt balance sheet")
    elif norm_de > 1.5:
        score -= 10
        reasons.append(f"Higher leverage (D/E {norm_de:.2f})")

    if rev_g >= 12:
        score += 10
        reasons.append(f"Strong revenue growth ({rev_g:.1f}% YoY)")
    elif rev_g < 0:
        score -= 10
        reasons.append(f"Declining revenue ({rev_g:.1f}% YoY)")

    if div_y >= 1.5:
        score += 5
        reasons.append(f"Dividend yield ({div_y:.2f}%)")

    return FundamentalSignal(
        symbol=symbol.upper(),
        company_name=name,
        sector=sector,
        pe_ratio=round(pe, 2),
        roe_pct=round(roe, 1),
        debt_to_equity=round(de, 2),
        div_yield_pct=round(div_y, 2),
        revenue_growth_pct=round(rev_g, 1),
        score=float(np.clip(round(score, 1), 5, 98)),
        reasons=reasons or ["Standard fundamental profile"],
    )


def analyze_news(symbol: str, company_name: str) -> NewsSignal:
    encoded = urllib.parse.quote(f"{company_name or symbol.replace('.NS', '')} NSE stock")
    url = f"https://news.google.com/rss/search?q={encoded}&hl=en-IN&gl=IN&ceid=IN:en"
    headlines = []
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            root = ET.fromstring(resp.read())
        for item in root.findall("./channel/item/title")[:5]:
            if item.text:
                headlines.append(item.text.strip())
    except Exception:
        pass

    score = 50.0
    event_flag = False
    for h in headlines:
        low = h.lower()
        for kw, pts in POSITIVE_KEYWORDS.items():
            if kw in low:
                score += pts
        for kw, pts in NEGATIVE_KEYWORDS.items():
            if kw in low:
                score += pts
        if any(ek in low for ek in EVENT_RISK_KEYWORDS):
            event_flag = True

    score = float(np.clip(round(score, 1), 10, 95))
    label = "Positive Catalyst" if score >= 62 else ("Negative / Caution" if score <= 40 else "Neutral")
    return NewsSignal(symbol=symbol.upper(), score=score, sentiment_label=label, event_risk_flag=event_flag, top_headlines=headlines[:3])


def run_backtest(daily: pd.DataFrame, rr_ratio: float = 2.0) -> BacktestStats:
    df = _clean_ohlcv(daily)
    if len(df) < 40:
        return BacktestStats(0, 0.0, 0.0, 0.0, 0.0, "Insufficient Data")

    close, high, low = df["Close"], df["High"], df["Low"]
    ema9, ema21 = _ema(close, 9), _ema(close, 21)
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rsi14 = (100.0 - (100.0 / (1.0 + (gain / loss.replace(0, np.nan))))).fillna(50.0)
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean().bfill()

    cost = 0.0012
    trades, in_trade = [], False
    entry_p = sl_p = tgt_p = 0.0

    for i in range(25, len(df)):
        c, h, l = float(close.iloc[i]), float(high.iloc[i]), float(low.iloc[i])
        if in_trade:
            if l <= sl_p:
                trades.append(((sl_p - entry_p) / entry_p) - cost)
                in_trade = False
            elif h >= tgt_p:
                trades.append(((tgt_p - entry_p) / entry_p) - cost)
                in_trade = False
            continue
        if float(ema9.iloc[i]) > float(ema21.iloc[i]) and 42.0 <= float(rsi14.iloc[i]) <= 64.0:
            entry_p = c
            stop_dist = max(_safe_float(atr14.iloc[i], entry_p * 0.015) * 1.25, entry_p * 0.012)
            sl_p = entry_p - stop_dist
            tgt_p = entry_p + stop_dist * rr_ratio
            in_trade = True

    if not trades:
        return BacktestStats(0, 0.0, 0.0, 0.0, 0.0, "No Signals")

    arr = np.array(trades)
    wins, losses = arr[arr > 0], arr[arr <= 0]
    win_rate = float(len(wins) / len(arr) * 100.0)
    gp = float(wins.sum()) if len(wins) else 0.0
    gl = float(abs(losses.sum())) if len(losses) else 0.0
    pf = round(gp / gl, 2) if gl > 0 else 9.99
    eq = np.cumprod(1.0 + arr)
    dd = (eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq)
    max_dd = float(abs(dd.min()) * 100.0) if len(dd) else 0.0
    net_ret = float((eq[-1] - 1.0) * 100.0)

    edge = "Verified Positive Edge" if (len(arr) >= 6 and win_rate >= 45.0 and pf >= 1.35) else ("Marginal Edge" if pf >= 1.0 else "Weak Historical Edge")
    return BacktestStats(len(arr), round(win_rate, 1), pf, round(net_ret, 1), round(max_dd, 1), edge)


def build_intraday_call(q: QuantSignal, f: FundamentalSignal, n: NewsSignal, bt: BacktestStats) -> StrategyCall:
    score = round(q.score * 0.72 + n.score * 0.28, 1)
    warnings = ["Intraday trades carry high risk of rapid loss — always set your Stop-Loss immediately and exit before 3:15 PM."]
    if n.event_risk_flag:
        warnings.append("News/event headline active — higher intraday whipsaw risk.")
    if q.volume_ratio < 1.1:
        warnings.append("Volume is below 20-day average — breakout confirmation is weaker.")

    if score >= 74 and q.ema9 > q.ema21 and n.score >= 45 and not n.event_risk_flag:
        action = "STRONG BUY"
    elif score >= 64 and q.ema9 > q.ema21 and n.score >= 42:
        action = "BUY"
    elif score >= 48:
        action = "WATCH"
    else:
        action = "AVOID"

    entry = q.last_price
    sl_dist = max(entry * 0.006, min(q.atr14 * 0.70, entry * 0.015))
    sl = round(entry - sl_dist, 2)
    t1 = round(entry + sl_dist * 1.5, 2)
    t2 = round(entry + sl_dist * 2.5, 2)
    why = f"Quant {q.score}/100 ({q.reasons[0]}); Vol {q.volume_ratio}x; News {n.sentiment_label} ({n.score}/100)."
    return StrategyCall(q.symbol, f.company_name, f.sector, "INTRADAY", action, score, entry, sl, t1, t2, "1 : 1.5 (T1) / 1 : 2.5 (T2)", why, warnings, bt.win_rate_pct, bt.profit_factor, bt.edge_label)


def build_delivery_call(q: QuantSignal, f: FundamentalSignal, n: NewsSignal, bt: BacktestStats) -> StrategyCall:
    score = round(f.score * 0.45 + q.score * 0.35 + n.score * 0.20, 1)
    warnings = []
    falling_knife = q.last_price < q.ema50 and q.ema9 < q.ema21
    if falling_knife:
        warnings.append("Downtrend warning (below EMA 21 & 50) — high risk of capital getting stuck.")
    if n.score <= 40:
        warnings.append("Negative news sentiment — wait for clarity before delivery entry.")
    if n.event_risk_flag:
        warnings.append("Event/earnings headline active — expect short-term swings.")

    if score >= 70 and not falling_knife and f.score >= 60 and n.score >= 45:
        action = "STRONG BUY"
    elif score >= 60 and not falling_knife and f.score >= 52 and n.score >= 40:
        action = "BUY"
    elif score >= 48:
        action = "WATCH"
    else:
        action = "AVOID"

    entry = q.last_price
    sl_dist = min(max(entry * 0.035, q.atr14 * 1.8), entry * 0.08)
    sl = round(entry - sl_dist, 2)
    t1 = round(entry + sl_dist * 2.0, 2)
    t2 = round(entry + sl_dist * 3.0, 2)
    why = f"Fundamentals {f.score}/100 ({f.reasons[0]}); Trend {q.score}/100 ({q.reasons[0]}); News {n.sentiment_label} ({n.score}/100)."
    return StrategyCall(q.symbol, f.company_name, f.sector, "DELIVERY", action, score, entry, sl, t1, t2, "1 : 2.0 (T1) / 1 : 3.0 (T2)", why, warnings, bt.win_rate_pct, bt.profit_factor, bt.edge_label)


def evaluate_tracked_delivery(trade: dict, q: QuantSignal, n: NewsSignal) -> Tuple[str, str, float, Optional[float]]:
    entry, sl, t1, ltp = float(trade["entry_price"]), float(trade["stop_loss"]), float(trade["target_1"]), q.last_price
    pnl_pct = round(((ltp - entry) / entry) * 100.0, 2) if entry > 0 else 0.0

    if ltp <= sl:
        return ("EXIT NOW — STOP-LOSS HIT", f"Price (₹{ltp}) breached Stop-Loss (₹{sl}). Exit to protect capital.", pnl_pct, None)
    if ltp >= t1:
        trail_sl = round(max(entry * 1.01, ltp - q.atr14 * 1.2), 2)
        return ("BOOK PARTIAL / TRAIL SL", f"Target 1 (₹{t1}) hit (+{pnl_pct}%). Book partial profit or trail SL to ₹{trail_sl}.", pnl_pct, trail_sl)
    if n.score <= 36:
        return ("EXIT EARLY — NEGATIVE NEWS", f"News sentiment turned negative ({n.score}/100). Headlines: {'; '.join(n.top_headlines[:2])}", pnl_pct, None)
    if ltp < q.ema50 and q.ema9 < q.ema21 and pnl_pct < -2.0:
        return ("REVIEW / EXIT — TREND BROKE", f"Stock slipped below EMA 21 & 50 (LTP ₹{ltp}, P&L {pnl_pct}%). Dynamics weakened before SL.", pnl_pct, None)
    if pnl_pct >= 3.5:
        trail_sl = round(max(sl, entry * 1.005), 2)
        return ("HOLD & TRAIL SL TO BREAKEVEN", f"Up +{pnl_pct}% with intact trend. Consider raising Stop-Loss from ₹{sl} to ₹{trail_sl}.", pnl_pct, trail_sl)
    return ("HOLD — THESIS INTACT", f"Price ₹{ltp} ({pnl_pct:+.2f}%), Trend & News ({n.sentiment_label}) remain within plan. Keep SL at ₹{sl}.", pnl_pct, None)


def load_tracked_trades() -> List[dict]:
    if "tracked_trades" not in st.session_state:
        if os.path.exists(TRACKER_FILE):
            try:
                with open(TRACKER_FILE, "r", encoding="utf-8") as f:
                    st.session_state["tracked_trades"] = json.load(f)
            except Exception:
                st.session_state["tracked_trades"] = []
        else:
            st.session_state["tracked_trades"] = []
    return st.session_state["tracked_trades"]


def save_tracked_trades(trades: List[dict]) -> None:
    st.session_state["tracked_trades"] = trades
    try:
        with open(TRACKER_FILE, "w", encoding="utf-8") as f:
            json.dump(trades, f, indent=2)
    except Exception:
        pass


@st.cache_data(ttl=900, show_spinner=False)
def scan_market_universe(symbols: Tuple[str, ...], top_n: int = 20):
    """
    2-Stage Institutional Funnel:
    Stage 1: Fast bulk download of entire NSE universe -> Quant screen all stocks.
    Stage 2: Deep Fundamentals + Live News + 2Y Backtest in parallel on the Top N candidates.
    """
    import yfinance as yf

    # Stage 1: Bulk download 1-year daily OHLCV for all tickers in parallel
    data = yf.download(
        list(symbols),
        period="1y",
        interval="1d",
        group_by="ticker",
        threads=True,
        progress=False,
        auto_adjust=False,
    )

    quant_results: List[Tuple[str, QuantSignal, pd.DataFrame]] = []
    is_multi = isinstance(data.columns, pd.MultiIndex)

    for sym in symbols:
        try:
            sub = data[sym] if (is_multi and sym in data.columns.get_level_values(0)) else (data if not is_multi else None)
            if sub is None or sub.empty:
                continue
            q = analyze_quant(sym, sub)
            if q is not None:
                quant_results.append((sym, q, sub))
        except Exception:
            continue

    total_scanned = len(quant_results)
    # Pick top N by Quant score for deep Fundamental + News + Backtest scoring
    quant_results.sort(key=lambda x: x[1].score, reverse=True)
    shortlist = quant_results[:top_n]

    def enrich(item):
        sym, q, sub_df = item
        try:
            info = yf.Ticker(sym).info or {}
        except Exception:
            info = {}
        f = analyze_fundamentals(sym, info)
        n = analyze_news(sym, f.company_name)
        bt = run_backtest(sub_df, rr_ratio=2.0)
        return sym, q, f, n, bt

    intraday_calls, delivery_calls = [], []
    raw_map: Dict[str, Tuple[QuantSignal, FundamentalSignal, NewsSignal]] = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for sym, q, f, n, bt in pool.map(enrich, shortlist):
            raw_map[sym] = (q, f, n)
            intraday_calls.append(build_intraday_call(q, f, n, bt))
            delivery_calls.append(build_delivery_call(q, f, n, bt))

    intraday_calls.sort(key=lambda c: c.score, reverse=True)
    delivery_calls.sort(key=lambda c: c.score, reverse=True)
    return total_scanned, intraday_calls, delivery_calls, raw_map


def render_call_card(c: StrategyCall, allow_track: bool = False):
    color = "green" if "BUY" in c.action else ("orange" if c.action == "WATCH" else "red")
    with st.container(border=True):
        top = st.columns([3, 1])
        top[0].markdown(f"### {c.symbol.replace('.NS', '')} — :{color}[{c.action}]")
        top[0].caption(f"{c.company_name} · {c.sector} · Score: **{c.score}/100** · R:R `{c.rr_label}`")
        if allow_track and top[1].button("Track Delivery", key=f"track_{c.symbol}_{c.category}"):
            trades = [t for t in load_tracked_trades() if t["symbol"] != c.symbol]
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
            st.success(f"Added {c.symbol.replace('.NS', '')} to Active Delivery Tracker!")

        cols = st.columns(4)
        cols[0].metric("Entry Price", f"₹{c.entry_price:,.2f}")
        cols[1].metric("Strategy Stop-Loss", f"₹{c.stop_loss:,.2f}")
        cols[2].metric("Target 1", f"₹{c.target_1:,.2f}")
        cols[3].metric("Target 2", f"₹{c.target_2:,.2f}")

        st.write(f"**Why:** {c.why}")
        if c.edge_label != "N/A":
            st.caption(f"1Y Backtest Check: **{c.edge_label}** · Win Rate {c.win_rate_pct}% · Profit Factor {c.profit_factor}")
        for w in c.warnings:
            st.warning(w)


# --- App UI ---
st.title("ProQuant NSE — Broad Market Screener & Active Trade Tracker")
st.info(
    "**Important Risk Disclosure:** No automated scanner or strategy is 100% error-free or guaranteed to make money. "
    "Use this tool for research, paper-trading, and disciplined Stop-Loss planning—not as guaranteed financial advice. "
    "You place and manage every order manually on Groww."
)

with st.sidebar:
    st.header("Market Universe")
    preset_name = st.selectbox("Select NSE Universe to Scan", list(UNIVERSE_PRESETS.keys()), index=0)
    show_only_qualified = st.checkbox("Show Qualified Picks Only (BUY / STRONG BUY)", value=True)
    extra_custom = st.text_input("Add Extra NSE/BSE Tickers (optional, e.g. ZOMATO.NS, RELIANCE.BO)")
    if st.button("Refresh Full Market Scan", type="primary"):
        st.cache_data.clear()

base_list = list(UNIVERSE_PRESETS[preset_name])
if extra_custom.strip():
    for s in extra_custom.split(","):
        sym = s.strip().upper()
        if sym and sym not in base_list:
            base_list.append(sym)

with st.spinner(f"Scanning {len(base_list)} NSE stocks (Stage 1 Quant -> Stage 2 Fundamentals, News & Backtest)..."):
    total_scanned, intraday_calls, delivery_calls, raw_map = scan_market_universe(tuple(base_list), top_n=20)

deliv_shown = [c for c in delivery_calls if c.action in ("STRONG BUY", "BUY")] if show_only_qualified else delivery_calls
intra_shown = [c for c in intraday_calls if c.action in ("STRONG BUY", "BUY")] if show_only_qualified else intraday_calls

st.caption(f"Scanned **{total_scanned} live NSE stocks** -> Shortlisted top 20 for deep Fundamental + News + Backtest verification.")

tracked_list = load_tracked_trades()
tab_delivery, tab_intraday, tab_tracker = st.tabs([
    f"Delivery Qualified Picks ({len(deliv_shown)})",
    f"Intraday Qualified Picks ({len(intra_shown)})",
    f"Active Delivery Tracker ({len(tracked_list)})",
])

with tab_delivery:
    st.subheader("Qualified Delivery / Swing Picks")
    st.caption("Filtered from the full selected NSE universe. Excludes downtrends (below EMA 21/50), weak fundamentals, and negative news.")
    if not deliv_shown:
        st.info("No stocks currently meet the strict BUY criteria in this market session. Uncheck 'Show Qualified Picks Only' in the sidebar to see WATCH candidates.")
    for call in deliv_shown:
        render_call_card(call, allow_track=True)

with tab_intraday:
    st.subheader("Qualified Intraday Momentum Picks (Same-Day Only)")
    st.caption("Requires strong bullish alignment, clean news, and tight Stop-Loss. Never carry a losing intraday trade overnight.")
    if not intra_shown:
        st.info("No stocks currently meet the strict Intraday BUY criteria. Uncheck 'Show Qualified Picks Only' in the sidebar to see WATCH candidates.")
    for call in intra_shown:
        render_call_card(call, allow_track=False)

with tab_tracker:
    st.subheader("Active Delivery Trade Tracker (Live Re-Evaluation)")
    st.caption("Monitors your open delivery trades for Stop-Loss hits, Target hits, Trend breaks, or Negative News so your capital doesn't stay trapped.")

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
                trades = [t for t in load_tracked_trades() if t["symbol"] != m_sym]
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
        st.info("No delivery trades being tracked yet. Click **Track Delivery** on any Delivery Pick or add one manually above.")
    else:
        import yfinance as yf
        for idx, tr in enumerate(tracked):
            sym = tr["symbol"]
            if sym in raw_map:
                q, f, n = raw_map[sym]
            else:
                try:
                    tkr = yf.Ticker(sym)
                    d = tkr.history(period="6mo", interval="1d")
                    q = analyze_quant(sym, d)
                    f = analyze_fundamentals(sym, tkr.info or {}) if q else None
                    n = analyze_news(sym, f.company_name if f else sym) if q else None
                except Exception:
                    q = f = n = None

            with st.container(border=True):
                if q is None or f is None or n is None:
                    st.warning(f"{sym}: Could not fetch live data right now.")
                    continue
                status, reason, pnl_pct, trail_sl = evaluate_tracked_delivery(tr, q, n)
                badge_col = "red" if "EXIT" in status else ("green" if "BOOK" in status or "TRAIL" in status else "blue")
                head = st.columns([4, 1])
                head[0].markdown(f"### {sym.replace('.NS', '')} — :{badge_col}[{status}]")
                head[0].caption(f"Tracked since {tr.get('added_on', '—')} · Live Price: ₹{q.last_price:,.2f} ({pnl_pct:+.2f}%)")
                if head[1].button("Remove", key=f"rm_{sym}_{idx}"):
                    save_tracked_trades([t for t in tracked if t["symbol"] != sym])
                    st.rerun()

                mcols = st.columns(4)
                mcols[0].metric("Your Entry", f"₹{tr['entry_price']:,.2f}")
                mcols[1].metric("Live LTP", f"₹{q.last_price:,.2f}", f"{pnl_pct:+.2f}%")
                mcols[2].metric("Active Stop-Loss", f"₹{tr['stop_loss']:,.2f}")
                mcols[3].metric("Target 1", f"₹{tr['target_1']:,.2f}")
                st.write(f"**Action Guidance:** {reason}")
                if trail_sl:
                    st.success(f"Suggested New Stop-Loss: ₹{trail_sl:,.2f} (update your Groww GTT order)")
