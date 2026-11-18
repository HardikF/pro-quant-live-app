from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import pandas as pd


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
    orb_high: float
    orb_low: float
    orb_status: str
    score: float
    horizon: str
    entry_trigger: str
    reasons: List[str]


class QuantEngine:
    """Computes technical trend, momentum, RSI, ATR, volume surge, and ORB signals."""

    @staticmethod
    def _ema(series: pd.Series, span: int) -> pd.Series:
        return series.ewm(span=span, adjust=False).mean()

    @staticmethod
    def _rsi(series: pd.Series, period: int = 14) -> float:
        if len(series) < period + 1:
            return 50.0
        delta = series.diff()
        gain = delta.clip(lower=0).rolling(window=period).mean()
        loss = (-delta.clip(upper=0)).rolling(window=period).mean()
        last_loss = float(loss.iloc[-1])
        if last_loss == 0:
            return 100.0
        rs = float(gain.iloc[-1]) / last_loss
        return float(100.0 - (100.0 / (1.0 + rs)))

    @staticmethod
    def _atr(df: pd.DataFrame, period: int = 14) -> float:
        if len(df) < 2:
            return float(df["High"].iloc[-1] - df["Low"].iloc[-1])
        high_low = df["High"] - df["Low"]
        high_close = (df["High"] - df["Close"].shift()).abs()
        low_close = (df["Low"] - df["Close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr_val = tr.rolling(window=min(period, len(tr))).mean().iloc[-1]
        return float(atr_val) if pd.notna(atr_val) else float(high_low.iloc[-1])

    def analyze_from_yfinance(self, symbol: str) -> Optional[QuantSignal]:
        try:
            import yfinance as yf
        except ImportError:
            return None

        ticker = yf.Ticker(symbol)
        daily = ticker.history(period="6mo", interval="1d")
        if daily is None or daily.empty or len(daily) < 5:
            return None

        intraday = ticker.history(period="1d", interval="15m")
        return self.score_dataframe(symbol, daily, intraday)

    def analyze_from_row(self, row: pd.Series) -> QuantSignal:
        """Fallback scorer when only a daily CSV snapshot row is provided."""
        last_p = float(row["last"])
        prev_c = float(row.get("prev_close", last_p))
        high_p = float(row.get("high", last_p * 1.01))
        low_p = float(row.get("low", last_p * 0.99))
        open_p = float(row.get("open", prev_c))
        rsi_val = float(row.get("rsi14", 50.0))
        ema20 = float(row.get("ema20", last_p))
        ema50 = float(row.get("ema50", last_p))
        vol_ratio = float(row.get("volume_ratio", 1.0))

        change_pct = ((last_p - prev_c) / prev_c * 100.0) if prev_c > 0 else 0.0
        atr14 = max(high_p - low_p, last_p * 0.012)

        score = 50.0
        reasons = []
        if last_p > ema20 > ema50:
            score += 18
            reasons.append("Price in bullish uptrend above 20 & 50 EMA")
        elif last_p < ema20 < ema50:
            score -= 15
            reasons.append("Price trading below 20 & 50 EMA")

        if 40 <= rsi_val <= 62:
            score += 12
            reasons.append(f"Healthy RSI ({rsi_val:.1f}) with room to run")
        elif rsi_val < 35:
            score += 14
            reasons.append(f"Oversold RSI ({rsi_val:.1f}) bounce setup")
        elif rsi_val > 72:
            score -= 14
            reasons.append(f"Overbought RSI ({rsi_val:.1f}) — pullback risk")

        if vol_ratio >= 1.3:
            score += 10
            reasons.append(f"Strong volume surge ({vol_ratio:.1f}x avg)")

        orb_status = "Above Open" if last_p > open_p else "Below Open"
        if last_p > open_p and change_pct > 0:
            score += 6
            reasons.append("Positive intraday momentum")

        score = float(np.clip(round(score, 1), 5, 98))
        horizon = "Intraday / 1-3 Days" if vol_ratio >= 1.3 else "Swing (1-4 Weeks)"
        trigger = f"Buy above ₹{high_p * 0.999:.2f}" if score >= 60 else f"Wait for close above ₹{ema20:.2f}"

        return QuantSignal(
            symbol=str(row["symbol"]).upper(),
            last_price=round(last_p, 2),
            prev_close=round(prev_c, 2),
            change_pct=round(change_pct, 2),
            ema9=round(last_p, 2),
            ema21=round(ema20, 2),
            ema50=round(ema50, 2),
            rsi14=round(rsi_val, 1),
            atr14=round(atr14, 2),
            volume_ratio=round(vol_ratio, 2),
            orb_high=round(high_p, 2),
            orb_low=round(low_p, 2),
            orb_status=orb_status,
            score=score,
            horizon=horizon,
            entry_trigger=trigger,
            reasons=reasons or ["Neutral technical consolidation"],
        )

    def score_dataframe(
        self, symbol: str, daily: pd.DataFrame, intraday: Optional[pd.DataFrame] = None
    ) -> QuantSignal:
        close = daily["Close"]
        last_p = float(close.iloc[-1])
        prev_c = float(close.iloc[-2]) if len(close) >= 2 else last_p
        change_pct = ((last_p - prev_c) / prev_c * 100.0) if prev_c > 0 else 0.0

        ema9 = float(self._ema(close, 9).iloc[-1])
        ema21 = float(self._ema(close, 21).iloc[-1])
        ema50 = float(self._ema(close, 50).iloc[-1])
        rsi14 = self._rsi(close, 14)
        atr14 = self._atr(daily, 14)

        avg_vol20 = float(daily["Volume"].tail(20).mean()) if "Volume" in daily else 1.0
        cur_vol = float(daily["Volume"].iloc[-1]) if "Volume" in daily else 1.0
        vol_ratio = (cur_vol / avg_vol20) if avg_vol20 > 0 else 1.0

        if intraday is not None and not intraday.empty:
            orb_high = float(intraday["High"].iloc[0])
            orb_low = float(intraday["Low"].iloc[0])
            if last_p > orb_high:
                orb_status = "Above 15m ORB High"
            elif last_p < orb_low:
                orb_status = "Below 15m ORB Low"
            else:
                orb_status = "Inside 15m ORB"
        else:
            orb_high = float(daily["High"].iloc[-1])
            orb_low = float(daily["Low"].iloc[-1])
            orb_status = "Daily Range"

        score = 50.0
        reasons = []

        if ema9 > ema21 > ema50:
            score += 18
            reasons.append("Strong bullish alignment (EMA 9 > 21 > 50)")
        elif ema9 > ema21:
            score += 10
            reasons.append("Short-term bullish crossover (EMA 9 > 21)")
        elif ema9 < ema21 < ema50:
            score -= 15
            reasons.append("Bearish trend (EMA 9 < 21 < 50)")

        if 42 <= rsi14 <= 64:
            score += 10
            reasons.append(f"Healthy momentum RSI ({rsi14:.1f})")
        elif rsi14 < 35:
            score += 14
            reasons.append(f"Oversold RSI ({rsi14:.1f}) mean-reversion setup")
        elif rsi14 > 72:
            score -= 12
            reasons.append(f"Overbought RSI ({rsi14:.1f}) — avoid chasing")

        if vol_ratio >= 1.3:
            score += 10
            reasons.append(f"Institutional volume surge ({vol_ratio:.1f}x 20D avg)")

        if orb_status == "Above 15m ORB High":
            score += 10
            reasons.append("Intraday breakout above 15-min Opening Range")
        elif orb_status == "Below 15m ORB Low":
            score -= 8
            reasons.append("Trading below 15-min Opening Range support")

        score = float(np.clip(round(score, 1), 5, 98))
        horizon = "Intraday / Short-Term" if orb_status == "Above 15m ORB High" else "Swing (1-4 Weeks)"
        trigger = (
            f"Enter near ₹{last_p:.2f} (above ORB ₹{orb_high:.2f})"
            if orb_status == "Above 15m ORB High"
            else f"Enter on strength above ₹{max(last_p, ema9):.2f}"
        )

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
            orb_high=round(orb_high, 2),
            orb_low=round(orb_low, 2),
            orb_status=orb_status,
            score=score,
            horizon=horizon,
            entry_trigger=trigger,
            reasons=reasons or ["Neutral technical setup"],
        )
