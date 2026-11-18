from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class BacktestStats:
    symbol: str
    total_trades: int
    win_rate_pct: float
    profit_factor: float
    net_return_pct: float
    max_drawdown_pct: float
    edge_label: str


class StrategyBacktester:
    """
    Walk-forward backtester for EMA(9/21) + RSI(14) swing/intraday setups
    with Indian equity cost deduction (approx 0.12% round-trip slippage + STT/brokerage).
    """

    ROUND_TRIP_COST_PCT = 0.0012  # 0.12% conservative cost + slippage

    @staticmethod
    def _ema(s: pd.Series, span: int) -> pd.Series:
        return s.ewm(span=span, adjust=False).mean()

    @staticmethod
    def _rsi(s: pd.Series, period: int = 14) -> pd.Series:
        delta = s.diff()
        gain = delta.clip(lower=0).rolling(window=period).mean()
        loss = (-delta.clip(upper=0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, np.nan)
        return (100.0 - (100.0 / (1.0 + rs))).fillna(50.0)

    def run_on_dataframe(self, symbol: str, df: pd.DataFrame, rr_ratio: float = 2.0) -> BacktestStats:
        if df is None or len(df) < 40:
            return BacktestStats(symbol, 0, 0.0, 0.0, 0.0, 0.0, "Insufficient Data")

        close = df["Close"].astype(float)
        high = df["High"].astype(float)
        low = df["Low"].astype(float)

        ema9 = self._ema(close, 9)
        ema21 = self._ema(close, 21)
        rsi14 = self._rsi(close, 14)
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ], axis=1).max(axis=1)
        atr14 = tr.rolling(14).mean().bfill()

        trades = []
        in_trade = False
        entry_p = sl_p = tgt_p = 0.0

        for i in range(25, len(df)):
            c = float(close.iloc[i])
            h = float(high.iloc[i])
            l = float(low.iloc[i])

            if in_trade:
                if l <= sl_p:
                    ret = ((sl_p - entry_p) / entry_p) - self.ROUND_TRIP_COST_PCT
                    trades.append(ret)
                    in_trade = False
                elif h >= tgt_p:
                    ret = ((tgt_p - entry_p) / entry_p) - self.ROUND_TRIP_COST_PCT
                    trades.append(ret)
                    in_trade = False
                continue

            # Entry rule: EMA9 > EMA21 and RSI between 42 and 64
            if float(ema9.iloc[i]) > float(ema21.iloc[i]) and 42.0 <= float(rsi14.iloc[i]) <= 64.0:
                entry_p = c
                stop_dist = max(float(atr14.iloc[i]) * 1.25, entry_p * 0.012)
                sl_p = entry_p - stop_dist
                tgt_p = entry_p + stop_dist * rr_ratio
                in_trade = True

        if not trades:
            return BacktestStats(symbol, 0, 0.0, 0.0, 0.0, 0.0, "No Signals in Period")

        arr = np.array(trades)
        wins = arr[arr > 0]
        losses = arr[arr <= 0]
        win_rate = float(len(wins) / len(arr) * 100.0)
        gross_profit = float(wins.sum()) if len(wins) else 0.0
        gross_loss = float(abs(losses.sum())) if len(losses) else 0.0
        profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else 9.99

        equity = np.cumprod(1.0 + arr)
        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / peak
        max_dd = float(abs(dd.min()) * 100.0) if len(dd) else 0.0
        net_ret = float((equity[-1] - 1.0) * 100.0)

        if len(arr) >= 8 and win_rate >= 45.0 and profit_factor >= 1.35:
            edge = "Verified Positive Edge"
        elif profit_factor >= 1.0:
            edge = "Marginal Edge"
        else:
            edge = "Weak Historical Edge"

        return BacktestStats(
            symbol=symbol.upper(),
            total_trades=len(arr),
            win_rate_pct=round(win_rate, 1),
            profit_factor=profit_factor,
            net_return_pct=round(net_ret, 1),
            max_drawdown_pct=round(max_dd, 1),
            edge_label=edge,
        )

    def run_yfinance(self, symbol: str, period: str = "2y", rr_ratio: float = 2.0) -> BacktestStats:
        try:
            import yfinance as yf
            df = yf.Ticker(symbol).history(period=period, interval="1d")
            return self.run_on_dataframe(symbol, df, rr_ratio=rr_ratio)
        except Exception:
            return BacktestStats(symbol, 0, 0.0, 0.0, 0.0, 0.0, "Fetch Error")
