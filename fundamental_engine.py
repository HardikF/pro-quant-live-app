from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import pandas as pd


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
    valuation_label: str
    reasons: List[str]


class FundamentalEngine:
    """Scores valuation, profitability (ROE), balance-sheet safety, and growth."""

    def analyze_from_yfinance(self, symbol: str) -> Optional[FundamentalSignal]:
        try:
            import yfinance as yf
            info = yf.Ticker(symbol).info or {}
        except Exception:
            return None

        if not info:
            return None

        name = info.get("shortName") or info.get("longName") or symbol
        sector = info.get("sector") or "Unknown"
        pe = float(info.get("trailingPE") or info.get("forwardPE") or 0.0)
        roe = float(info.get("returnOnEquity") or 0.0) * 100.0
        de = float(info.get("debtToEquity") or 0.0)
        div_y = float(info.get("dividendYield") or 0.0) * 100.0
        rev_g = float(info.get("revenueGrowth") or 0.0) * 100.0

        return self._score_metrics(symbol, name, sector, pe, roe, de, div_y, rev_g)

    def analyze_from_row(self, row: pd.Series) -> FundamentalSignal:
        return self._score_metrics(
            symbol=str(row["symbol"]).upper(),
            name=str(row.get("name", row["symbol"])),
            sector=str(row.get("sector", "Unknown")),
            pe=float(row.get("pe", 0.0)),
            roe=float(row.get("roe_pct", 15.0)),
            de=float(row.get("debt_to_equity", 0.5)),
            div_y=float(row.get("div_yield_pct", 1.0)),
            rev_g=float(row.get("revenue_growth_pct", 10.0)),
        )

    @staticmethod
    def _score_metrics(
        symbol: str,
        name: str,
        sector: str,
        pe: float,
        roe: float,
        de: float,
        div_y: float,
        rev_g: float,
    ) -> FundamentalSignal:
        score = 50.0
        reasons = []

        if 0 < pe <= 16:
            score += 18
            reasons.append(f"Attractive valuation (P/E {pe:.1f}x)")
        elif 16 < pe <= 24:
            score += 10
            reasons.append(f"Fair valuation (P/E {pe:.1f}x)")
        elif pe > 40:
            score -= 14
            reasons.append(f"Expensive valuation (P/E {pe:.1f}x)")

        if roe >= 18:
            score += 15
            reasons.append(f"High capital efficiency (ROE {roe:.1f}%)")
        elif roe >= 12:
            score += 8
            reasons.append(f"Solid ROE ({roe:.1f}%)")
        elif 0 < roe < 8:
            score -= 8
            reasons.append(f"Weak ROE ({roe:.1f}%)")

        if de <= 0.5 or de <= 50:
            # yfinance sometimes reports D/E in %, e.g. 45 vs 0.45
            norm_de = de / 100.0 if de > 10 else de
            if norm_de <= 0.5:
                score += 8
                reasons.append("Low debt / strong balance sheet")
            elif norm_de > 1.5:
                score -= 12
                reasons.append(f"Elevated leverage (D/E {norm_de:.2f})")

        if rev_g >= 12:
            score += 10
            reasons.append(f"Strong revenue growth ({rev_g:.1f}% YoY)")
        elif rev_g < 0:
            score -= 10
            reasons.append(f"Contracting revenue ({rev_g:.1f}% YoY)")

        if div_y >= 2.0:
            score += 6
            reasons.append(f"Dividend yield cushion ({div_y:.2f}%)")

        score = float(np.clip(round(score, 1), 5, 98))
        if score >= 70:
            val_label = "High-Quality Value"
        elif score >= 52:
            val_label = "Fair / Balanced"
        else:
            val_label = "Expensive / Weak Fundamentals"

        return FundamentalSignal(
            symbol=symbol.upper(),
            company_name=name,
            sector=sector,
            pe_ratio=round(pe, 2),
            roe_pct=round(roe, 1),
            debt_to_equity=round(de, 2),
            div_yield_pct=round(div_y, 2),
            revenue_growth_pct=round(rev_g, 1),
            score=score,
            valuation_label=val_label,
            reasons=reasons or ["Standard fundamental profile"],
        )
