from dataclasses import dataclass
from typing import List
import urllib.parse
import xml.etree.ElementTree as ET
import numpy as np


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
class NewsSignal:
    symbol: str
    headline_count: int
    score: float
    sentiment_label: str
    event_risk_flag: bool
    top_headlines: List[str]


class NewsEngine:
    """Fetches & scores news sentiment and binary event risk."""

    def fetch_google_news_rss(self, query: str, max_items: int = 6) -> List[str]:
        import urllib.request
        encoded = urllib.parse.quote(f"{query} NSE stock")
        url = f"https://news.google.com/rss/search?q={encoded}&hl=en-IN&gl=IN&ceid=IN:en"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=6) as resp:
                xml_data = resp.read()
            root = ET.fromstring(xml_data)
            titles = []
            for item in root.findall("./channel/item/title")[:max_items]:
                if item.text:
                    titles.append(item.text.strip())
            return titles
        except Exception:
            return []

    def analyze(self, symbol: str, company_name: str = "", offline_headlines: List[str] = None) -> NewsSignal:
        headlines = offline_headlines if offline_headlines is not None else self.fetch_google_news_rss(company_name or symbol)

        score = 50.0
        event_flag = False
        for h in headlines:
            lower = h.lower()
            for kw, pts in POSITIVE_KEYWORDS.items():
                if kw in lower:
                    score += pts
            for kw, pts in NEGATIVE_KEYWORDS.items():
                if kw in lower:
                    score += pts
            if any(ek in lower for ek in EVENT_RISK_KEYWORDS):
                event_flag = True

        score = float(np.clip(round(score, 1), 10, 95))
        if score >= 62:
            label = "Positive Catalyst"
        elif score <= 40:
            label = "Negative / Caution"
        else:
            label = "Neutral"

        return NewsSignal(
            symbol=symbol.upper(),
            headline_count=len(headlines),
            score=score,
            sentiment_label=label,
            event_risk_flag=event_flag,
            top_headlines=headlines[:3],
        )
