import os
from dataclasses import dataclass, field
from typing import List

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


@dataclass
class EngineConfig:
    # Portfolio & Risk Settings
    total_capital: float = float(os.getenv("PROQUANT_CAPITAL", "500000"))
    risk_per_trade_pct: float = float(os.getenv("PROQUANT_RISK_PCT", "1.0"))
    max_portfolio_heat_pct: float = float(os.getenv("PROQUANT_MAX_HEAT_PCT", "5.0"))
    max_sector_alloc_pct: float = float(os.getenv("PROQUANT_MAX_SECTOR_PCT", "25.0"))
    target_rr_ratio: float = float(os.getenv("PROQUANT_RR_RATIO", "2.0"))

    # Engine Weights (must sum to 1.0)
    weight_quant: float = 0.40
    weight_fundamental: float = 0.35
    weight_news: float = 0.25

    # Optional API Keys (free yfinance + Google News RSS work without any keys)
    newsapi_key: str = os.getenv("NEWSAPI_KEY", "")
    telegram_bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "")

    # Default NIFTY 50 Liquid Watchlist
    watchlist: List[str] = field(default_factory=lambda: [
        "RELIANCE.NS",
        "TCS.NS",
        "HDFCBANK.NS",
        "INFY.NS",
        "ICICIBANK.NS",
        "SBIN.NS",
        "ITC.NS",
        "LT.NS",
        "BHARTIARTL.NS",
        "AXISBANK.NS",
    ])
