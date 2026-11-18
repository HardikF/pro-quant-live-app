from dataclasses import dataclass
from typing import Dict, List
from config import EngineConfig
from .quant_engine import QuantSignal
from .fundamental_engine import FundamentalSignal
from .news_engine import NewsSignal


@dataclass
class TradeCall:
    symbol: str
    company_name: str
    sector: str
    action: str           # STRONG BUY / BUY / WATCH / AVOID
    composite_score: float
    horizon: str
    why: str
    when_to_enter: str
    entry_price: float
    stop_loss: float
    target_1: float
    target_2: float
    qty: int
    capital_required: float
    max_loss_inr: float
    portfolio_pct: float
    risk_warnings: List[str]


class RiskEngine:
    """Combines all 3 engines into sized, risk-capped plain-English Trade Calls."""

    def __init__(self, config: EngineConfig):
        self.cfg = config

    def generate_calls(
        self,
        items: List[tuple],  # List of (QuantSignal, FundamentalSignal, NewsSignal)
    ) -> List[TradeCall]:
        raw_calls: List[TradeCall] = []
        max_risk_per_trade = self.cfg.total_capital * (self.cfg.risk_per_trade_pct / 100.0)

        for q, f, n in items:
            q_sig: QuantSignal = q
            f_sig: FundamentalSignal = f
            n_sig: NewsSignal = n

            composite = round(
                q_sig.score * self.cfg.weight_quant
                + f_sig.score * self.cfg.weight_fundamental
                + n_sig.score * self.cfg.weight_news,
                1,
            )

            warnings: List[str] = []
            if n_sig.event_risk_flag:
                warnings.append("Binary event/news in headlines — consider half position size")
            if n_sig.score <= 38:
                warnings.append("Negative news sentiment detected")

            if composite >= 70 and n_sig.score > 38:
                action = "STRONG BUY"
            elif composite >= 58 and n_sig.score > 35:
                action = "BUY"
            elif composite >= 45:
                action = "WATCH"
            else:
                action = "AVOID"

            entry = q_sig.last_price
            stop_dist = max(q_sig.atr14 * 1.25, entry * 0.012)
            stop_loss = round(entry - stop_dist, 2)
            target_1 = round(entry + stop_dist * self.cfg.target_rr_ratio, 2)
            target_2 = round(entry + stop_dist * (self.cfg.target_rr_ratio + 1.0), 2)

            effective_risk = max_risk_per_trade * (0.5 if n_sig.event_risk_flag else 1.0)
            qty = max(1, int(effective_risk // stop_dist)) if action in ("STRONG BUY", "BUY") else 0
            cap_req = round(qty * entry, 2)
            # Cap single position at 25% of total capital
            max_pos_cap = self.cfg.total_capital * 0.25
            if cap_req > max_pos_cap and entry > 0:
                qty = max(1, int(max_pos_cap // entry))
                cap_req = round(qty * entry, 2)
                warnings.append("Position capped at 25% max single-stock allocation")

            max_loss = round(qty * stop_dist, 2)
            port_pct = round((cap_req / self.cfg.total_capital) * 100.0, 1) if self.cfg.total_capital > 0 else 0.0

            why = (
                f"Quant {q_sig.score}/100 ({q_sig.reasons[0]}); "
                f"Fundamentals {f_sig.score}/100 ({f_sig.reasons[0]}); "
                f"News {n_sig.score}/100 ({n_sig.sentiment_label})."
            )

            raw_calls.append(
                TradeCall(
                    symbol=q_sig.symbol,
                    company_name=f_sig.company_name,
                    sector=f_sig.sector,
                    action=action,
                    composite_score=composite,
                    horizon=q_sig.horizon,
                    why=why,
                    when_to_enter=q_sig.entry_trigger,
                    entry_price=entry,
                    stop_loss=stop_loss,
                    target_1=target_1,
                    target_2=target_2,
                    qty=qty,
                    capital_required=cap_req,
                    max_loss_inr=max_loss,
                    portfolio_pct=port_pct,
                    risk_warnings=warnings,
                )
            )

        # Sort highest score first & enforce Portfolio Heat + Sector Caps
        raw_calls.sort(key=lambda c: c.composite_score, reverse=True)
        running_heat_pct = 0.0
        sector_alloc: Dict[str, float] = {}

        for call in raw_calls:
            if call.action not in ("STRONG BUY", "BUY"):
                continue
            trade_heat_pct = (call.max_loss_inr / self.cfg.total_capital) * 100.0
            cur_sec_pct = sector_alloc.get(call.sector, 0.0)
            if running_heat_pct + trade_heat_pct > self.cfg.max_portfolio_heat_pct:
                call.action = "WATCH (HEAT CAP)"
                call.qty = 0
                call.capital_required = 0.0
                call.max_loss_inr = 0.0
                call.portfolio_pct = 0.0
                call.risk_warnings.append(f"Skipped: Portfolio heat limit ({self.cfg.max_portfolio_heat_pct}%) reached")
            elif cur_sec_pct + call.portfolio_pct > self.cfg.max_sector_alloc_pct:
                call.action = "WATCH (SECTOR CAP)"
                call.qty = 0
                call.capital_required = 0.0
                call.max_loss_inr = 0.0
                call.portfolio_pct = 0.0
                call.risk_warnings.append(f"Skipped: Sector limit ({self.cfg.max_sector_alloc_pct}%) reached for {call.sector}")
            else:
                running_heat_pct += trade_heat_pct
                sector_alloc[call.sector] = cur_sec_pct + call.portfolio_pct

        return raw_calls
