from __future__ import annotations

import csv
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PERF_LOG = Path("performance.csv")

# In-memory aggregator
_strategy_stats: dict[str, dict] = defaultdict(
    lambda: {"trades": 0, "wins": 0, "losses": 0, "pnl_eur": 0.0, "fees_eur": 0.0}
)


def _init_csv() -> None:
    if not PERF_LOG.exists():
        with open(PERF_LOG, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                # Identificación
                "trade_id",
                "timestamp",
                "strategy",
                "side",  # buy | sell (buy=opening, sell=closing)
                "symbol",

                # Precios y PnL
                "entry_price",
                "exit_price",
                "entry_timestamp",
                "exit_timestamp",
                "duration_minutes",
                "amount",
                "pnl_eur",
                "pnl_pct",
                "fee_eur",

                # Contexto del histograma en COMPRA
                "buy_histogram",
                "buy_valley_value",
                "buy_valley_to_entry_diff",
                "buy_reversal_slope",
                "buy_macd_line",
                "buy_signal_line",
                "buy_price",

                # Contexto del histograma en VENTA
                "sell_histogram",
                "sell_peak_value",
                "sell_peak_to_exit_diff",
                "sell_reversal_slope",
                "sell_macd_line",
                "sell_signal_line",
                "sell_price",

                # Contexto de mercado
                "market_trend_20",
                "market_volatility_20",
                "market_volume_ratio",
                "hour_of_day",
                "day_of_week",
            ])


def log_strategy_trade(
    trade_id: str,
    strategy: str,
    side: str,
    symbol: str,
    entry_price: float | None = None,
    exit_price: float | None = None,
    entry_timestamp: str = "",
    exit_timestamp: str = "",
    duration_minutes: float = 0.0,
    amount: float = 0.0,
    pnl_eur: float = 0.0,
    pnl_pct: float = 0.0,
    fee_eur: float = 0.0,
    # Contexto compra
    buy_histogram: float = 0.0,
    buy_valley_value: float = 0.0,
    buy_valley_to_entry_diff: float = 0.0,
    buy_reversal_slope: float = 0.0,
    buy_macd_line: float = 0.0,
    buy_signal_line: float = 0.0,
    buy_price: float = 0.0,
    # Contexto venta
    sell_histogram: float = 0.0,
    sell_peak_value: float = 0.0,
    sell_peak_to_exit_diff: float = 0.0,
    sell_reversal_slope: float = 0.0,
    sell_macd_line: float = 0.0,
    sell_signal_line: float = 0.0,
    sell_price: float = 0.0,
    # Mercado
    market_trend_20: float = 0.0,
    market_volatility_20: float = 0.0,
    market_volume_ratio: float = 0.0,
    hour_of_day: int = 0,
    day_of_week: int = 0,
) -> None:
    """Log a completed trade with full ML context."""
    _init_csv()

    stat = _strategy_stats[strategy]
    stat["trades"] += 1
    stat["pnl_eur"] += pnl_eur
    stat["fees_eur"] += fee_eur
    if side == "sell":
        if pnl_eur > 0:
            stat["wins"] += 1
        elif pnl_eur < 0:
            stat["losses"] += 1

    now = datetime.now(timezone.utc).isoformat()
    with open(PERF_LOG, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            trade_id,
            now,
            strategy,
            side,
            symbol,
            round(entry_price, 2) if entry_price else "",
            round(exit_price, 2) if exit_price else "",
            entry_timestamp,
            exit_timestamp,
            round(duration_minutes, 1),
            round(amount, 8),
            round(pnl_eur, 2),
            round(pnl_pct, 2),
            round(fee_eur, 4),
            round(buy_histogram, 2),
            round(buy_valley_value, 2),
            round(buy_valley_to_entry_diff, 2),
            round(buy_reversal_slope, 2),
            round(buy_macd_line, 2),
            round(buy_signal_line, 2),
            round(buy_price, 2),
            round(sell_histogram, 2),
            round(sell_peak_value, 2),
            round(sell_peak_to_exit_diff, 2),
            round(sell_reversal_slope, 2),
            round(sell_macd_line, 2),
            round(sell_signal_line, 2),
            round(sell_price, 2),
            round(market_trend_20, 2),
            round(market_volatility_20, 2),
            round(market_volume_ratio, 2),
            hour_of_day,
            day_of_week,
        ])


def performance_summary() -> str:
    """Return a plain-text summary per strategy."""
    _init_csv()

    trades_by_strategy: dict[str, list[dict]] = defaultdict(list)
    with open(PERF_LOG) as f:
        for row in csv.DictReader(f):
            trades_by_strategy[row["strategy"]].append(row)

    if not trades_by_strategy:
        return "No trades yet."

    lines: list[str] = []
    total_pnl = 0.0
    total_fees = 0.0

    for strategy in sorted(trades_by_strategy):
        trades = trades_by_strategy[strategy]
        sells = [t for t in trades if t["side"] == "sell"]
        pnl = sum(float(t["pnl_eur"]) for t in sells)
        fees = sum(float(t["fee_eur"]) for t in trades)
        wins = sum(1 for t in sells if float(t["pnl_eur"]) > 0)
        losses = sum(1 for t in sells if float(t["pnl_eur"]) < 0)
        total_pnl += pnl
        total_fees += fees

        wr = f"{wins/(wins+losses)*100:.0f}%" if (wins + losses) > 0 else "-"
        lines.append(
            f"  📈 *{strategy}*  {len(trades)} ops  "
            f"PnL {pnl:+.2f}€  fees {fees:.2f}€  "
            f"win {wr}  ({wins}W/{losses}L)"
        )

    lines.insert(0, f"📊 *RESUMEN POR ESTRATEGIA*  PnL total {total_pnl:+.2f}€  fees {total_fees:.2f}€")
    return "\n".join(lines)
