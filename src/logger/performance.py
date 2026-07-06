from __future__ import annotations

import csv
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

PERF_LOG = Path("performance.csv")

# In-memory aggregator for quick reporting
_strategy_stats: dict[str, dict] = defaultdict(
    lambda: {
        "trades": 0,
        "wins": 0,
        "losses": 0,
        "pnl_eur": 0.0,
        "fees_eur": 0.0,
    }
)


def _init_csv() -> None:
    if not PERF_LOG.exists():
        with open(PERF_LOG, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "timestamp",
                "strategy",
                "side",
                "symbol",
                "entry_price",
                "exit_price",
                "amount",
                "pnl_eur",
                "pnl_pct",
                "fee_eur",
            ])


def log_strategy_trade(
    strategy: str,
    side: str,
    symbol: str,
    entry_price: float | None,
    exit_price: float | None,
    amount: float,
    pnl_eur: float = 0.0,
    pnl_pct: float = 0.0,
    fee_eur: float = 0.0,
) -> None:
    """Log a completed trade to per-strategy CSV and update in-memory stats."""
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

    with open(PERF_LOG, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            datetime.now(timezone.utc).isoformat(),
            strategy,
            side,
            symbol,
            round(entry_price, 2) if entry_price else "",
            round(exit_price, 2) if exit_price else "",
            round(amount, 8),
            round(pnl_eur, 2),
            round(pnl_pct, 2),
            round(fee_eur, 4),
        ])


def performance_summary() -> str:
    """Return a plain-text summary per strategy."""
    _init_csv()

    # Re-read from CSV to get complete picture (in case of restart)
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
