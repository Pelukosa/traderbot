from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

TRADES_LOG = Path("trades.csv")


def _init_csv() -> None:
    if not TRADES_LOG.exists():
        with open(TRADES_LOG, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "timestamp", "side", "symbol",
                "amount", "price", "cost_eur",
                "fee_eur", "order_id",
                "pnl_eur", "pnl_pct", "balance_eur",
            ])


def log_trade(
    side: str,
    symbol: str,
    amount: float,
    price: float,
    fee_eur: float,
    order_id: str = "",
    pnl_eur: float = 0.0,
    pnl_pct: float = 0.0,
    balance_eur: float = 0.0,
) -> None:
    _init_csv()
    cost = round(amount * price, 2)
    with open(TRADES_LOG, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            datetime.now(timezone.utc).isoformat(),
            side,
            symbol,
            round(amount, 8),
            round(price, 2),
            cost,
            round(fee_eur, 4),
            order_id,
            round(pnl_eur, 2),
            round(pnl_pct, 2),
            round(balance_eur, 2),
        ])


def summary() -> str:
    """Return a plain-text summary of all trades."""
    _init_csv()
    trades = []
    with open(TRADES_LOG) as f:
        for row in csv.DictReader(f):
            trades.append(row)

    if not trades:
        return "No trades yet."

    buys = [t for t in trades if t["side"] == "buy"]
    sells = [t for t in trades if t["side"] == "sell"]
    total_pnl = sum(float(t["pnl_eur"]) for t in sells)
    total_fees = sum(float(t["fee_eur"]) for t in trades)
    wins = sum(1 for t in sells if float(t["pnl_eur"]) > 0)
    losses = sum(1 for t in sells if float(t["pnl_eur"]) < 0)

    lines = [
        f"📊 *Trades: {len(trades)}*  ({len(buys)} buys / {len(sells)} sells)",
        f"   PnL total: {total_pnl:+.2f}€",
        f"   Comisiones totales: {total_fees:.2f}€",
        f"   Operaciones ganadoras: {wins}  perdedoras: {losses}",
        f"   Ratio acierto: {wins/(wins+losses)*100:.0f}%" if (wins+losses) > 0 else "",
        "",
    ]

    for t in trades[-5:]:  # last 5
        lines.append(
            f"   {t['timestamp'][:16]}  {t['side'].upper():5s}  "
            f"{t['amount']} @ {t['price']}€  "
            f"{' PnL:'+t['pnl_eur']+'€' if t['pnl_eur'] != '0.0' else ''}"
        )

    return "\n".join(lines)
