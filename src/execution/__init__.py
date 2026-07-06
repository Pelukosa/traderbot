from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from src.config import settings
from src.exchange import ExchangeManager
from src.logger.trades import log_trade
from src.logger.performance import log_strategy_trade
from src.strategy import Signal, STRATEGY_REGISTRY


@dataclass
class Position:
    symbol: str
    side: str  # "long" | "short"
    entry_price: float
    amount: float
    stop_loss: float | None = None
    take_profit: float | None = None
    active: bool = True


class OrderError(Exception):
    """Raised when order execution fails."""


class ExecutionManager:
    """Manages order placement, stop-loss/take-profit, and position lifecycle.

    In **simulation** mode no real orders are placed — only logged.
    In **paper** mode the exchange sandbox is used.
    In **live** mode real orders are executed.
    """

    def __init__(self, exchange: ExchangeManager) -> None:
        self._exchange = exchange
        self._position: Position | None = None

    @property
    def in_position(self) -> bool:
        return self._position is not None and self._position.active

    def _simulate_only(self) -> bool:
        return settings.risk_mode == "simulation"

    async def execute_signal(self, signal: Signal, price: float, strategy_name: str = "") -> None:
        """React to a strategy signal: open/close positions."""
        symbol = settings.trading_symbol

        if signal.action == "buy" and not self.in_position:
            # Calculate 69% of EUR balance
            try:
                bal = await self._exchange.fetch_balance()
                eur_free = float(bal.get("EUR", {}).get("free", 0))
                logger.info("EUR balance: {:.2f}", eur_free)
            except Exception:
                eur_free = 0.0
                logger.warning("Could not fetch EUR balance, defaulting to 0")

            invest_eur = eur_free * 0.69
            if invest_eur < 5:
                logger.warning("Insufficient EUR ({} < 5€) — skipping buy", round(invest_eur, 2))
                return

            size = invest_eur / price
            logger.info("Investing {:.2f}€ → {:.6f} BTC @ {:.2f}", invest_eur, size, price)
            await self._open_long(symbol, size, price, eur_free, strategy_name)

        elif signal.action == "sell" and self.in_position:
            await self._close_position(symbol, strategy_name)

        # Also execute macd_divergence signals directly (they run solo)
        # Check exits whether in position or not
        if self.in_position:
            await self._check_exits(price)

    async def _open_long(self, symbol: str, amount: float, price: float, eur_free: float = 0.0, strategy_name: str = "") -> None:
        sl = price * (1 - settings.stop_loss_percent / 100.0)
        tp = price * (1 + settings.take_profit_percent / 100.0)

        if self._simulate_only():
            logger.info(
                "[SIM] BUY {} {} @ {:.2f}  SL={:.2f}  TP={:.2f}",
                amount, symbol, price, sl, tp,
            )
            log_trade("buy", symbol, amount, price, 0.0, balance_eur=eur_free)
        else:
            try:
                order = await self._exchange.create_market_buy_order(
                    symbol, amount
                )
                logger.info("ORDER PLACED: {}", order)
                # Extract real fill price and fee from order response
                fills = order.get("fills", [])
                if fills:
                    avg_price = sum(f["price"] * f["amount"] for f in fills) / sum(f["amount"] for f in fills)
                    total_fee = sum(f.get("fee", {}).get("cost", 0) for f in fills)
                    filled_amount = sum(f["amount"] for f in fills)
                else:
                    avg_price = float(order.get("price", price))
                    total_fee = float(order.get("fee", {}).get("cost", 0))
                    filled_amount = float(order.get("filled", amount))
                log_trade("buy", symbol, filled_amount, avg_price, total_fee,
                          order_id=order.get("id", ""), balance_eur=eur_free)
                amount = filled_amount
                price = avg_price
            except Exception as exc:
                raise OrderError(f"Buy order failed: {exc}") from exc

        self._position = Position(
            symbol=symbol,
            side="long",
            entry_price=price,
            amount=amount,
            stop_loss=sl,
            take_profit=tp,
        )

    async def _close_position(self, symbol: str, strategy_name: str = "") -> None:
        if self._position is None:
            return

        entry_price = self._position.entry_price
        entry_amount = self._position.amount

        if self._simulate_only():
            logger.info(
                "[SIM] SELL {} {} @ market",
                entry_amount, symbol,
            )
            log_trade("sell", symbol, entry_amount, entry_price, 0.0)
        else:
            try:
                bal = await self._exchange.fetch_balance()
                base = symbol.split("/")[0]
                free = float(bal.get(base, {}).get("free", 0))
                if free < entry_amount * 0.99:
                    logger.warning("Insufficient {} balance: {:.6f}", base, free)
                    sell_amount = free
                else:
                    sell_amount = entry_amount
                if sell_amount > 0:
                    order = await self._exchange.create_market_sell_order(
                        symbol, sell_amount
                    )
                    logger.info("SELL ORDER PLACED: {}", order)
                    # Extract real fill price and fee
                    fills = order.get("fills", [])
                    if fills:
                        sell_price = sum(f["price"] * f["amount"] for f in fills) / sum(f["amount"] for f in fills)
                        sell_fee = sum(f.get("fee", {}).get("cost", 0) for f in fills)
                    else:
                        sell_price = float(order.get("price", entry_price))
                        sell_fee = float(order.get("fee", {}).get("cost", 0))
                    pnl_eur = (sell_price - entry_price) * sell_amount
                    pnl_pct = ((sell_price - entry_price) / entry_price) * 100
                    # Fetch EUR balance after sell
                    try:
                        bal2 = await self._exchange.fetch_balance()
                        eur_after = float(bal2.get("EUR", {}).get("free", 0))
                    except Exception:
                        eur_after = 0.0
                    log_trade("sell", symbol, sell_amount, sell_price, sell_fee,
                              order_id=order.get("id", ""), pnl_eur=pnl_eur, pnl_pct=pnl_pct,
                              balance_eur=eur_after)
                    log_strategy_trade(
                        strategy_name or "unknown", "sell", symbol,
                        entry_price, sell_price, sell_amount,
                        pnl_eur=pnl_eur, pnl_pct=pnl_pct, fee_eur=sell_fee,
                    )
                    logger.info("Position closed  entry={:.2f}  exit={:.2f}  pnl={:+.2f}€ ({:+.2f}%)",
                                entry_price, sell_price, pnl_eur, pnl_pct)
            except Exception as exc:
                raise OrderError(f"Sell order failed: {exc}") from exc

        self._position = None

    async def _check_exits(self, current_price: float) -> None:
        if self._position is None or not self._position.active:
            return

        sl = self._position.stop_loss
        tp = self._position.take_profit
        if sl is not None and current_price <= sl:
            logger.warning("STOP-LOSS HIT @ {:.2f}", current_price)
            await self._close_position(self._position.symbol)

        elif tp is not None and current_price >= tp:
            logger.info("TAKE-PROFIT HIT @ {:.2f}", current_price)
            await self._close_position(self._position.symbol)

    async def emergency_close_all(self) -> None:
        if self._position:
            await self._close_position(self._position.symbol)
