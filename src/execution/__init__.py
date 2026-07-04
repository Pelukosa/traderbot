from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from src.config import settings
from src.exchange import ExchangeManager
from src.strategy import Signal


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

    async def execute_signal(self, signal: Signal, price: float) -> None:
        """React to a strategy signal: open/close positions."""
        symbol = settings.trading_symbol
        size = settings.max_position_size_btc

        if signal.action == "buy" and not self.in_position:
            await self._open_long(symbol, size, price)

        elif signal.action == "sell" and self.in_position:
            await self._close_position(symbol)

        # Check stop / take-profit on every tick if in position
        if self.in_position:
            await self._check_exits(price)

    async def _open_long(self, symbol: str, amount: float, price: float) -> None:
        sl = price * (1 - settings.stop_loss_percent / 100.0)
        tp = price * (1 + settings.take_profit_percent / 100.0)

        if self._simulate_only():
            logger.info(
                "[SIM] BUY {} {} @ {:.2f}  SL={:.2f}  TP={:.2f}",
                amount, symbol, price, sl, tp,
            )
        else:
            try:
                order = await self._exchange.create_limit_buy_order(
                    symbol, amount, price
                )
                logger.info("ORDER PLACED: {}", order)
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

    async def _close_position(self, symbol: str) -> None:
        if self._position is None:
            return

        if self._simulate_only():
            logger.info(
                "[SIM] SELL {} {} @ market",
                self._position.amount,
                symbol,
            )
        else:
            try:
                bal = await self._exchange.fetch_balance()
                base = symbol.split("/")[0]
                free = float(bal.get(base, {}).get("free", 0))
                if free < self._position.amount * 0.99:
                    logger.warning("Insufficient {} balance: {:.6f}", base, free)
                order = await self._exchange.create_limit_sell_order(
                    symbol, self._position.amount, self._position.entry_price
                )
                logger.info("ORDER PLACED: {}", order)
            except Exception as exc:
                raise OrderError(f"Sell order failed: {exc}") from exc

        pnl = 0.0  # Real PnL calculated once order fills
        logger.info(
            "Position closed  entry={:.2f}  pnl={:+.2f}%",
            self._position.entry_price,
            pnl,
        )
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
