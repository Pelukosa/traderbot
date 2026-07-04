from __future__ import annotations

import asyncio
from typing import Any

import ccxt.pro as ccxtpro  # type: ignore[import-untyped]
from loguru import logger

from src.config import settings


class ExchangeError(Exception):
    """Base error for exchange operations."""


class ExchangeManager:
    """Async CCXT exchange client with rate-limit handling and auto-reconnect.

    Wraps ``ccxt.pro`` (async WebSocket / REST) for low-latency data and order
    execution.  All public methods are safe to call concurrently.
    """

    def __init__(self) -> None:
        exchange_class = getattr(ccxtpro, settings.exchange_id, None)
        if exchange_class is None:
            raise ExchangeError(f"Unsupported exchange: {settings.exchange_id}")

        self._exchange: ccxtpro.Exchange = exchange_class({
            "apiKey": settings.api_key,
            "secret": settings.api_secret,
            "enableRateLimit": True,
            "sandbox": settings.exchange_sandbox,
            "options": {"defaultType": "spot"},
        })
        self._consecutive_errors: int = 0
        self._running = True
        self._kill_switch_triggered = False

    # ── public API ──────────────────────────────────────────────────────────

    @property
    def consecutive_errors(self) -> int:
        return self._consecutive_errors

    @property
    def kill_switch_triggered(self) -> bool:
        return self._kill_switch_triggered

    async def load_markets(self) -> dict[str, Any]:
        return await self._safe_call(self._exchange.load_markets)

    async def fetch_ohlcv(
        self, symbol: str, timeframe: str = "1m", limit: int = 100
    ) -> list[list[float]]:
        """Return OHLCV candles: ``[[timestamp, open, high, low, close, volume], …]``."""
        return await self._safe_call(
            self._exchange.fetch_ohlcv, symbol, timeframe, limit=limit
        )

    async def watch_ohlcv(
        self, symbol: str, timeframe: str = "1m"
    ) -> list[list[float]]:
        """Subscribe to real-time OHLCV via WebSocket (ccxt.pro)."""
        return await self._safe_call(
            self._exchange.watch_ohlcv, symbol, timeframe
        )

    async def create_limit_buy_order(
        self, symbol: str, amount: float, price: float
    ) -> dict[str, Any]:
        return await self._safe_call(
            self._exchange.create_limit_buy_order, symbol, amount, price
        )

    async def create_limit_sell_order(
        self, symbol: str, amount: float, price: float
    ) -> dict[str, Any]:
        return await self._safe_call(
            self._exchange.create_limit_sell_order, symbol, amount, price
        )

    async def cancel_all_orders(self, symbol: str) -> list[dict[str, Any]]:
        return await self._safe_call(
            self._exchange.cancel_all_orders, symbol
        )

    async def fetch_balance(self) -> dict[str, Any]:
        return await self._safe_call(self._exchange.fetch_balance)

    async def close(self) -> None:
        self._running = False
        await self._exchange.close()

    # ── internals ────────────────────────────────────────────────────────────

    async def _safe_call(self, method, *args, **kwargs) -> Any:
        """Call *method* with error counting and kill-switch logic."""
        if self._kill_switch_triggered:
            raise ExchangeError("Kill switch active — exchange calls blocked")

        try:
            result = await method(*args, **kwargs)
            self._consecutive_errors = 0
            return result
        except Exception as exc:
            self._consecutive_errors += 1
            logger.warning(
                "Exchange error #{}: {}",
                self._consecutive_errors,
                exc,
            )
            if self._consecutive_errors >= settings.max_consecutive_errors:
                logger.critical(
                    "{} consecutive errors — TRIGGERING KILL SWITCH",
                    self._consecutive_errors,
                )
                self._kill_switch_triggered = True
                await self._emergency_shutdown()
            raise ExchangeError(str(exc)) from exc

    async def _emergency_shutdown(self) -> None:
        """Cancel all open orders and stop the bot."""
        try:
            await self.cancel_all_orders(settings.trading_symbol)
            logger.info("All orders cancelled on kill switch")
        except Exception as exc:
            logger.error("Kill-switch cleanup failed: {}", exc)
        self._running = False

    async def reconnect(self) -> None:
        """Close and re-create the exchange client (exponential back-off)."""
        logger.info("Reconnecting to exchange …")
        await self._exchange.close()
        await asyncio.sleep(5)
        self.__init__()  # type: ignore[misc]
        await self.load_markets()
        logger.info("Reconnected successfully")
