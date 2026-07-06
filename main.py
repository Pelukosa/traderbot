#!/usr/bin/env python3
"""traderbot — Asynchronous crypto trading bot.

Usage
-----
    python main.py                  # reads .env
    RISK_MODE=paper python main.py  # override via env

The bot loads market data, feeds it to the selected strategy, and reacts to
signals via the execution manager. A kill switch triggers on 3+ consecutive
exchange errors and shuts everything down safely.
"""

from __future__ import annotations

import asyncio
import sys

from loguru import logger

from src.config import settings
from src.exchange import ExchangeManager
from src.execution import ExecutionManager
from src.logger import setup_logger
from src.strategy import STRATEGY_REGISTRY

from src.strategy import Signal


async def main_loop() -> None:
    setup_logger()
    logger.info("traderbot starting — mode={}", settings.risk_mode)

    exchange = ExchangeManager()
    strategy_cls = STRATEGY_REGISTRY.get(settings.strategy)
    if strategy_cls is None:
        logger.error("Unknown strategy: {}", settings.strategy)
        sys.exit(1)

    strategy = strategy_cls()
    execution = ExecutionManager(exchange)

    try:
        await exchange.load_markets()
        logger.info("Markets loaded — connected to {}", settings.exchange_id)

        # Warm-up: fetch enough candles for the strategy windows
        ohlcv = await exchange.fetch_ohlcv(
            settings.trading_symbol, timeframe="1m", limit=100
        )
        logger.info("Initial OHLCV candles: {}", len(ohlcv))

        # ── Main loop ───────────────────────────────────────────────────────
        while True:
            # 1. Kill switch check
            if exchange.kill_switch_triggered:
                logger.critical("Kill switch active — stopping traderbot")
                await execution.emergency_close_all()
                break

            # 2. Reconnect if needed (consecutive errors but not yet killed)
            if exchange.consecutive_errors > 0:
                logger.warning("Reconnecting after {} errors …", exchange.consecutive_errors)
                await exchange.reconnect()
                ohlcv = await exchange.fetch_ohlcv(
                    settings.trading_symbol, timeframe="1m", limit=100
                )
                continue

            # 3. Get fresh data (WebSocket)
            try:
                candle = await exchange.watch_ohlcv(
                    settings.trading_symbol, timeframe="1m"
                )
                if candle:
                    ohlcv.append(candle[-1])
                    # Keep a rolling window
                    if len(ohlcv) > 200:
                        ohlcv = ohlcv[-200:]
            except Exception:
                logger.exception("Error fetching live data")
                continue

            # 4. Run strategy
            try:
                signal = await strategy.compute_signal(ohlcv)
            except Exception:
                logger.exception("Strategy error")
                continue

            # 5. Execute
            if signal.action != "hold":
                current_price = float(ohlcv[-1][4])  # close price
                logger.info("EXECUTING {} @ {:.2f} — conf={:.2f}", signal.action.upper(), current_price, signal.confidence)
                await execution.execute_signal(signal, current_price, strategy_name=strategy.name)

            # Sleep 60s — check every new 1m candle
            await asyncio.sleep(45)

    except asyncio.CancelledError:
        logger.info("Shutdown requested")
    except Exception:
        logger.exception("Fatal error in main loop")
    finally:
        logger.info("Cleaning up …")
        await exchange.close()
        logger.info("traderbot stopped")


def main() -> None:
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        logger.info("Received SIGINT — exiting")


if __name__ == "__main__":
    main()
