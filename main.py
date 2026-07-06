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

        # ── Restore position on restart ──
        # Check if we already have BTC (position from a previous run)
        try:
            bal = await exchange.fetch_balance()
            base = settings.trading_symbol.split("/")[0]
            btc_free = float(bal.get(base, {}).get("free", 0))
            eur_free = float(bal.get("EUR", {}).get("free", 0))
            logger.info("Restore check — {}: {:.8f}  EUR: {:.2f}", base, btc_free, eur_free)
            if btc_free >= 0.00001:
                # We're already in position — don't buy until we sell
                # Use current market price as estimated entry
                try:
                    ticker_ohlcv = await exchange.fetch_ohlcv(settings.trading_symbol, timeframe="1m", limit=1)
                    current_price = float(ticker_ohlcv[0][4]) if ticker_ohlcv else 0.0
                except Exception:
                    current_price = 0.0
                execution.set_in_position(btc_free, entry_price=current_price)
                logger.info("Position restored — {} BTC at ~{:.2f}, waiting for sell signal", btc_free, current_price or 0)
            else:
                logger.info("No position found — ready to buy")
        except Exception:
            logger.warning("Could not fetch balance for position restore — starting fresh")

        # Warm-up: fetch enough candles for the strategy windows
        ohlcv = await exchange.fetch_ohlcv(
            settings.trading_symbol, timeframe="1m", limit=100
        )
        logger.info("Initial OHLCV candles: {}", len(ohlcv))

        # ── Main loop ───────────────────────────────────────────────────────
        sync_counter = 0
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

            # Sync position with exchange every ~5 minutes
            sync_counter += 1
            if sync_counter >= 7 and execution.in_position:  # ~5 min (7*45s)
                sync_counter = 0
                try:
                    bal = await exchange.fetch_balance()
                    base = settings.trading_symbol.split("/")[0]
                    real_btc = float(bal.get(base, {}).get("free", 0))
                    if real_btc < 0.00001:
                        logger.info("Position sync — no BTC found. Resetting internal state.")
                        execution.reset_position()
                except Exception:
                    pass

            # 5. Execute
            if signal.action != "hold":
                current_price = float(ohlcv[-1][4])  # close price
                logger.info("EXECUTING {} @ {:.2f} — conf={:.2f}", signal.action.upper(), current_price, signal.confidence)
                await execution.execute_signal(signal, current_price, strategy_name=strategy.name, ohlcv=ohlcv)

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
