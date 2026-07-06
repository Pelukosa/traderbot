#!/usr/bin/env python3
"""traderbot — Asynchronous crypto trading bot (1h timeframe).

Usage
-----
    python main.py                  # reads .env
    RISK_MODE=paper python main.py  # override via env

DUAL LOOP:
  - Buy signals: checked once per hour at candle close.
  - Sell signals/SL/trailing: checked every 5 minutes for fast exits.
"""
from __future__ import annotations

import asyncio
import sys
import time

from loguru import logger

from src.config import settings
from src.exchange import ExchangeManager
from src.execution import ExecutionManager
from src.logger import setup_logger
from src.strategy import STRATEGY_REGISTRY

from src.strategy import Signal

TIMEFRAME = "1h"
FAST_TICK = 5 * 60  # 5 min for sell/trailing/SL checks

# ── Estrategia actual ──
STRATEGY_NAME = "macd_rsi_filtro"


async def main_loop() -> None:
    setup_logger()
    logger.info("traderbot starting — mode={}  timeframe={}  fast-tick={}s",
                settings.risk_mode, TIMEFRAME, FAST_TICK)

    exchange = ExchangeManager()
    strategy_cls = STRATEGY_REGISTRY.get(STRATEGY_NAME)
    if strategy_cls is None:
        logger.error("Unknown strategy: {}", STRATEGY_NAME)
        sys.exit(1)

    strategy = strategy_cls()
    execution = ExecutionManager(exchange)

    # Track last hour we checked for buy signals
    last_buy_check_hour = -1

    try:
        await exchange.load_markets()
        logger.info("Markets loaded — connected to {}", settings.exchange_id)

        # ── Restore position on restart ──
        try:
            bal = await exchange.fetch_balance()
            base = settings.trading_symbol.split("/")[0]
            btc_free = float(bal.get(base, {}).get("free", 0))
            eur_free = float(bal.get("EUR", {}).get("free", 0))
            logger.info("Restore check — {}: {:.8f}  EUR: {:.2f}", base, btc_free, eur_free)
            if btc_free >= 0.00001:
                try:
                    ticker_ohlcv = await exchange.fetch_ohlcv(settings.trading_symbol, timeframe=TIMEFRAME, limit=1)
                    current_price = float(ticker_ohlcv[0][4]) if ticker_ohlcv else 0.0
                except Exception:
                    current_price = 0.0
                execution.set_in_position(btc_free, entry_price=current_price)
                logger.info("Position restored — {} BTC at ~{:.2f}, waiting for sell signal",
                            btc_free, current_price or 0)
            else:
                logger.info("No position found — ready to buy")
        except Exception:
            logger.warning("Could not fetch balance for position restore — starting fresh")

        # Warm-up: fetch enough candles
        ohlcv = await exchange.fetch_ohlcv(
            settings.trading_symbol, timeframe=TIMEFRAME, limit=100
        )
        logger.info("Initial OHLCV candles: {}  ({})", len(ohlcv), TIMEFRAME)
        ticker_price = float(ohlcv[-1][4]) if ohlcv else 0.0

        # ── Main loop ───────────────────────────────────────────────────────
        while True:
            try:
                # 1. Kill switch check
                if exchange.kill_switch_triggered:
                    logger.critical("Kill switch active — stopping traderbot")
                    await execution.emergency_close_all()
                    break

                # 2. Reconnect if needed
                if exchange.consecutive_errors > 0:
                    logger.warning("Reconnecting after {} errors …", exchange.consecutive_errors)
                    await exchange.reconnect()
                    ohlcv = await exchange.fetch_ohlcv(
                        settings.trading_symbol, timeframe=TIMEFRAME, limit=100
                    )
                    continue

                # 3. Fetch fresh 1h OHLCV data
                new_ohlcv = await exchange.fetch_ohlcv(
                    settings.trading_symbol, timeframe=TIMEFRAME, limit=100
                )
                if new_ohlcv:
                    ohlcv = new_ohlcv[-200:] if len(new_ohlcv) > 200 else new_ohlcv
                current_price = float(ohlcv[-1][4])

                # 4. Get a 1m ticker for fast checks (SL/trailing)
                try:
                    ticker_1m = await exchange.fetch_ohlcv(
                        settings.trading_symbol, timeframe="1m", limit=1
                    )
                    ticker_price = float(ticker_1m[0][4]) if ticker_1m else current_price
                except Exception:
                    ticker_price = current_price

                # 5. Check exits (SL / trailing / max time) — EVERY TICK (5 min)
                if execution.in_position:
                    await execution._check_exits(ticker_price)

                    # Also check for a sell signal from the strategy
                    try:
                        signal = await strategy.compute_signal(ohlcv)
                        if signal.action == "sell":
                            logger.info(
                                "SELL SIGNAL (fast tick) @ {:.2f} — conf={:.2f}",
                                ticker_price, signal.confidence,
                            )
                            await execution.execute_signal(
                                signal, ticker_price,
                                strategy_name=strategy.name, ohlcv=ohlcv,
                            )
                    except Exception:
                        logger.exception("Sell signal error (fast tick)")
                        pass

                # 6. Sync position with exchange
                if execution.in_position:
                    try:
                        bal = await exchange.fetch_balance()
                        base = settings.trading_symbol.split("/")[0]
                        real_btc = float(bal.get(base, {}).get("free", 0))
                        if real_btc < 0.00001:
                            logger.info("Position sync — no BTC found. Resetting.")
                            execution.reset_position()
                    except Exception:
                        pass

                # 7. Buy signal check — ONLY at hour boundary (candle close)
                current_hour = int(time.time() // 3600)
                if current_hour != last_buy_check_hour:
                    last_buy_check_hour = current_hour
                    logger.info("Hourly buy check — {} candles loaded", len(ohlcv))

                    if not execution.in_position:
                        try:
                            signal = await strategy.compute_signal(ohlcv)
                            if signal.action == "buy":
                                logger.info(
                                    "BUY SIGNAL (hourly) @ {:.2f} — conf={:.2f}",
                                    current_price, signal.confidence,
                                )
                                await execution.execute_signal(
                                    signal, current_price,
                                    strategy_name=strategy.name, ohlcv=ohlcv,
                                )
                        except Exception:
                            logger.exception("Buy signal error")
                    else:
                        logger.debug("In position — skipping buy check")

                # 8. Sleep
                await asyncio.sleep(FAST_TICK)

            except Exception:
                logger.exception("Error in main loop tick")
                await asyncio.sleep(30)

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
