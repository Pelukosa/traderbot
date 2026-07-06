from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
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
    highest_price: float = 0.0  # for trailing stop
    entry_time: datetime | None = None  # for max-duration check


MAX_POSITION_MINUTES: int = 60  # cierra automáticamente tras 1 hora


@dataclass
class TradeContext:
    """Context captured at entry time, passed to sell for ML logging."""
    trade_id: str
    strategy_name: str = ""
    entry_timestamp: str = ""
    # Buy-side histogram context
    buy_histogram: float = 0.0
    buy_valley_value: float = 0.0
    buy_valley_to_entry_diff: float = 0.0
    buy_reversal_slope: float = 0.0
    buy_macd_line: float = 0.0
    buy_signal_line: float = 0.0
    buy_price: float = 0.0
    # OHLVC snapshot at entry for market context
    entry_ohlcv: list[list[float]] | None = None


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
        self._trade_ctx: TradeContext | None = None

    @property
    def in_position(self) -> bool:
        return self._position is not None and self._position.active

    def set_in_position(self, btc_amount: float, entry_price: float = 0.0) -> None:
        """Restore a position after restart (buy already happened)."""
        price = entry_price
        sl = price * 0.985 if price > 0 else None  # 1.5% SL if we know price
        self._position = Position(
            symbol=settings.trading_symbol,
            side="long",
            entry_price=price or 0.001,  # avoid div by zero
            amount=btc_amount,
            stop_loss=sl,
            take_profit=None,
            highest_price=price or 0.001,
            entry_time=datetime.now(timezone.utc),
        )
        logger.info("Position restored — {:.8f} BTC at ~{:.2f}, waiting for sell signal", btc_amount, price or 0.001)

    def _simulate_only(self) -> bool:
        return settings.risk_mode == "simulation"

    @staticmethod
    def _market_context(ohlcv: list[list[float]]) -> dict[str, float]:
        """Compute market context from the OHLCV window."""
        if not ohlcv or len(ohlcv) < 20:
            return {"trend_20": 0.0, "volatility_20": 0.0, "volume_ratio": 1.0}

        closes = np.array([c[4] for c in ohlcv[-20:]])
        volumes = np.array([c[5] for c in ohlcv[-20:]])
        avg_volume = np.mean(volumes)

        trend = (closes[-1] - closes[0]) / closes[0] * 100  # % change over 20 candles
        volatility = float(np.std(closes / closes.mean())) * 100  # normalized std

        # Volume ratio: last candle vs avg of last 20
        vol_ratio = float(volumes[-1] / avg_volume) if avg_volume > 0 else 1.0

        return {
            "trend_20": round(trend, 2),
            "volatility_20": round(volatility, 2),
            "volume_ratio": round(vol_ratio, 2),
        }

    async def execute_signal(self, signal: Signal, price: float, strategy_name: str = "", ohlcv: list[list[float]] | None = None) -> None:
        """React to a strategy signal: open/close positions."""
        symbol = settings.trading_symbol

        # ── Sync position with real exchange balance ──
        # If we think we're in position but exchange has no BTC, reset
        # This handles manual sells from the exchange app
        if self.in_position:
            try:
                bal = await self._exchange.fetch_balance()
                base = symbol.split("/")[0]
                real_btc = float(bal.get(base, {}).get("free", 0))
                if real_btc < 0.00001:
                    logger.info("No BTC in exchange — position was closed externally. Resetting.")
                    self._position = None
                    self._trade_ctx = None
            except Exception:
                pass

        if signal.action == "buy" and not self.in_position:
            try:
                bal = await self._exchange.fetch_balance()
                eur_free = float(bal.get("EUR", {}).get("free", 0))
                logger.info("EUR balance: {:.2f}", eur_free)
            except Exception:
                eur_free = 0.0
                logger.warning("Could not fetch EUR balance, defaulting to 0")

            invest_eur = eur_free * 0.69
            if invest_eur < 5 and eur_free >= 5.5:
                invest_eur = eur_free * 0.95
                logger.info("69% ({:.2f}€) < 5€ mínimo — usando 95% ({:.2f}€)", eur_free * 0.69, invest_eur)
            if invest_eur < 5:
                logger.warning("Insufficient EUR ({} < 5€) — skipping buy", round(invest_eur, 2))
                return

            # Save trade context for ML logging at close time
            meta = signal.metadata
            self._trade_ctx = TradeContext(
                trade_id=uuid.uuid4().hex[:12],
                strategy_name=strategy_name,
                entry_timestamp=datetime.now(timezone.utc).isoformat(),
                buy_histogram=meta.get("histogram", 0.0),
                buy_valley_value=meta.get("valley_value", 0.0),
                buy_valley_to_entry_diff=meta.get("valley_to_entry_diff", 0.0),
                buy_reversal_slope=meta.get("reversal_slope", 0.0),
                buy_macd_line=meta.get("macd_line", 0.0),
                buy_signal_line=meta.get("signal_line", 0.0),
                buy_price=price,
                entry_ohlcv=ohlcv,
            )

            size = invest_eur / price
            logger.info("Investing {:.2f}€ → {:.6f} BTC @ {:.2f}", invest_eur, size, price)
            try:
                await self._open_long(symbol, size, price, eur_free)
            except OrderError as e:
                logger.error("Buy execution failed (will retry on next signal): {}", e)
                self._trade_ctx = None

        elif signal.action == "sell" and self.in_position:
            await self._close_position(symbol, ohlcv)

        # Check stop / take-profit on every tick if in position
        if self.in_position:
            await self._check_exits(price)

    async def _open_long(self, symbol: str, amount: float, price: float, eur_free: float = 0.0) -> None:
        # SL dinámico: basado en la profundidad del valle del histograma
        # Si el valle estaba en -15 y entramos en -12, el SL se calcula
        # como la mitad del recorrido desde entrada hasta valle
        valley_depth = 0.0
        if self._trade_ctx and self._trade_ctx.buy_valley_value != 0:
            valley_depth = abs(self._trade_ctx.buy_valley_value)

        # SL base: 0.6% fijo, se amplía si el valle es muy profundo (máx 2%)
        sl_pct = max(0.6, min(valley_depth * 0.08, 2.0))
        sl = price * (1 - sl_pct / 100.0)

        # Sin TP fijo — la venta la decide la señal del histograma
        tp = None

        logger.info("SL dinámico: {:.2f}% -> {:.2f}  (valley_depth={:.2f})", sl_pct, sl, valley_depth)

        if self._simulate_only():
            logger.info(
                "[SIM] BUY {} {} @ {:.2f}  SL={:.2f}  TP={:.2f}",
                amount, symbol, price, sl, tp,
            )
            log_trade("buy", symbol, amount, price, 0.0, balance_eur=eur_free)
        else:
            try:
                raw = await self._exchange.safe_call(
                    self._exchange.exchange.create_order,
                    symbol, "market", "buy", amount,
                )
                logger.info("RAW ORDER RESPONSE: {}", raw)
                if raw is None:
                    logger.error("Kraken returned None order — insufficient funds or rate limit")
                    return
                if not isinstance(raw, dict):
                    logger.error("Unexpected order response type: {} — {}", type(raw), raw)
                    return
                # Kraken wraps result in result/txid structure
                info = raw.get("info") or raw
                order_id = raw.get("id", "") or (info.get("txid", [""])[0] if isinstance(info.get("txid"), list) else "")
                fills = raw.get("fills", []) or info.get("fills", [])
                if fills:
                    avg_price = sum(f["price"] * f["amount"] for f in fills) / sum(f["amount"] for f in fills)
                    total_fee = sum(f.get("fee", {}).get("cost", 0) for f in fills)
                    filled_amount = sum(f["amount"] for f in fills)
                else:
                    avg_price = float(raw.get("price") or info.get("price", price))
                    total_fee = float(raw.get("fee", {}).get("cost", 0) or info.get("fee", 0))
                    filled_amount = float(raw.get("filled", amount) or info.get("filled", amount))
                log_trade("buy", symbol, filled_amount, avg_price, total_fee,
                          order_id=order_id, balance_eur=eur_free)
                amount = filled_amount
                price = avg_price
                # Update buy price in trade context with real fill
                if self._trade_ctx:
                    self._trade_ctx.buy_price = price
            except Exception as exc:
                raise OrderError(f"Buy order failed: {exc}") from exc

        self._position = Position(
            symbol=symbol,
            side="long",
            entry_price=price,
            amount=amount,
            stop_loss=sl,
            take_profit=tp,
            highest_price=price,
            entry_time=datetime.now(timezone.utc),
        )

    async def _close_position(self, symbol: str, current_ohlcv: list[list[float]] | None = None) -> None:
        if self._position is None:
            return

        entry_price = self._position.entry_price
        entry_amount = self._position.amount
        trade_ctx = self._trade_ctx
        entry_ts = trade_ctx.entry_timestamp if trade_ctx else ""
        strategy_name = trade_ctx.strategy_name if trade_ctx else ""
        trade_id = trade_ctx.trade_id if trade_ctx else ""
        exit_ts = datetime.now(timezone.utc).isoformat()

        # Compute duration
        duration_min = 0.0
        if entry_ts:
            try:
                entry_dt = datetime.fromisoformat(entry_ts)
                exit_dt = datetime.fromisoformat(exit_ts)
                duration_min = (exit_dt - entry_dt).total_seconds() / 60.0
            except Exception:
                pass

        if self._simulate_only():
            logger.info("[SIM] SELL {} {} @ market", entry_amount, symbol)
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
                    order = await self._exchange.create_market_sell_order(symbol, sell_amount)
                    logger.info("SELL ORDER PLACED: {}", order)
                    fills = order.get("fills", [])
                    if fills:
                        sell_price = sum(f["price"] * f["amount"] for f in fills) / sum(f["amount"] for f in fills)
                        sell_fee = sum(f.get("fee", {}).get("cost", 0) for f in fills)
                    else:
                        sell_price = float(order.get("price") or entry_price)
                        sell_fee = float(order.get("fee", {}).get("cost", 0))
                    pnl_eur = (sell_price - entry_price) * sell_amount
                    pnl_pct = ((sell_price - entry_price) / entry_price) * 100

                    try:
                        bal2 = await self._exchange.fetch_balance()
                        eur_after = float(bal2.get("EUR", {}).get("free", 0))
                    except Exception:
                        eur_after = 0.0

                    log_trade("sell", symbol, sell_amount, sell_price, sell_fee,
                              order_id=order.get("id", ""), pnl_eur=pnl_eur, pnl_pct=pnl_pct,
                              balance_eur=eur_after)

                    # — Full ML context logging —
                    # Buy-side context from trade_ctx (captured at entry)
                    buy_h = trade_ctx.buy_histogram if trade_ctx else 0.0
                    buy_v = trade_ctx.buy_valley_value if trade_ctx else 0.0
                    buy_v2e = trade_ctx.buy_valley_to_entry_diff if trade_ctx else 0.0
                    buy_rs = trade_ctx.buy_reversal_slope if trade_ctx else 0.0
                    buy_macd = trade_ctx.buy_macd_line if trade_ctx else 0.0
                    buy_sig = trade_ctx.buy_signal_line if trade_ctx else 0.0
                    buy_px = trade_ctx.buy_price if trade_ctx else entry_price

                    # Market context from entry OHLCV
                    entry_market = self._market_context(trade_ctx.entry_ohlcv) if trade_ctx and trade_ctx.entry_ohlcv else {}

                    # Market context from current OHLCV (for sell context)
                    sell_market = self._market_context(current_ohlcv) if current_ohlcv else {}
                    now = datetime.now(timezone.utc)

                    log_strategy_trade(
                        trade_id=trade_id or uuid.uuid4().hex[:12],
                        strategy=strategy_name or "unknown",
                        side="sell",
                        symbol=symbol,
                        entry_price=entry_price,
                        exit_price=sell_price,
                        entry_timestamp=entry_ts,
                        exit_timestamp=exit_ts,
                        duration_minutes=duration_min,
                        amount=sell_amount,
                        pnl_eur=pnl_eur,
                        pnl_pct=pnl_pct,
                        fee_eur=sell_fee,
                        # Buy context
                        buy_histogram=buy_h,
                        buy_valley_value=buy_v,
                        buy_valley_to_entry_diff=buy_v2e,
                        buy_reversal_slope=buy_rs,
                        buy_macd_line=buy_macd,
                        buy_signal_line=buy_sig,
                        buy_price=buy_px,
                        # Sell context (from signal metadata — not available here, use market)
                        sell_histogram=0.0,
                        sell_peak_value=0.0,
                        sell_peak_to_exit_diff=0.0,
                        sell_reversal_slope=0.0,
                        sell_macd_line=0.0,
                        sell_signal_line=0.0,
                        sell_price=sell_price,
                        # Market context
                        market_trend_20=entry_market.get("trend_20", sell_market.get("trend_20", 0.0)),
                        market_volatility_20=entry_market.get("volatility_20", sell_market.get("volatility_20", 0.0)),
                        market_volume_ratio=entry_market.get("volume_ratio", sell_market.get("volume_ratio", 1.0)),
                        hour_of_day=now.hour,
                        day_of_week=now.weekday(),
                    )

                    logger.info("Position closed  entry={:.2f}  exit={:.2f}  pnl={:+.2f}€ ({:+.2f}%%)  dur={:.0f}min",
                                entry_price, sell_price, pnl_eur, pnl_pct, duration_min)
            except Exception as exc:
                raise OrderError(f"Sell order failed: {exc}") from exc

        self._position = None
        self._trade_ctx = None

    def reset_position(self) -> None:
        """Force-reset internal position state (used after external manual sell)."""
        self._position = None
        self._trade_ctx = None
        logger.info("Position state reset")

    async def _check_exits(self, current_price: float) -> None:
        if self._position is None or not self._position.active:
            return

        pos = self._position
        now = datetime.now(timezone.utc)

        # ── 1. Trailing stop-loss ──
        # Si el precio sube, movemos el SL hacia arriba (nunca hacia abajo)
        if current_price > pos.highest_price and pos.entry_price > 1:  # entry_price=0.001 = unknown
            pos.highest_price = current_price
            # Trailing: SL se sitúa al 60% de la ganancia máxima
            # Si subió un 2%, el trailing SL está en +1.2%
            gain_pct = (current_price - pos.entry_price) / pos.entry_price * 100
            if gain_pct > 0.3:  # solo trailing si ha subido al menos 0.3%
                trail_pct = gain_pct * 0.6  # retiene el 60% de la ganancia
                new_sl = pos.entry_price * (1 + trail_pct / 100.0)
                if pos.stop_loss is None or new_sl > pos.stop_loss:
                    pos.stop_loss = new_sl
                    logger.info(
                        "Trailing SL actualizado: {:.2f}  (gain={:+.2f}%, trail={:.2f}%)",
                        new_sl, gain_pct, trail_pct,
                    )

        # ── 2. Stop-loss ──
        sl = pos.stop_loss
        if sl is not None and current_price <= sl:
            logger.warning(
                "STOP-LOSS HIT @ {:.2f}  (entry={:.2f}, sl={:.2f})",
                current_price, pos.entry_price, sl,
            )
            await self._close_position(pos.symbol)
            return

        # ── 3. Tiempo máximo en posición ──
        if pos.entry_time is not None:
            elapsed = (now - pos.entry_time).total_seconds() / 60.0
            if elapsed >= MAX_POSITION_MINUTES:
                logger.warning(
                    "Tiempo máximo alcanzado ({:.0f}min) — cerrando posición @ {:.2f}",
                    elapsed, current_price,
                )
                await self._close_position(pos.symbol)
                return

    async def emergency_close_all(self) -> None:
        if self._position:
            await self._close_position(self._position.symbol)
