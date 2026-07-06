from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
from loguru import logger

from src.config import settings


@dataclass
class Signal:
    """Trading signal produced by a strategy."""

    action: str = "hold"  # "buy" | "sell" | "hold"
    confidence: float = 0.0  # 0.0 – 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseStrategy(ABC):
    """Abstract base for all trading strategies."""

    def __init__(self) -> None:
        self.name = self.__class__.__name__

    @abstractmethod
    async def compute_signal(self, ohlcv: list[list[float]]) -> Signal:
        ...


# ── SMA Crossover ──────────────────────────────────────────────────────────


class SmaCrossover(BaseStrategy):
    """SMA crossover: buy when fast MA crosses above slow MA, sell on cross below."""

    def __init__(self) -> None:
        super().__init__()
        self.fast = settings.sma_fast
        self.slow = settings.sma_slow
        self._prev_fast: float | None = None
        self._prev_slow: float | None = None

    async def compute_signal(self, ohlcv: list[list[float]]) -> Signal:
        closes = np.array([c[4] for c in ohlcv], dtype=np.float64)
        if len(closes) < self.slow:
            return Signal(metadata={"reason": "insufficient data"})

        fast_ma = np.mean(closes[-self.fast :])
        slow_ma = np.mean(closes[-self.slow :])
        meta: dict[str, Any] = {"fast_ma": round(fast_ma, 2), "slow_ma": round(slow_ma, 2)}

        if self._prev_fast is not None and self._prev_slow is not None:
            if self._prev_fast <= self._prev_slow and fast_ma > slow_ma:
                confidence = min(abs(fast_ma - slow_ma) / slow_ma, 1.0)
                logger.info("SMA BUY  fast={:.2f}  slow={:.2f}  conf={:.2f}", fast_ma, slow_ma, confidence)
                self._prev_fast, self._prev_slow = fast_ma, slow_ma
                return Signal(action="buy", confidence=confidence, metadata=meta)
            elif self._prev_fast >= self._prev_slow and fast_ma < slow_ma:
                confidence = min(abs(fast_ma - slow_ma) / slow_ma, 1.0)
                logger.info("SMA SELL  fast={:.2f}  slow={:.2f}  conf={:.2f}", fast_ma, slow_ma, confidence)
                self._prev_fast, self._prev_slow = fast_ma, slow_ma
                return Signal(action="sell", confidence=confidence, metadata=meta)

        self._prev_fast, self._prev_slow = fast_ma, slow_ma
        return Signal(metadata=meta)


# ── RSI ────────────────────────────────────────────────────────────────────


class RSI(BaseStrategy):
    """Relative Strength Index.

    Buys when RSI crosses below *oversold_threshold* and back above.
    Sells when RSI crosses above *overbought_threshold* and back below.
    """

    period: int = 14
    oversold: float = 30.0
    overbought: float = 70.0

    def __init__(self) -> None:
        super().__init__()
        self._prev_rsi: float | None = None

    @staticmethod
    def _compute_rsi(closes: np.ndarray, period: int = 14) -> float:
        deltas = np.diff(closes)
        gains = deltas.copy()
        losses = deltas.copy()
        gains[gains < 0] = 0.0
        losses[losses > 0] = 0.0
        losses = -losses

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    async def compute_signal(self, ohlcv: list[list[float]]) -> Signal:
        closes = np.array([c[4] for c in ohlcv], dtype=np.float64)
        if len(closes) < self.period + 1:
            return Signal(metadata={"reason": "insufficient data"})

        rsi = self._compute_rsi(closes, self.period)
        meta: dict[str, Any] = {"rsi": round(rsi, 2)}

        if self._prev_rsi is not None:
            # Oversold bounce → buy
            if self._prev_rsi <= self.oversold and rsi > self.oversold:
                confidence = min((self.oversold - rsi) / self.oversold * -1, 1.0)
                logger.info("RSI BUY  rsi={:.2f}  conf={:.2f}", rsi, confidence)
                self._prev_rsi = rsi
                return Signal(action="buy", confidence=abs(confidence), metadata=meta)
            # Overbought drop → sell
            elif self._prev_rsi >= self.overbought and rsi < self.overbought:
                confidence = min((rsi - self.overbought) / (100 - self.overbought) * -1, 1.0)
                logger.info("RSI SELL  rsi={:.2f}  conf={:.2f}", rsi, confidence)
                self._prev_rsi = rsi
                return Signal(action="sell", confidence=abs(confidence), metadata=meta)

        self._prev_rsi = rsi
        return Signal(metadata=meta)


# ── MACD ────────────────────────────────────────────────────────────────────


class MACD(BaseStrategy):
    """Moving Average Convergence Divergence.

    Signal line: 9-period EMA of MACD line.
    Buy when MACD line crosses above signal line.
    Sell when MACD line crosses below signal line.
    """

    fast: int = 12
    slow: int = 26
    signal: int = 9

    def __init__(self) -> None:
        super().__init__()
        self._prev_macd: float | None = None
        self._prev_signal_line: float | None = None

    @staticmethod
    def _ema(data: np.ndarray, period: int) -> np.ndarray:
        result = np.empty_like(data)
        result[:] = np.nan
        result[period - 1] = np.mean(data[:period])
        multiplier = 2.0 / (period + 1)
        for i in range(period, len(data)):
            result[i] = (data[i] - result[i - 1]) * multiplier + result[i - 1]
        return result

    def _calc(self, closes: np.ndarray) -> tuple[float, float, float]:
        ema_fast = self._ema(closes, self.fast)
        ema_slow = self._ema(closes, self.slow)
        macd_line = ema_fast - ema_slow
        valid = ~np.isnan(macd_line)
        macd_line = macd_line[valid]
        signal_line = self._ema(macd_line, self.signal)
        return macd_line[-1], signal_line[-1], macd_line[-1] - signal_line[-1]

    async def compute_signal(self, ohlcv: list[list[float]]) -> Signal:
        closes = np.array([c[4] for c in ohlcv], dtype=np.float64)
        min_len = self.slow + self.signal + 2
        if len(closes) < min_len:
            return Signal(metadata={"reason": "insufficient data"})

        macd_val, sig_val, hist_val = self._calc(closes)
        meta: dict[str, Any] = {"macd": round(macd_val, 2), "signal": round(sig_val, 2), "histogram": round(hist_val, 2)}

        if self._prev_macd is not None and self._prev_signal_line is not None:
            # Cross above → buy
            if self._prev_macd <= self._prev_signal_line and macd_val > sig_val:
                confidence = min(abs(macd_val - sig_val) / max(abs(sig_val), 1.0), 1.0)
                logger.info("MACD BUY  macd={:.2f}  signal={:.2f}  conf={:.2f}", macd_val, sig_val, confidence)
                self._prev_macd, self._prev_signal_line = macd_val, sig_val
                return Signal(action="buy", confidence=confidence, metadata=meta)
            # Cross below → sell
            elif self._prev_macd >= self._prev_signal_line and macd_val < sig_val:
                confidence = min(abs(macd_val - sig_val) / max(abs(sig_val), 1.0), 1.0)
                logger.info("MACD SELL  macd={:.2f}  signal={:.2f}  conf={:.2f}", macd_val, sig_val, confidence)
                self._prev_macd, self._prev_signal_line = macd_val, sig_val
                return Signal(action="sell", confidence=confidence, metadata=meta)

        self._prev_macd, self._prev_signal_line = macd_val, sig_val
        return Signal(metadata=meta)


# ── MACD Histogram Valley/Peak (con confirmación) ─────────────────────────


class MACDDivergence(BaseStrategy):
    """MACD Histogram Valley & Peak detection con confirmación de 2 velas.

    En lugar de anticipar el cruce, detecta el **valle** (mínimo) o el **pico**
    (máximo) del histograma y entra **2 velas después** de que la tendencia se
    revierta.

    CÓMO FUNCIONA
    -------------
    El histograma (MACD - línea de señal) forma montañas y valles.

    **COMPRA (valle):**
      El histograma viene bajando (cada vez más negativo): -1, -5, -12, -15.
      En -15 toca fondo (valle) y empieza a subir: -14, -12, -8…
      → Entramos en la 2ª vela después del valle (en el -12 del ejemplo),
         asumiendo que el valle ya pasó y la tendencia alcista se confirma.

    **VENTA (pico):**
      El histograma viene subiendo (cada vez más positivo): 5, 12, 18, 21.
      En 21 toca techo (pico) y empieza a bajar: 18, 17, 14…
      → Vendemos en la 2ª vela después del pico (en el 17 del ejemplo).

    Esta estrategia NO espera al cruce MACD — gana 5-10 velas respecto al
    MACD clásico. Es la única estrategia que se ejecuta en solitario.
    """

    fast: int = 12
    slow: int = 26
    signal: int = 9
    confirm_velas: int = 2  # velas después del valle/pico para entrar

    def __init__(self) -> None:
        super().__init__()
        self._hist_buffer: list[float] = []  # rolling buffer of histogram values
        self._awaiting_action: str | None = None  # "buy" or "sell" pending
        self._confirm_count: int = 0

    @staticmethod
    def _ema(data: np.ndarray, period: int) -> np.ndarray:
        result = np.empty_like(data)
        result[:] = np.nan
        result[period - 1] = np.mean(data[:period])
        multiplier = 2.0 / (period + 1)
        for i in range(period, len(data)):
            result[i] = (data[i] - result[i - 1]) * multiplier + result[i - 1]
        return result

    def _histogram(self, closes: np.ndarray) -> float:
        ema_fast = self._ema(closes, self.fast)
        ema_slow = self._ema(closes, self.slow)
        macd_line = ema_fast - ema_slow
        valid = ~np.isnan(macd_line)
        macd_line = macd_line[valid]
        signal_line = self._ema(macd_line, self.signal)
        return float(macd_line[-1] - signal_line[-1])

    async def compute_signal(self, ohlcv: list[list[float]]) -> Signal:
        closes = np.array([c[4] for c in ohlcv], dtype=np.float64)
        min_len = self.slow + self.signal + 5
        if len(closes) < min_len:
            return Signal(metadata={"reason": "insufficient data"})

        hist = self._histogram(closes)
        self._hist_buffer.append(hist)

        # Keep last 5 values for detection
        if len(self._hist_buffer) > 5:
            self._hist_buffer = self._hist_buffer[-5:]

        meta: dict[str, Any] = {"histogram": round(hist, 2), "buffer": [round(h, 2) for h in self._hist_buffer]}

        # Need at least 3 values to detect anything
        if len(self._hist_buffer) < 4:
            return Signal(metadata=meta)

        # ── Check if we're in confirmation wait mode ──
        if self._awaiting_action:
            self._confirm_count += 1
            if self._confirm_count >= self.confirm_velas:
                # Execute the pending action
                action = self._awaiting_action
                self._awaiting_action = None
                self._confirm_count = 0
                confidence = min(abs(self._hist_buffer[-1] - self._hist_buffer[-2]) / max(abs(self._hist_buffer[-1]), 1.0), 1.0) + 0.3
                confidence = min(confidence, 1.0)
                signal_type = "valley_confirmed" if action == "buy" else "peak_confirmed"
                logger.info(
                    "MACD-DIV {}  hist={:.2f}  conf={:.2f}  type={}",
                    action.upper(), hist, confidence, signal_type,
                )
                return Signal(action=action, confidence=confidence, metadata={
                    **meta, "signal": signal_type,
                })
            else:
                # Still confirming, hold
                return Signal(metadata={**meta, "confirming": self._awaiting_action, "count": self._confirm_count})

        # ── DETECTAR VALLE (compra) ──
        # El histograma venía bajando (h3 > h2 > h1) y luego H1 es el valle:
        #   h[0] > h[1] > h[2] (valle) < h[3]   → valle detectado en h[2]
        #   La 1ª reversión es h[3], entramos tras h[4] (2 velas después)
        h = self._hist_buffer
        if len(h) >= 4:
            # Detectar valle: h[-4] >= h[-3] > h[-2] < h[-1]
            # (el valle está en h[-2], h[-1] es la 1ª reversión)
            if h[-4] >= h[-3] > h[-2] < h[-1] and h[-2] < 0 and h[-1] < 0:
                # Valle detectado en h[-2], h[-1] es reversión (vela 1)
                # Esperamos 2 velas de confirmación
                self._awaiting_action = "buy"
                self._confirm_count = 1  # h[-1] is the 1st confirmation candle
                logger.info(
                    "MACD-DIV VALLE detectado  hist={:.2f}  bottom={:.2f}  "
                    "esperando {} velas…",
                    hist, h[-2], self.confirm_velas,
                )
                return Signal(metadata={**meta, "signal": "valley_detected", "confirming": "buy", "count": 1})

            # Detectar pico: h[-4] <= h[-3] < h[-2] > h[-1]
            if h[-4] <= h[-3] < h[-2] > h[-1] and h[-2] > 0 and h[-1] > 0:
                # Pico detectado en h[-2], h[-1] es reversión (vela 1)
                self._awaiting_action = "sell"
                self._confirm_count = 1
                logger.info(
                    "MACD-DIV PICO detectado  hist={:.2f}  peak={:.2f}  "
                    "esperando {} velas…",
                    hist, h[-2], self.confirm_velas,
                )
                return Signal(metadata={**meta, "signal": "peak_detected", "confirming": "sell", "count": 1})

        return Signal(metadata=meta)


# ── Strategy registry ───────────────────────────────────────────────────────

STRATEGY_REGISTRY: dict[str, type[BaseStrategy]] = {
    "sma_crossover": SmaCrossover,
    "rsi": RSI,
    "macd": MACD,
    "macd_divergence": MACDDivergence,
}


class MajorityVote(BaseStrategy):
    """Meta-strategy: runs SMA, RSI and MACD in parallel and takes the
    majority decision.  At least 2 of 3 must agree on buy or sell.

    Confidence = average confidence of agreeing strategies.
    """

    def __init__(self) -> None:
        super().__init__()
        self._strategies: list[BaseStrategy] = [
            SmaCrossover(),
            RSI(),
            MACD(),
        ]

    async def compute_signal(self, ohlcv: list[list[float]]) -> Signal:
        votes: dict[str, list[float]] = {"buy": [], "sell": []}
        details: dict[str, Any] = {}

        for s in self._strategies:
            sig = await s.compute_signal(ohlcv)
            details[s.name] = {"action": sig.action, "confidence": round(sig.confidence, 2)}
            if sig.action in ("buy", "sell"):
                votes[sig.action].append(sig.confidence)

        meta: dict[str, Any] = {"votes": details}

        if len(votes["buy"]) >= 2:
            confidence = sum(votes["buy"]) / len(votes["buy"])
            logger.info(
                "MAJORITY BUY  ({}/{})  avg_conf={:.2f}  {}",
                len(votes["buy"]), len(self._strategies), confidence,
                {k: v["action"] for k, v in details.items()},
            )
            return Signal(action="buy", confidence=confidence, metadata=meta)

        if len(votes["sell"]) >= 2:
            confidence = sum(votes["sell"]) / len(votes["sell"])
            logger.info(
                "MAJORITY SELL  ({}/{})  avg_conf={:.2f}  {}",
                len(votes["sell"]), len(self._strategies), confidence,
                {k: v["action"] for k, v in details.items()},
            )
            return Signal(action="sell", confidence=confidence, metadata=meta)

        return Signal(metadata=meta)


STRATEGY_REGISTRY["majority_vote"] = MajorityVote
