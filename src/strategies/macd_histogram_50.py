"""MACD Histogram -50 Valley con doble entrada.

Estrategia:
  1. Señal de compra cuando el histograma MACD cruza por debajo de -50
     y la barra siguiente confirma reversión (cierra más alta/menos negativa).
  2. Primera compra: 45% del capital disponible.
  3. Si en las siguientes 4 velas el precio cae un 2% por debajo de la
     primera entrada → segunda compra con otro 45%.
  4. Take profit: +1% del precio de entrada (o precio ponderado si hubo 2ª compra).
  5. Stop loss: -1% del precio ponderado (solo se activa tras la 2ª compra).
"""
from __future__ import annotations

import numpy as np
from loguru import logger

from src.strategies.base import BaseStrategy, Signal


class MACDHistogram50(BaseStrategy):
    name = "macd_histogram_50"

    fast: int = 12
    slow: int = 26
    signal: int = 9
    cross_threshold: float = -50.0  # histogram must cross below this

    def __init__(self) -> None:
        self._awaiting_cross: bool = False
        self._cross_bar_value: float | None = None

    @staticmethod
    def _ema(data: np.ndarray, period: int) -> np.ndarray:
        result = np.empty_like(data)
        result[:] = np.nan
        result[period - 1] = np.mean(data[:period])
        multiplier = 2.0 / (period + 1)
        for i in range(period, len(data)):
            result[i] = (data[i] - result[i - 1]) * multiplier + result[i - 1]
        return result

    def _histogram(self, closes: np.ndarray) -> tuple[float, float, float]:
        ema_fast = self._ema(closes, self.fast)
        ema_slow = self._ema(closes, self.slow)
        macd_line = ema_fast - ema_slow
        valid = ~np.isnan(macd_line)
        macd_line = macd_line[valid]
        signal_line = self._ema(macd_line, self.signal)
        return (
            float(macd_line[-1]),
            float(signal_line[-1]),
            float(macd_line[-1] - signal_line[-1]),
        )

    async def compute_signal(self, ohlcv: list[list[float]]) -> Signal:
        closes = np.array([c[4] for c in ohlcv], dtype=np.float64)
        min_len = self.slow + self.signal + 5
        if len(closes) < min_len:
            return Signal(metadata={"reason": "insufficient data"})

        macd_val, sig_val, hist = self._histogram(closes)
        meta = {
            "histogram": round(hist, 2),
            "macd": round(macd_val, 2),
            "signal": round(sig_val, 2),
            "threshold": self.cross_threshold,
        }

        # Need previous histogram value to detect crossing
        if len(closes) < min_len + 1:
            return Signal(metadata=meta)

        # Get previous histogram
        prev_closes = closes[:-1]
        _, _, prev_hist = self._histogram(prev_closes)
        meta["prev_histogram"] = round(prev_hist, 2)

        # Step 1: Detect cross below -50
        if prev_hist > self.cross_threshold and hist < self.cross_threshold and hist < prev_hist:
            self._awaiting_cross = True
            self._cross_bar_value = hist
            logger.info(
                "MACD-HIST50 CROSS  hist={:.2f}  prev={:.2f}  (crossed below {})",
                hist, prev_hist, self.cross_threshold,
            )
            return Signal(metadata={**meta, "signal": "cross_detected", "awaiting_reversal": True})

        # Step 2: Confirm reversal (next bar higher than cross bar)
        if self._awaiting_cross and self._cross_bar_value is not None:
            if hist > self._cross_bar_value:
                confidence = min(abs(hist - self._cross_bar_value) / max(abs(hist), 1.0), 1.0)
                self._awaiting_cross = False
                self._cross_bar_value = None
                logger.info(
                    "MACD-HIST50 REVERSAL CONFIRMED  hist={:.2f}  cross_bar={:.2f}  conf={:.2f}",
                    hist, self._cross_bar_value if self._cross_bar_value else 0, confidence,
                )
                return Signal(
                    action="buy",
                    confidence=confidence,
                    metadata={**meta, "signal": "reversal_confirmed", "buy": True},
                )
            else:
                # Reversal not confirmed, reset
                logger.debug(
                    "MACD-HIST50 reversal NOT confirmed  hist={:.2f}  cross_bar={:.2f}",
                    hist, self._cross_bar_value,
                )
                self._awaiting_cross = False
                self._cross_bar_value = None

        return Signal(metadata=meta)
