"""MACD Divergencia: valle/pico del histograma."""
from __future__ import annotations

import numpy as np
from loguru import logger

from src.strategies.base import BaseStrategy, Signal


class MACDDivergence(BaseStrategy):
    name = "macd_divergence"
    fast: int = 12
    slow: int = 26
    signal: int = 9
    confirm_velas: int = 1
    min_histogram_abs: float = 50.0

    def __init__(self) -> None:
        self._hist_buffer: list[float] = []
        self._awaiting_action: str | None = None
        self._confirm_count: int = 0
        self._detected_valley: float | None = None
        self._detected_peak: float | None = None

    @staticmethod
    def _ema(data: np.ndarray, period: int) -> np.ndarray:
        result = np.empty_like(data); result[:] = np.nan
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
        return float(macd_line[-1]), float(signal_line[-1]), float(macd_line[-1] - signal_line[-1])

    async def compute_signal(self, ohlcv: list[list[float]]) -> Signal:
        closes = np.array([c[4] for c in ohlcv], dtype=np.float64)
        min_len = self.slow + self.signal + 5
        if len(closes) < min_len:
            return Signal(metadata={"reason": "insufficient data"})

        hist, macd_val, sig_val = self._calc(closes)
        self._hist_buffer.append(hist)
        if len(self._hist_buffer) > 5:
            self._hist_buffer = self._hist_buffer[-5:]

        meta = {"histogram": round(hist, 2), "macd": round(macd_val, 2), "signal": round(sig_val, 2),
                "buffer": [round(h, 2) for h in self._hist_buffer]}

        if len(self._hist_buffer) < 4:
            return Signal(metadata=meta)

        h = self._hist_buffer

        if self._awaiting_action:
            self._confirm_count += 1
            if self._confirm_count >= self.confirm_velas:
                action = self._awaiting_action
                confidence = min(abs(h[-1] - h[-2]) / max(abs(h[-1]), 1.0), 1.0) + 0.3
                confidence = min(confidence, 1.0)
                self._detected_valley = None; self._detected_peak = None
                self._awaiting_action = None; self._confirm_count = 0
                logger.info("MACD-DIV {}  hist={:.2f}  conf={:.2f}", action.upper(), hist, confidence)
                return Signal(action=action, confidence=confidence, metadata={**meta, "signal": f"{action}_confirmed"})
            return Signal(metadata={**meta, "confirming": self._awaiting_action, "count": self._confirm_count})

        if h[-4] >= h[-3] > h[-2] < h[-1] and h[-2] < 0 and h[-1] < 0 and abs(h[-2]) >= self.min_histogram_abs:
            self._detected_valley = h[-2]; self._awaiting_action = "buy"; self._confirm_count = 1
            logger.info("MACD-DIV VALLE  hist={:.2f}  bottom={:.2f}", hist, h[-2])
            return Signal(metadata={**meta, "signal": "valley_detected", "confirming": "buy", "count": 1})

        if h[-4] <= h[-3] < h[-2] > h[-1] and h[-2] > 0 and h[-1] > 0 and abs(h[-2]) >= self.min_histogram_abs:
            self._detected_peak = h[-2]; self._awaiting_action = "sell"; self._confirm_count = 1
            logger.info("MACD-DIV PICO  hist={:.2f}  peak={:.2f}", hist, h[-2])
            return Signal(metadata={**meta, "signal": "peak_detected", "confirming": "sell", "count": 1})

        return Signal(metadata=meta)
