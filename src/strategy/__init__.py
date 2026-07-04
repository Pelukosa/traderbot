from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

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
    """Abstract base for all trading strategies.

    Subclasses must implement :meth:`compute_signal` which receives the latest
    OHLCV data and returns a :class:`Signal`.
    """

    def __init__(self) -> None:
        self.name = self.__class__.__name__

    @abstractmethod
    async def compute_signal(self, ohlcv: list[list[float]]) -> Signal:
        ...


class SmaCrossover(BaseStrategy):
    """Simple moving-average crossover strategy.

    Buys when the fast SMA crosses **above** the slow SMA, sells when it crosses
    **below**.  Window sizes are read from ``settings``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.fast = settings.sma_fast
        self.slow = settings.sma_slow
        self._prev_fast: float | None = None
        self._prev_slow: float | None = None

    async def compute_signal(self, ohlcv: list[list[float]]) -> Signal:
        # ohlcv format: [[timestamp, open, high, low, close, volume], …]
        closes = np.array([c[4] for c in ohlcv], dtype=np.float64)

        if len(closes) < self.slow:
            return Signal(action="hold", confidence=0.0, metadata={"reason": "insufficient data"})

        fast_ma = np.mean(closes[-self.fast :])
        slow_ma = np.mean(closes[-self.slow :])

        signal = Signal(action="hold", confidence=0.0, metadata={"fast": fast_ma, "slow": slow_ma})

        if self._prev_fast is not None and self._prev_slow is not None:
            # Cross above  → buy
            if self._prev_fast <= self._prev_slow and fast_ma > slow_ma:
                signal.action = "buy"
                signal.confidence = min(abs(fast_ma - slow_ma) / slow_ma, 1.0)
                logger.info("BUY signal  fast={:.2f}  slow={:.2f}", fast_ma, slow_ma)
            # Cross below → sell
            elif self._prev_fast >= self._prev_slow and fast_ma < slow_ma:
                signal.action = "sell"
                signal.confidence = min(abs(fast_ma - slow_ma) / slow_ma, 1.0)
                logger.info("SELL signal  fast={:.2f}  slow={:.2f}", fast_ma, slow_ma)

        self._prev_fast = fast_ma
        self._prev_slow = slow_ma
        return signal
