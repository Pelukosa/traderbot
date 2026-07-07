"""SMA Crossover: compra cuando fast cruza sobre slow."""
from __future__ import annotations

import numpy as np
from loguru import logger

from src.strategies.base import BaseStrategy, Signal
from src.config import settings


class SmaCrossover(BaseStrategy):
    name = "sma_crossover"

    def __init__(self) -> None:
        self.fast=settings.sma_fast; self.slow=settings.sma_slow
        self._prev_fast: float|None=None; self._prev_slow: float|None=None

    async def compute_signal(self, ohlcv: list[list[float]]) -> Signal:
        closes=np.array([c[4] for c in ohlcv],dtype=np.float64)
        if len(closes)<self.slow: return Signal(metadata={"reason":"insufficient data"})
        f=np.mean(closes[-self.fast:]); s=np.mean(closes[-self.slow:])
        m={"fast_ma":round(f,2),"slow_ma":round(s,2)}
        if self._prev_fast is not None and self._prev_slow is not None:
            if self._prev_fast<=self._prev_slow and f>s:
                c=min(abs(f-s)/s,1); self._prev_fast,self._prev_slow=f,s
                return Signal(action="buy",confidence=c,metadata=m)
            if self._prev_fast>=self._prev_slow and f<s:
                c=min(abs(f-s)/s,1); self._prev_fast,self._prev_slow=f,s
                return Signal(action="sell",confidence=c,metadata=m)
        self._prev_fast,self._prev_slow=f,s
        return Signal(metadata=m)
