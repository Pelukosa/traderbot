"""Simple RSI strategy: sobrecompra/sobreventa."""
from __future__ import annotations

import numpy as np
from loguru import logger

from src.strategies.base import BaseStrategy, Signal


class RSI(BaseStrategy):
    name = "rsi"
    period: int = 14; oversold: float = 30.0; overbought: float = 70.0

    def __init__(self) -> None:
        self._prev_rsi: float | None = None

    @staticmethod
    def _compute_rsi(closes: np.ndarray, period: int = 14) -> float:
        deltas = np.diff(closes)
        gains = deltas.copy(); losses = deltas.copy()
        gains[gains<0]=0; losses[losses>0]=0; losses=-losses
        ag=np.mean(gains[-period:]) if len(gains)>=period else 0
        al=np.mean(losses[-period:]) if len(losses)>=period else 0
        if al==0: return 100.0
        return 100.0-(100.0/(1.0+ag/al))

    async def compute_signal(self, ohlcv: list[list[float]]) -> Signal:
        closes=np.array([c[4] for c in ohlcv],dtype=np.float64)
        if len(closes)<self.period+1: return Signal(metadata={"reason":"insufficient data"})
        rsi=self._compute_rsi(closes,self.period)
        meta={"rsi":round(rsi,2)}
        if self._prev_rsi is not None:
            if self._prev_rsi<=self.oversold and rsi>self.oversold:
                conf=min(abs(self.oversold-rsi)/self.oversold,1.0)
                self._prev_rsi=rsi
                return Signal(action="buy",confidence=abs(conf),metadata=meta)
            if self._prev_rsi>=self.overbought and rsi<self.overbought:
                conf=min(abs(rsi-self.overbought)/(100-self.overbought),1.0)
                self._prev_rsi=rsi
                return Signal(action="sell",confidence=abs(conf),metadata=meta)
        self._prev_rsi=rsi
        return Signal(metadata=meta)
