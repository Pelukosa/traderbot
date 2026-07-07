"""MACD clasico: compra/vende por cruce de lineas."""
from __future__ import annotations

import numpy as np
from loguru import logger

from src.strategies.base import BaseStrategy, Signal


class MACD(BaseStrategy):
    name = "macd"
    fast: int = 12; slow: int = 26; signal: int = 9

    def __init__(self) -> None:
        self._prev_macd: float | None = None
        self._prev_signal: float | None = None

    @staticmethod
    def _ema(data: np.ndarray, period: int) -> np.ndarray:
        r=np.empty_like(data);r[:]=np.nan
        r[period-1]=np.mean(data[:period])
        m=2.0/(period+1)
        for i in range(period,len(data)): r[i]=(data[i]-r[i-1])*m+r[i-1]
        return r

    def _calc(self, closes: np.ndarray) -> tuple[float, float, float]:
        ef=self._ema(closes,self.fast); es=self._ema(closes,self.slow)
        m=ef-es;v=~np.isnan(m);m=m[v]
        s=self._ema(m,self.signal)
        return m[-1], s[-1], m[-1]-s[-1]

    async def compute_signal(self, ohlcv: list[list[float]]) -> Signal:
        closes=np.array([c[4] for c in ohlcv],dtype=np.float64)
        if len(closes)<self.slow+self.signal+2: return Signal(metadata={"reason":"insufficient data"})
        m,s,h=self._calc(closes)
        meta={"macd":round(m,2),"signal":round(s,2),"histogram":round(h,2)}
        if self._prev_macd is not None and self._prev_signal is not None:
            if self._prev_macd<=self._prev_signal and m>s:
                conf=min(abs(m-s)/max(abs(s),1),1)
                self._prev_macd,self._prev_signal=m,s
                return Signal(action="buy",confidence=conf,metadata=meta)
            if self._prev_macd>=self._prev_signal and m<s:
                conf=min(abs(m-s)/max(abs(s),1),1)
                self._prev_macd,self._prev_signal=m,s
                return Signal(action="sell",confidence=conf,metadata=meta)
        self._prev_macd,self._prev_signal=m,s
        return Signal(metadata=meta)
