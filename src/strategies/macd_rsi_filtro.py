"""MACD Histogram Valley + RSI < umbral (hibrida)."""
from __future__ import annotations

import numpy as np
from loguru import logger

from src.strategies.base import BaseStrategy, Signal


class MACDRSIFiltro(BaseStrategy):
    name = "macd_rsi_filtro"
    fast: int = 12; slow: int = 26; signal: int = 9
    confirm_velas: int = 1; min_histogram_abs: float = 60.0
    rsi_max: float = 40.0
    sl_percent: float = 4.95; trailing_min_gain: float = 1.05; trail_retain: float = 0.6

    def __init__(self) -> None:
        self._hist_buffer: list[float] = []
        self._awaiting_action: str | None = None
        self._confirm_count: int = 0
        self._detected_valley: float | None = None
        self._detected_peak: float | None = None

    @staticmethod
    def _ema(data: np.ndarray, period: int) -> np.ndarray:
        result = np.empty_like(data); result[:]=np.nan
        result[period-1]=np.mean(data[:period])
        m=2.0/(period+1)
        for i in range(period,len(data)): result[i]=(data[i]-result[i-1])*m+result[i-1]
        return result

    @staticmethod
    def _compute_rsi(closes: np.ndarray, period: int=14) -> float:
        deltas=np.diff(closes)
        gains=deltas.copy(); losses=deltas.copy()
        gains[gains<0]=0; losses[losses>0]=0; losses=-losses
        ag=np.mean(gains[-period:]) if len(gains)>=period else 0
        al=np.mean(losses[-period:]) if len(losses)>=period else 0
        if al==0: return 100.0
        return 100.0-(100.0/(1.0+ag/al))

    def _histogram(self, closes: np.ndarray) -> float:
        ef=self._ema(closes,self.fast); es=self._ema(closes,self.slow)
        m=ef-es; v=~np.isnan(m); m=m[v]
        return float(m[-1]-self._ema(m,self.signal)[-1])

    async def compute_signal(self, ohlcv: list[list[float]]) -> Signal:
        closes=np.array([c[4] for c in ohlcv],dtype=np.float64)
        if len(closes)<self.slow+self.signal+5:
            return Signal(metadata={"reason":"insufficient data"})
        hist=self._histogram(closes); rsi=self._compute_rsi(closes)
        self._hist_buffer.append(hist)
        if len(self._hist_buffer)>5: self._hist_buffer=self._hist_buffer[-5:]
        meta={"histogram":round(hist,2),"rsi":round(rsi,2),"buffer":[round(h,2) for h in self._hist_buffer]}
        if len(self._hist_buffer)<4: return Signal(metadata=meta)
        h=self._hist_buffer

        if self._awaiting_action:
            self._confirm_count+=1
            if self._confirm_count>=self.confirm_velas:
                a=self._awaiting_action; conf=min(min(abs(h[-1]-h[-2])/max(abs(h[-1]),1),1)+0.3,1)
                self._detected_valley=None;self._detected_peak=None;self._awaiting_action=None;self._confirm_count=0
                logger.info("MACD-RSI {}  hist={:.2f}  rsi={:.2f}  conf={:.2f}",a.upper(),hist,rsi,conf)
                return Signal(action=a,confidence=conf,metadata={**meta,"signal":f"{a}_confirmed"})
            return Signal(metadata={**meta,"confirming":self._awaiting_action,"count":self._confirm_count})

        if h[-4]>=h[-3]>h[-2]<h[-1] and h[-2]<0 and h[-1]<0 and abs(h[-2])>=self.min_histogram_abs:
            if rsi<self.rsi_max:
                self._detected_valley=h[-2];self._awaiting_action="buy";self._confirm_count=1
                logger.info("MACD-RSI VALLE+RSI<{}  hist={:.2f}  rsi={:.2f}",self.rsi_max,hist,rsi)
                return Signal(metadata={**meta,"signal":"valley_rsi_detected","confirming":"buy","count":1})
            logger.debug("MACD valley but RSI={:.1f}>={:.1f}",rsi,self.rsi_max)

        if h[-4]<=h[-3]<h[-2]>h[-1] and h[-2]>0 and h[-1]>0 and abs(h[-2])>=self.min_histogram_abs:
            self._detected_peak=h[-2];self._awaiting_action="sell";self._confirm_count=1
            logger.info("MACD-RSI PICO  hist={:.2f}  rsi={:.2f}",hist,rsi)
            return Signal(metadata={**meta,"signal":"peak_detected","confirming":"sell","count":1})
        return Signal(metadata=meta)
