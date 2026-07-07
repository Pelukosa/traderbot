"""Strategy registry — importa todas las estrategias y las pone disponibles."""
from src.strategies.base import BaseStrategy, Signal
from src.strategies.sma_crossover import SmaCrossover
from src.strategies.rsi import RSI
from src.strategies.macd import MACD
from src.strategies.macd_divergence import MACDDivergence
from src.strategies.macd_rsi_filtro import MACDRSIFiltro

STRATEGY_REGISTRY: dict[str, type[BaseStrategy]] = {
    "sma_crossover": SmaCrossover,
    "rsi": RSI,
    "macd": MACD,
    "macd_divergence": MACDDivergence,
    "macd_rsi_filtro": MACDRSIFiltro,
}
