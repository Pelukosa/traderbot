"""Base class for all strategies."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Signal:
    """Trading signal produced by a strategy."""
    action: str  # "buy" | "sell" | "hold"
    confidence: float
    metadata: dict[str, Any]

    def __init__(self, action: str = "hold", confidence: float = 0.0, metadata: dict | None = None):
        self.action = action
        self.confidence = confidence
        self.metadata = metadata or {}


class BaseStrategy(ABC):
    name: str = "base"

    @abstractmethod
    async def compute_signal(self, ohlcv: list[list[float]]) -> Signal:
        ...
