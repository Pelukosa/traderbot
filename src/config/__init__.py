from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Exchange
    exchange_id: str = "kraken"
    exchange_sandbox: bool = True
    api_key: str = ""
    api_secret: str = ""

    # Risk
    risk_mode: str = "simulation"  # simulation | paper | live
    max_position_size_btc: float = 0.01
    stop_loss_percent: float = 2.0
    take_profit_percent: float = 4.0

    # Strategy
    strategy: str = "sma_crossover"
    sma_fast: int = 10
    sma_slow: int = 30
    trading_symbol: str = "BTC/EUR"

    # Kill switch
    max_consecutive_errors: int = 3

    @property
    def is_live(self) -> bool:
        return self.risk_mode == "live"


settings = Settings()
