"""
Predicción en producción — ligero, rápido, sin dependencias de entrenamiento.

Uso típico:
    from ml_strategy.predict_signal import predict

    df = ...  # DataFrame con las últimas ~50 velas (columnas OHLCV)
    signal = predict(df)         # → MLPredictionSignal
    if signal.signal == 1:
        print("COMPRAR", signal.confidence)

También se puede integrar como estrategia del bot:
    from ml_strategy.predict_signal import MLStrategy  # compatible con BaseStrategy
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ── Añadir la raíz del proyecto al path ──
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ml_strategy.features import compute_features, FEATURE_COLUMNS  # type: ignore

# ── Rutas del modelo ──
_DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "models" / "bitcoin_xgb_model.json"

# ── Cache del modelo en memoria (se carga una sola vez) ──
_model_cache: dict[str, object] = {}


@dataclass
class MLPredictionSignal:
    """Señal de predicción del modelo ML."""
    signal: int            # 0 = no hacer nada, 1 = comprar
    confidence: float      # probabilidad de la clase predicha [0, 1]
    proba_0: float         # P(clase 0)
    proba_1: float         # P(clase 1)
    features_ok: bool      # True si se pudieron calcular todas las features

    def __repr__(self) -> str:
        action = "🟢 COMPRAR" if self.signal == 1 else "🔴 ESPERAR"
        return (f"MLPredictionSignal({action}, conf={self.confidence:.3f}, "
                f"P(0)={self.proba_0:.3f}, P(1)={self.proba_1:.3f})")


def _load_model(model_path: str | Path | None = None) -> object:
    """Carga el modelo XGBoost desde disco (con caché en memoria)."""
    path = str(model_path or _DEFAULT_MODEL_PATH)

    if path in _model_cache:
        return _model_cache[path]

    from xgboost import XGBClassifier  # type: ignore

    if not Path(path).exists():
        raise FileNotFoundError(
            f"No se encontró el modelo en {path}. "
            f"Ejecuta primero: python -m ml_strategy.train_model"
        )

    model = XGBClassifier()
    model.load_model(path)
    _model_cache[path] = model
    return model


def predict(
    df: pd.DataFrame,
    model_path: str | Path | None = None,
    threshold: float = 0.5,
) -> MLPredictionSignal:
    """
    Predice señal de compra para la vela siguiente.

    Parámetros
    ----------
    df : pd.DataFrame
        Velas recientes con columnas open, high, low, close, volume.
        Debe tener al menos 52 filas (50 para SMA(50) + margen).
        El orden debe ser cronológico (más antiguo primero).
    model_path : str | Path | None
        Ruta al modelo .json. Si es None, usa el default.
    threshold : float
        Umbral de probabilidad para decidir clase 1 (default 0.5).

    Retorna
    -------
    MLPredictionSignal
        Señal con la predicción y confianza.
    """
    model = _load_model(model_path)

    # ── Validación mínima de datos ──
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en el DataFrame: {missing}")

    if len(df) < 52:
        raise ValueError(
            f"Se necesitan al menos 52 velas para calcular SMA(50). "
            f"Recibidas: {len(df)}"
        )

    # ── Calcular features ──
    feats_df = compute_features(df)

    # ── Tomar la ÚLTIMA fila (feature más reciente) ──
    last_row = feats_df.iloc[-1]

    if last_row.isna().any():
        # Alguna feature no se pudo calcular (pocos datos al inicio)
        nan_features = last_row[last_row.isna()].index.tolist()
        return MLPredictionSignal(
            signal=0,
            confidence=0.0,
            proba_0=1.0,
            proba_1=0.0,
            features_ok=False,
        )

    # ── Preparar input para el modelo ──
    X = pd.DataFrame([last_row[FEATURE_COLUMNS].values], columns=FEATURE_COLUMNS)

    # ── Predecir ──
    proba = model.predict_proba(X)[0]  # [P(0), P(1)]

    if len(proba) < 2:
        # Solo una clase en el modelo (edge case)
        return MLPredictionSignal(
            signal=0, confidence=1.0,
            proba_0=1.0, proba_1=0.0,
            features_ok=True,
        )

    proba_0, proba_1 = float(proba[0]), float(proba[1])
    signal = 1 if proba_1 >= threshold else 0
    confidence = proba_1 if signal == 1 else proba_0

    return MLPredictionSignal(
        signal=signal,
        confidence=float(confidence),
        proba_0=proba_0,
        proba_1=proba_1,
        features_ok=True,
    )


def predict_from_ohlcv(
    ohlcv: list[list[float]],
    model_path: str | Path | None = None,
    threshold: float = 0.5,
) -> MLPredictionSignal:
    """
    Predice a partir del formato OHLCV nativo de CCXT.

    Parámetros
    ----------
    ohlcv : list[list[float]]
        Lista de velas en formato CCXT:
        ``[[timestamp_ms, open, high, low, close, volume], …]``.
        Debe tener al menos 52 velas (para SMA(50)).
    model_path : str | Path | None
        Ruta al modelo .json.
    threshold : float
        Umbral de decisión (default 0.5).

    Retorna
    -------
    MLPredictionSignal

    Ejemplo de uso en main.py
    -------------------------
        ohlcv = await exchange.fetch_ohlcv(symbol, timeframe="1h", limit=100)
        signal = predict_from_ohlcv(ohlcv)
        if signal.signal == 1:
            await execution.place_buy_order(...)
    """
    if len(ohlcv) < 52:
        return MLPredictionSignal(
            signal=0, confidence=0.0,
            proba_0=1.0, proba_1=0.0,
            features_ok=False,
        )

    df = pd.DataFrame(
        ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)

    return predict(df, model_path=model_path, threshold=threshold)


async def fetch_and_predict(
    symbol: str = "BTC/EUR",
    timeframe: str = "1h",
    limit: int = 100,
    model_path: str | Path | None = None,
    threshold: float = 0.5,
) -> MLPredictionSignal:
    """
    Se conecta a Kraken (CCXT), obtiene las últimas velas y predice.

    No necesita API keys — OHLCV es un endpoint público.

    Parámetros
    ----------
    symbol : str
        Par de trading (default "BTC/EUR").
    timeframe : str
        Temporalidad (default "1h").
    limit : int
        Número de velas a obtener (default 100, mínimo 52).
    model_path : str | Path | None
        Ruta al modelo .json.
    threshold : float
        Umbral de decisión (default 0.5).

    Retorna
    -------
    MLPredictionSignal

    Ejemplo
    -------
        import asyncio
        from ml_strategy import fetch_and_predict

        signal = asyncio.run(fetch_and_predict())
        print(signal)
    """
    try:
        import ccxt  # type: ignore
    except ImportError:
        raise ImportError("ccxt no está instalado. Ejecuta: pip install ccxt")

    exchange = ccxt.kraken({
        "enableRateLimit": True,
    })

    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    finally:
        # Limpiar sesión (Kraken no requiere mantener conexión para REST)
        pass

    if not ohlcv or len(ohlcv) < 52:
        return MLPredictionSignal(
            signal=0, confidence=0.0,
            proba_0=1.0, proba_1=0.0,
            features_ok=False,
        )

    return predict_from_ohlcv(ohlcv, model_path=model_path, threshold=threshold)


# ── Integración opcional con el sistema de estrategias del bot ──
# Descomenta esto si quieres usar el modelo como una estrategia más en main.py:

# from src.strategies.base import BaseStrategy, Signal as BotSignal
#
# class MLStrategy(BaseStrategy):
#     """Estrategia que usa el modelo XGBoost entrenado."""
#     name = "ml_xgboost"
#
#     def __init__(self, model_path: str | None = None, threshold: float = 0.5):
#         self.model_path = model_path
#         self.threshold = threshold
#
#     async def compute_signal(self, ohlcv: list[list[float]]) -> BotSignal:
#         pred = predict_from_ohlcv(
#             ohlcv, model_path=self.model_path, threshold=self.threshold
#         )
#         if pred.signal == 1 and pred.features_ok:
#             return BotSignal(
#                 action="buy",
#                 confidence=pred.confidence,
#                 metadata={"proba_1": pred.proba_1, "model": "xgboost"},
#             )
#         return BotSignal(metadata={"reason": "no_signal", "proba_1": pred.proba_1})


# ── CLI rápida para test ──
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Predicción rápida con el modelo ML")
    parser.add_argument("--data", type=Path, default=_PROJECT_ROOT / "BTCUSD_1h_Binance.csv",
                        help="CSV de velas (usa las últimas filas para predecir)")
    parser.add_argument("--model", type=Path, default=None,
                        help="Ruta al modelo .json")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Umbral de decisión (default: 0.5)")
    parser.add_argument("--tail", type=int, default=60,
                        help="Número de velas recientes a usar (default: 60)")
    args = parser.parse_args()

    if not args.data.exists():
        print(f"ERROR: No se encuentra {args.data}")
        sys.exit(1)

    df = pd.read_csv(args.data)
    # Normalizar columnas al formato esperado (misma lógica que train_model.py)
    col_map: dict[str, str] = {}
    time_col: str | None = None
    for col in df.columns:
        key = col.strip().lower().replace(" ", "")
        if key in ("timestamp", "opentime"):
            time_col = col
        elif key == "open":
            col_map[col] = "open"
        elif key == "high":
            col_map[col] = "high"
        elif key == "low":
            col_map[col] = "low"
        elif key == "close":
            col_map[col] = "close"
        elif key == "volume":
            col_map[col] = "volume"
    df.rename(columns=col_map, inplace=True)
    if time_col:
        df[time_col] = pd.to_datetime(df[time_col])
        df.set_index(time_col, inplace=True)
    df.sort_index(inplace=True)
    df_tail = df.tail(args.tail)

    print(f"Datos: {len(df_tail)} velas — {df_tail.index[0]} → {df_tail.index[-1]}")
    signal = predict(df_tail, model_path=args.model, threshold=args.threshold)
    print(signal)
