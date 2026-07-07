"""
ml_strategy — Módulo de Machine Learning para trading con XGBoost.

Estructura:
    features.py       → Ingeniería de features compartida (train + predict)
    train_model.py    → Script de entrenamiento offline
    predict_signal.py → Función ligera de predicción para producción

Flujo típico:
    1. python -m ml_strategy.train_model   → entrena y guarda bitcoin_xgb_model.json
    2. En producción:
       from ml_strategy.predict_signal import predict
       signal = predict(df_ultimas_velas)   # 1 = comprar, 0 = esperar
"""

from ml_strategy.predict_signal import predict, predict_from_ohlcv, fetch_and_predict, MLPredictionSignal
from ml_strategy.features import compute_features, FEATURE_COLUMNS

__all__ = [
    "predict",
    "predict_from_ohlcv",
    "fetch_and_predict",
    "MLPredictionSignal",
    "compute_features",
    "FEATURE_COLUMNS",
]
