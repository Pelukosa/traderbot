"""
Ingeniería de features — compartida entre entrenamiento y predicción.

TODAS las features se calculan aquí para garantizar consistencia.
No se usan librerías externas pesadas: solo numpy + pandas.

Features: 31 indicadores técnicos calculados a partir de OHLCV.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ── Columnas de features que el modelo espera (orden canónico) ──
FEATURE_COLUMNS: list[str] = [
    # Momentum / Osciladores (5) — ya son % o normalizados
    "rsi_14",
    "stoch_k",
    "stoch_d",
    "roc_6",
    "roc_12",
    # MACD relativo (3) — dividido por close para ser agnóstico a precio
    "macd_line_rel",
    "macd_signal_rel",
    "macd_hist_rel",
    # Medias móviles / Tendencia (3) — ya son relativos
    "price_vs_sma20",
    "price_vs_sma50",
    "sma20_vs_sma50",
    # Bollinger Bands (3) — ya son relativos
    "bb_position",
    "bb_width",
    "bb_squeeze",
    # Volatilidad (2) — relativa
    "volatility_14",
    "atr_ratio",
    # Volumen (3) — relativos. ¡CUIDADO! Solo funcionan si train y test son mismo exchange.
    # Si entrenas en Binance y predices en Kraken, COMENTA estas 3 líneas:
    "vol_rel",
    "vol_short",
    "vol_trend",
    # Velas / Estructura de precio (5) — relativos
    "returns_1",
    "returns_6",
    "returns_12",
    "high_low_ratio",
    "candle_body_ratio",
    # Soportes / Resistencias (3) — relativos
    "dist_high_20",
    "dist_low_20",
    "consecutive_dir",
    # EMA cross (1) — relativo
    "ema_cross_signal",
]


def _ema(series: np.ndarray, period: int) -> np.ndarray:
    """EMA implementada con numpy puro (sin pandas-ta).

    Soporta series con NaN al inicio (ej: macd_line que hereda NaN de ema12/ema26).
    Busca el primer bloque de `period` valores consecutivos no-NaN para arrancar.
    """
    result = np.empty_like(series)
    result[:] = np.nan

    # Encontrar el primer bloque de `period` valores consecutivos válidos
    valid = ~np.isnan(series)
    run_len = 0
    start_idx = -1
    for i in range(len(series)):
        if valid[i]:
            run_len += 1
            if run_len >= period:
                start_idx = i  # i es el índice del último valor del primer bloque válido
                break
        else:
            run_len = 0

    if start_idx < 0:
        return result  # no hay suficientes datos consecutivos

    # seed: media del primer bloque válido
    result[start_idx] = np.mean(series[start_idx - period + 1:start_idx + 1])
    multiplier = 2.0 / (period + 1)
    for i in range(start_idx + 1, len(series)):
        result[i] = (series[i] - result[i - 1]) * multiplier + result[i - 1]
    return result


def _rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
    """RSI de Wilder (1978)."""
    deltas = np.diff(closes, prepend=closes[0])
    gain = np.where(deltas > 0, deltas, 0.0)
    loss = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.empty_like(closes); avg_gain[:] = np.nan
    avg_loss = np.empty_like(closes); avg_loss[:] = np.nan

    avg_gain[period] = np.mean(gain[1:period + 1])
    avg_loss[period] = np.mean(loss[1:period + 1])

    for i in range(period + 1, len(closes)):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period

    rs = avg_gain / np.where(avg_loss == 0, 1e-10, avg_loss)
    return 100.0 - 100.0 / (1.0 + rs)


def _rolling_std(series: np.ndarray, period: int) -> np.ndarray:
    """Desviación estándar móvil (soporta NaN al inicio)."""
    result = np.empty_like(series); result[:] = np.nan
    valid = ~np.isnan(series)
    for i in range(period - 1, len(series)):
        if valid[i - period + 1:i + 1].all():
            result[i] = np.std(series[i - period + 1:i + 1], ddof=1)
    return result


def _rolling_mean(series: np.ndarray, period: int) -> np.ndarray:
    """Media móvil simple (SMA) — soporta NaN al inicio."""
    result = np.empty_like(series); result[:] = np.nan
    valid = ~np.isnan(series)
    for i in range(period - 1, len(series)):
        if valid[i - period + 1:i + 1].all():
            result[i] = np.mean(series[i - period + 1:i + 1])
    return result


def _rolling_max(series: np.ndarray, period: int) -> np.ndarray:
    """Máximo móvil (soporta NaN al inicio)."""
    result = np.empty_like(series); result[:] = np.nan
    valid = ~np.isnan(series)
    for i in range(period - 1, len(series)):
        if valid[i - period + 1:i + 1].all():
            result[i] = np.max(series[i - period + 1:i + 1])
    return result


def _rolling_min(series: np.ndarray, period: int) -> np.ndarray:
    """Mínimo móvil (soporta NaN al inicio)."""
    result = np.empty_like(series); result[:] = np.nan
    valid = ~np.isnan(series)
    for i in range(period - 1, len(series)):
        if valid[i - period + 1:i + 1].all():
            result[i] = np.min(series[i - period + 1:i + 1])
    return result


def _true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """True Range para ATR."""
    tr = np.empty_like(high); tr[:] = np.nan
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.abs(high - prev_close))
    tr = np.maximum(tr, np.abs(low - prev_close))
    return tr


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula 31 features técnicas a partir de un DataFrame OHLCV.

    Parámetros
    ----------
    df : pd.DataFrame
        Debe contener columnas: open, high, low, close, volume.
        El índice debe ser datetime (se preserva).

    Retorna
    -------
    pd.DataFrame
        DataFrame con solo las columnas de FEATURE_COLUMNS, mismo índice.
    """
    opens = df["open"].values.astype(np.float64)
    highs = df["high"].values.astype(np.float64)
    lows = df["low"].values.astype(np.float64)
    closes = df["close"].values.astype(np.float64)
    volumes = df["volume"].values.astype(np.float64)
    n = len(closes)

    result = pd.DataFrame(index=df.index)

    # ═══════════════════════════════════════════════════════════════
    # Momentum / Osciladores
    # ═══════════════════════════════════════════════════════════════

    # RSI(14)
    result["rsi_14"] = _rsi(closes, 14)

    # Stochastic %K, %D (14, 3, 3)
    low14 = _rolling_min(lows, 14)
    high14 = _rolling_max(highs, 14)
    stoch_k_raw = (closes - low14) / np.where((high14 - low14) == 0, 1e-10, high14 - low14) * 100.0
    result["stoch_k"] = stoch_k_raw
    result["stoch_d"] = _rolling_mean(stoch_k_raw, 3)

    # ROC (Rate of Change) 6 y 12 periodos
    roc_6 = np.empty(n); roc_6[:] = np.nan
    roc_12 = np.empty(n); roc_12[:] = np.nan
    for i in range(6, n):
        roc_6[i] = (closes[i] - closes[i - 6]) / max(closes[i - 6], 1e-10) * 100.0
    for i in range(12, n):
        roc_12[i] = (closes[i] - closes[i - 12]) / max(closes[i - 12], 1e-10) * 100.0
    result["roc_6"] = roc_6
    result["roc_12"] = roc_12

    # ═══════════════════════════════════════════════════════════════
    # MACD (12, 26, 9) — relativo al precio para ser agnóstico a la moneda
    # ═══════════════════════════════════════════════════════════════
    ema12 = _ema(closes, 12)
    ema26 = _ema(closes, 26)
    macd_line = ema12 - ema26
    macd_signal_line = _ema(macd_line, 9)
    macd_hist = macd_line - macd_signal_line
    # Dividir por close para normalizar (BTC a $50K vs €52K dan MACD muy distinto)
    result["macd_line_rel"] = macd_line / np.where(closes == 0, 1e-10, closes)
    result["macd_signal_rel"] = macd_signal_line / np.where(closes == 0, 1e-10, closes)
    result["macd_hist_rel"] = macd_hist / np.where(closes == 0, 1e-10, closes)

    # ═══════════════════════════════════════════════════════════════
    # SMA / Tendencia (solo relativos, no absolutos)
    # ═══════════════════════════════════════════════════════════════
    sma20 = _rolling_mean(closes, 20)
    sma50 = _rolling_mean(closes, 50)
    result["price_vs_sma20"] = (closes - sma20) / np.where(sma20 == 0, 1e-10, sma20)
    result["price_vs_sma50"] = (closes - sma50) / np.where(sma50 == 0, 1e-10, sma50)
    result["sma20_vs_sma50"] = (sma20 - sma50) / np.where(sma50 == 0, 1e-10, sma50)

    # ═══════════════════════════════════════════════════════════════
    # Bollinger Bands (20, 2)
    # ═══════════════════════════════════════════════════════════════
    bb_middle = sma20
    bb_std = np.empty(n); bb_std[:] = np.nan
    for i in range(19, n):
        bb_std[i] = np.std(closes[i - 19:i + 1], ddof=1)
    bb_upper = bb_middle + 2.0 * bb_std
    bb_lower = bb_middle - 2.0 * bb_std
    # %B: posición del precio dentro de las bandas [0, 1], puede salirse
    result["bb_position"] = (closes - bb_lower) / np.where((bb_upper - bb_lower) == 0, 1e-10, bb_upper - bb_lower)
    # Bandwidth normalizado
    result["bb_width"] = (bb_upper - bb_lower) / np.where(bb_middle == 0, 1e-10, bb_middle)
    # Squeeze: ¿las bandas se están contrayendo? (width actual vs width hace 20 velas)
    bb_width_20_ago = np.roll(result["bb_width"].values, 20)
    bb_width_20_ago[:20] = np.nan
    result["bb_squeeze"] = result["bb_width"].values - bb_width_20_ago  # negativo = contrayéndose

    # ═══════════════════════════════════════════════════════════════
    # Volatilidad (solo relativa)
    # ═══════════════════════════════════════════════════════════════
    returns = np.diff(closes, prepend=closes[0]) / np.where(closes == 0, 1e-10, closes)
    result["volatility_14"] = _rolling_std(returns, 14)

    # ATR ratio (ATR / close) — agnóstico al precio
    tr = _true_range(highs, lows, closes)
    atr = _ema(tr, 14)
    result["atr_ratio"] = atr / np.where(closes == 0, 1e-10, closes)

    # ═══════════════════════════════════════════════════════════════
    # Volumen
    # ═══════════════════════════════════════════════════════════════
    vol_sma20 = _rolling_mean(volumes, 20)
    vol_sma5 = _rolling_mean(volumes, 5)
    result["vol_rel"] = volumes / np.where(vol_sma20 == 0, 1e-10, vol_sma20)
    result["vol_short"] = volumes / np.where(vol_sma5 == 0, 1e-10, vol_sma5)
    # Tendencia del volumen: ¿volumen reciente > volumen hace 10 velas?
    vol_10_ago = np.roll(volumes, 10); vol_10_ago[:10] = np.nan
    result["vol_trend"] = (volumes - vol_10_ago) / np.where(vol_10_ago == 0, 1e-10, vol_10_ago)

    # ═══════════════════════════════════════════════════════════════
    # Estructura de velas / Price action
    # ═══════════════════════════════════════════════════════════════
    result["returns_1"] = returns
    # Returns a 6 y 12 velas
    ret_6 = np.empty(n); ret_6[:] = np.nan
    ret_12 = np.empty(n); ret_12[:] = np.nan
    for i in range(6, n):
        ret_6[i] = (closes[i] - closes[i - 6]) / max(closes[i - 6], 1e-10)
    for i in range(12, n):
        ret_12[i] = (closes[i] - closes[i - 12]) / max(closes[i - 12], 1e-10)
    result["returns_6"] = ret_6
    result["returns_12"] = ret_12
    # High/Low ratio
    result["high_low_ratio"] = (highs - lows) / np.where(closes == 0, 1e-10, closes)
    # Candle body ratio: (close - open) / (high - low)
    body = np.abs(closes - opens)
    candle_range = highs - lows
    result["candle_body_ratio"] = body / np.where(candle_range == 0, 1e-10, candle_range)

    # ═══════════════════════════════════════════════════════════════
    # Soportes / Resistencias
    # ═══════════════════════════════════════════════════════════════
    high20 = _rolling_max(highs, 20)
    low20 = _rolling_min(lows, 20)
    result["dist_high_20"] = (high20 - closes) / np.where(closes == 0, 1e-10, closes)
    result["dist_low_20"] = (closes - low20) / np.where(closes == 0, 1e-10, closes)
    # Dirección consecutiva (sesgo reciente)
    direction = np.sign(np.diff(closes, prepend=closes[0]))
    consec = np.empty(n); consec[:] = np.nan
    consec[0] = direction[0]
    for i in range(1, n):
        consec[i] = consec[i - 1] + direction[i] if direction[i] == direction[i - 1] else direction[i]
    result["consecutive_dir"] = consec

    # ═══════════════════════════════════════════════════════════════
    # EMA cross
    # ═══════════════════════════════════════════════════════════════
    # Señal de cruce EMA12/26: positivo = dorado (EMA rápida arriba), negativo = muerte
    result["ema_cross_signal"] = (ema12 - ema26) / np.where(closes == 0, 1e-10, closes)

    return result
