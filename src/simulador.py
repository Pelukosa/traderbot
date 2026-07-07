"""
SIMON — Simulador de estrategias de trading.

Carga las velas 1h, prueba todas las estrategias registradas
y devuelve el TOP 3 con formato estandarizado.

Uso:
    uv run python scripts/simular.py
"""
from __future__ import annotations

import csv
import json
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

# ── Config ──
INITIAL_EUR = 1000.0       # capital simulado
INVEST_PCT = 95.0          # % del capital por operacion
TRAIL_RETAIN = 0.6         # % de ganancia que retiene el trailing
MAX_POSITION_CANDLES = 12  # maximo velas en posicion (12h)
MIN_TRADE = 5.0            # minimo para operar
MIN_OPS_FOR_RANKING = 1500  # mínimo de operaciones para aparecer en el TOP
FEE_RATE = 0.0             # comision simulada (% por operacion, 0.20 = OKX)

DATA_FILE = Path(__file__).parent.parent / "BTCUSD_1h_Binance.csv"
SIM_DIR = Path(__file__).parent.parent / "simulations"


# ── Resultado estandarizado ──

@dataclass
class ResultadoSim:
    """Siempre el mismo formato para todas las estrategias."""
    estrategia: str = ""
    descripcion: str = ""

    # Operaciones
    total_ops: int = 0
    ganadoras: int = 0
    perdedoras: int = 0
    winrate: float = 0.0           # %

    # Rentabilidad en %
    ganancia_media_por_op: float = 0.0   # % medio de todas las operaciones
    ganancia_media_ganadoras: float = 0.0
    perdida_media_perdedoras: float = 0.0
    mejor_operacion: float = 0.0   # %
    peor_operacion: float = 0.0    # %

    # Rentabilidad en €
    pnl_bruto: float = 0.0
    pnl_neto: float = 0.0
    comisiones: float = 0.0
    pnl_por_operacion: float = 0.0  # neto
    pnl_diario: float = 0.0
    pnl_mensual: float = 0.0

    # ROI
    roi_total: float = 0.0
    roi_diario: float = 0.0
    roi_mensual: float = 0.0

    # Tiempo
    tiempo_medio_h: float = 0.0
    tiempo_maximo_h: float = 0.0
    dias_simulados: float = 0.0
    ops_por_mes: float = 0.0

    # Riesgo
    max_drawdown: float = 0.0

    # Score (ranking)
    score: float = 0.0

    def a_dict(self) -> dict:
        return {k: round(v, 2) if isinstance(v, float) else v for k, v in asdict(self).items()}


# ── Indicadores tecnicos ──

def _ema(data: np.ndarray, period: int) -> np.ndarray:
    r = np.empty_like(data); r[:] = np.nan
    fs = np.where(~np.isnan(data))[0]
    if len(fs) == 0: return r
    s = fs[0] + period - 1
    if s >= len(data): return r
    r[s] = np.mean(data[s-period+1:s+1])
    m = 2.0/(period+1)
    for i in range(s+1, len(data)): r[i] = (data[i] - r[i-1]) * m + r[i-1]
    return r

def _sma(data: np.ndarray, period: int) -> np.ndarray:
    r = np.empty_like(data); r[:] = np.nan
    for i in range(period-1, len(data)): r[i] = np.mean(data[i-period+1:i+1])
    return r

def _std(data: np.ndarray, period: int) -> np.ndarray:
    r = np.empty_like(data); r[:] = np.nan
    for i in range(period-1, len(data)): r[i] = np.std(data[i-period+1:i+1])
    return r

def _rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
    d = np.diff(closes); r = np.full_like(closes, np.nan)
    if len(d) < period: return r
    seed = d[:period]
    up = seed[seed>=0].sum()/period; down = -seed[seed<0].sum()/period
    rs = up/down if down else 0; r[period] = 100-100/(1+rs)
    for i in range(period+1, len(closes)):
        dv = d[i-1]
        upv = dv if dv>0 else 0; downv = -dv if dv<0 else 0
        up = (up*(period-1)+upv)/period; down = (down*(period-1)+downv)/period
        rs = up/down if down else 0; r[i] = 100-100/(1+rs)
    return r

def _atr(ohlcv: list[list[float]], period: int = 14) -> np.ndarray:
    r = np.full(len(ohlcv), np.nan)
    for i in range(1, len(ohlcv)):
        h,l,pc = ohlcv[i][2], ohlcv[i][3], ohlcv[i-1][4]
        r[i] = max(h-l, abs(h-pc), abs(l-pc))
    return _ema(r, period)


# ── Carga de datos ──

def _parse_ts(ts_str: str) -> int:
    """Parse timestamp from string (datetime or unix ms)."""
    ts_str = ts_str.strip()
    if ts_str.isdigit() or (ts_str.startswith('-') and ts_str[1:].isdigit()):
        return int(ts_str)
    from datetime import datetime, timezone
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S%z"):
        try:
            return int(datetime.strptime(ts_str, fmt).replace(tzinfo=timezone.utc).timestamp() * 1000)
        except ValueError:
            continue
    return int(float(ts_str))


def cargar_velas(path: str | Path | None = None) -> list[list[float]]:
    p = Path(path) if path else DATA_FILE
    if not p.exists():
        print(f"ERROR: No encuentro {p}")
        print("Ejecuta primero: uv run python -c \"from src.simulador import descargar_velas; descargar_velas()\"")
        sys.exit(1)
    rows = []
    with open(p) as f:
        for r in csv.DictReader(f):
            # Binance CSV: Open time, Close time, Open, High, Low, Close, Volume
            # Kraken CSV: timestamp, open, high, low, close, volume
            ts_col = "Open time" if "Open time" in r else "timestamp"
            o_col = "Open" if "Open" in r else "open"
            h_col = "High" if "High" in r else "high"
            l_col = "Low" if "Low" in r else "low"
            c_col = "Close" if "Close" in r else "close"
            v_col = "Volume" if "Volume" in r else "volume"
            rows.append([_parse_ts(r[ts_col]), float(r[o_col]), float(r[h_col]),
                         float(r[l_col]), float(r[c_col]), float(r[v_col])])
    return rows


def descargar_velas():
    """Descarga 720 velas 1h de Kraken y las guarda."""
    import asyncio
    from src.exchange import ExchangeManager

    async def _dl():
        ex = ExchangeManager()
        await ex.load_markets()
        ohlcv = await ex.fetch_ohlcv("BTC/EUR", "1h", limit=720)
        with open(DATA_FILE, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timestamp", "open", "high", "low", "close", "volume"])
            for c in ohlcv:
                w.writerow([int(c[0]), c[1], c[2], c[3], c[4], c[5]])
        await ex.close()
        print(f"Descargadas {len(ohlcv)} velas 1h -> {DATA_FILE}")
    asyncio.run(_dl())


# ── Precalculos comunes ──

def precalcular(ohlcv: list[list[float]]) -> dict:
    closes = np.array([c[4] for c in ohlcv], dtype=np.float64)
    vols = np.array([c[5] for c in ohlcv], dtype=np.float64)
    highs = np.array([c[2] for c in ohlcv], dtype=np.float64)
    lows = np.array([c[3] for c in ohlcv], dtype=np.float64)

    ema20 = _ema(closes, 20)
    rsi_arr = _rsi(closes)
    bb_mid = _sma(closes, 20)
    bb_std = _std(closes, 20)
    bb_up = bb_mid + bb_std * 2.0
    bb_down = bb_mid - bb_std * 2.0
    atr_arr = _atr(ohlcv)
    vol_sma = _sma(vols, 20)
    ema_f = _ema(closes, 12); ema_s = _ema(closes, 26)
    macd_line = ema_f - ema_s
    macd_sig = _ema(macd_line, 9)
    hist_arr = macd_line - macd_sig

    return {
        "closes": closes, "vols": vols, "highs": highs, "lows": lows,
        "ema20": ema20, "rsi_arr": rsi_arr, "atr_arr": atr_arr,
        "bb_up": bb_up, "bb_down": bb_down, "bb_mid": bb_mid,
        "vol_sma": vol_sma, "macd_line": macd_line, "macd_sig": macd_sig,
        "hist_arr": hist_arr,
    }


# ── Funcion de senal para MACD puro ──

def senal_macd(ind: dict, i: int, cfg: dict) -> str | None:
    """MACD valley -> buy"""
    h = _hist_buf(ind["hist_arr"], i)
    if len(h) < 4: return None
    min_h = cfg.get("min_histogram_abs", 60)
    if h[-4] >= h[-3] > h[-2] < h[-1] and h[-2] < 0 and abs(h[-2]) >= min_h:
        return "buy"
    return None


def senal_macd_rsi(ind: dict, i: int, cfg: dict) -> str | None:
    """MACD valley + RSI < umbral -> buy"""
    h = _hist_buf(ind["hist_arr"], i); rsi = ind["rsi_arr"]
    if len(h) < 4: return None
    min_h = cfg.get("min_histogram_abs", 60)
    rsi_max = cfg.get("rsi_max", 40)
    rsi_v = float(rsi[i]) if i < len(rsi) and not np.isnan(rsi[i]) else 999
    if h[-4] >= h[-3] > h[-2] < h[-1] and h[-2] < 0 and abs(h[-2]) >= min_h and rsi_v < rsi_max:
        return "buy"
    return None


def senal_bb_rsi(ind: dict, i: int, cfg: dict) -> str | None:
    """Precio toca banda inferior de Bollinger + RSI bajo -> buy"""
    bb_d = ind["bb_down"]; rsi = ind["rsi_arr"]
    if i < 20 or i >= len(bb_d) or np.isnan(bb_d[i]): return None
    rsi_v = float(rsi[i]) if i < len(rsi) and not np.isnan(rsi[i]) else 999
    rsi_max = cfg.get("rsi_max", 35)
    cl = float(ind["closes"][i])
    if cl <= bb_d[i] and rsi_v < rsi_max:
        return "buy"
    return None


def senal_macd_vol(ind: dict, i: int, cfg: dict) -> str | None:
    """MACD valley + volumen anormalmente alto -> buy"""
    h = _hist_buf(ind["hist_arr"], i); vol = ind["vols"]; vol_sma = ind["vol_sma"]
    if len(h) < 4: return None
    min_h = cfg.get("min_histogram_abs", 60)
    vol_mult = cfg.get("vol_mult", 1.5)
    if h[-4] >= h[-3] > h[-2] < h[-1] and h[-2] < 0 and abs(h[-2]) >= min_h:
        if i >= 24 and vol_sma[i] > 0 and vol[i] > vol_sma[i] * vol_mult:
            return "buy"
    return None


def senal_momentum(ind: dict, i: int, cfg: dict) -> str | None:
    """Precio cruza EMA20 al alza con volumen alto -> buy"""
    if i < 1: return None
    ema20 = ind["ema20"]; vol = ind["vols"]; vol_sma = ind["vol_sma"]
    vol_mult = cfg.get("vol_mult", 1.5)
    prev = float(ind["closes"][i-1]); cur = float(ind["closes"][i])
    if prev <= ema20[i] and cur > ema20[i]:
        if vol_sma[i] > 0 and vol[i] > vol_sma[i] * vol_mult:
            return "buy"
    return None


def senal_breakout(ind: dict, i: int, cfg: dict) -> str | None:
    """Precio supera maximo de N velas -> buy"""
    lookback = cfg.get("lookback", 20)
    vol_th = cfg.get("vol_threshold", 1.5)
    if i < lookback: return None
    highs = ind["highs"]; vol = ind["vols"]
    max_prev = np.max(highs[i-lookback:i])
    avg_vol = np.mean(vol[i-lookback:i]) if i >= lookback else 0
    if float(highs[i]) > max_prev and avg_vol > 0 and vol[i] > avg_vol * vol_th:
        return "buy"
    return None


def senal_triple(ind: dict, i: int, cfg: dict) -> str | None:
    """TRIPLE CONFIRMACION: MACD valley + RSI < umbral + precio cerca de BB inferior.

    Busca la confluencia de 3 indicadores para minimizar falsas senales.
    """
    h = _hist_buf(ind["hist_arr"], i)
    rsi = ind["rsi_arr"]
    bb_d = ind["bb_down"]
    closes = ind["closes"]

    if len(h) < 4: return None
    min_h = cfg.get("min_histogram_abs", 50)
    rsi_max = cfg.get("rsi_max", 30)
    bb_dist_pct = cfg.get("bb_max_distance_pct", 2.0)  # % del precio maximo desde BB inferior

    rsi_v = float(rsi[i]) if i < len(rsi) and not np.isnan(rsi[i]) else 999
    cl = float(closes[i])

    # 1. MACD valley
    if not (h[-4] >= h[-3] > h[-2] < h[-1] and h[-2] < 0 and abs(h[-2]) >= min_h):
        return None

    # 2. RSI sobrevendido (mas restrictivo)
    if not (rsi_v < rsi_max):
        return None

    # 3. Precio cerca de banda inferior de Bollinger (a menos de X%)
    if i >= 20 and not np.isnan(bb_d[i]):
        if bb_d[i] > 0:
            dist_to_bb = (cl - bb_d[i]) / bb_d[i] * 100
            if dist_to_bb > bb_dist_pct:
                return None  # muy lejos de la banda inferior
    else:
        return None

    return "buy"


def _hist_buf(hist_arr: np.ndarray, i: int) -> list[float]:
    h = []
    for j in range(max(0, i-4), i+1):
        if not np.isnan(hist_arr[j]): h.append(float(hist_arr[j]))
        else: h = []
        if len(h) > 5: h = h[-5:]
    return h


# ── Nueva estrategia: MACD Histogram < -50 valley con doble entrada ──

def senal_macd_hist_50(ind: dict, i: int, cfg: dict) -> str | None:
    """MACD histogram crosses below -50 and then reverses → buy signal.

    Two-step confirmation:
    1. Histogram bar crosses from > -50 to < -50 (going down)
    2. Next completed bar is HIGHER (less negative) than the crossing bar

    Example: -2 → -6 → -17 → -40 → -52 (cross!) → -30 (next bar, -30 > -52) → BUY
    """
    hist_arr = ind["hist_arr"]
    if i < 2:
        return None
    h_prev = float(hist_arr[i-1]) if not np.isnan(hist_arr[i-1]) else None
    h_prev2 = float(hist_arr[i-2]) if not np.isnan(hist_arr[i-2]) else None
    if h_prev is None or h_prev2 is None:
        return None
    # Crossing: previous bar was > -50, current closed bar (i-1) is < -50, AND going down
    if h_prev2 > -50 and h_prev < -50 and h_prev < h_prev2:
        # Current bar (i) just started — check reversal at next bar
        pass  # We'll mark awaiting and check next iteration

    return None


def senal_macd_hist_50_stateful(state: dict, ind: dict, i: int, cfg: dict) -> str | None:
    """Stateful version that tracks the crossing and waits for reversal.

    Uses 'state' dict with keys: 'awaiting_cross_bar', 'cross_bar_value'
    """
    hist_arr = ind["hist_arr"]
    if i < 2:
        return None

    h_prev = float(hist_arr[i-1]) if not np.isnan(hist_arr[i-1]) else None
    h_prev2 = float(hist_arr[i-2]) if not np.isnan(hist_arr[i-2]) else None
    h_cur = float(hist_arr[i]) if not np.isnan(hist_arr[i]) else None

    if h_prev is None or h_prev2 is None:
        return None

    # Step 1: Detect crossing below -50 (going down)
    if h_prev2 > -50 and h_prev < -50 and h_prev < h_prev2:
        state["awaiting_cross_bar"] = True
        state["cross_bar_value"] = h_prev
        return None

    # Step 2: If awaiting, check if next bar closed higher (reversal confirmed)
    if state.get("awaiting_cross_bar") and h_cur is not None:
        cross_val = state.get("cross_bar_value", -999)
        if h_cur > cross_val:
            # Reversal confirmed!
            state["awaiting_cross_bar"] = False
            state["cross_bar_value"] = None
            return "buy"

    return None


# ── Simulación especial para MACD Hist -50 (doble entrada) ──

def simular_macd_hist_50(
    ohlcv: list[list[float]],
    pre: dict,
    cfg: dict,
    capital: float = INITIAL_EUR,
    fee_rate: float = FEE_RATE,
) -> ResultadoSim:
    """Simula la estrategia MACD Histogram -50 con doble entrada.

    - Compra 45% al confirmar valle < -50 con reversión
    - Si en 4 velas cae 2% → segunda compra de 45%
    - SL: -1% del precio ponderado (solo tras 2ª compra)
    - TP: +1% (del precio de entrada único o ponderado)
    """
    invest_pct = cfg.get("invest_percent", 45.0) / 100.0
    drop_2nd_pct = cfg.get("drop_2nd_entry_pct", 2.0) / 100.0
    tp_pct = cfg.get("tp_percent", 1.0) / 100.0
    sl_pct = cfg.get("sl_percent", 1.0) / 100.0
    max_candles = cfg.get("max_position_candles", 12)
    fee_por_op = fee_rate / 100.0

    closes = pre["closes"]
    hist_arr = pre["hist_arr"]

    trades: list[tuple] = []
    eur = capital
    btc = 0.0
    btc2 = 0.0  # second position
    in_pos = False
    has_second = False
    entry_p1 = 0.0
    entry_p2 = 0.0
    entry_i = 0
    total_fees = 0.0

    state: dict = {"awaiting_cross_bar": False, "cross_bar_value": None}

    for i in range(len(ohlcv)):
        close = float(closes[i])

        if in_pos:
            # Check weighted average price
            if has_second:
                avg_price = (entry_p1 * btc + entry_p2 * btc2) / (btc + btc2) if (btc + btc2) > 0 else entry_p1
            else:
                avg_price = entry_p1

            # Take profit
            if close >= avg_price * (1 + tp_pct):
                pnl = (close - avg_price) * (btc + btc2)
                fee = (btc + btc2) * close * fee_por_op
                total_fees += fee
                trades.append((avg_price, close, tp_pct * 100, i - entry_i, "TP"))
                eur += (btc + btc2) * close - fee
                btc = 0.0; btc2 = 0.0; in_pos = False; has_second = False
                continue

            # Stop loss (only active after second buy)
            if has_second and close <= avg_price * (1 - sl_pct):
                pnl = (close - avg_price) * (btc + btc2)
                fee = (btc + btc2) * close * fee_por_op
                total_fees += fee
                trades.append((avg_price, close, -sl_pct * 100, i - entry_i, "SL"))
                eur += (btc + btc2) * close - fee
                btc = 0.0; btc2 = 0.0; in_pos = False; has_second = False
                continue

            # Max time
            if i - entry_i >= max_candles:
                pnl = (close - avg_price) * (btc + btc2)
                fee = (btc + btc2) * close * fee_por_op
                total_fees += fee
                pnl_pct = (close - avg_price) / avg_price * 100
                trades.append((avg_price, close, pnl_pct, i - entry_i, "MAX_TIME"))
                eur += (btc + btc2) * close - fee
                btc = 0.0; btc2 = 0.0; in_pos = False; has_second = False
                continue

            # Second entry check: within 4 candles, price dropped 2% below first entry
            if not has_second and i - entry_i <= 4:
                if close <= entry_p1 * (1 - drop_2nd_pct):
                    invest2 = eur * invest_pct
                    if invest2 >= MIN_TRADE:
                        buy_fee = invest2 * fee_por_op
                        total_fees += buy_fee
                        btc2 = invest2 / close
                        eur -= invest2 + buy_fee
                        entry_p2 = close
                        has_second = True

        # Signal detection
        if not in_pos:
            accion = senal_macd_hist_50_stateful(state, pre, i, cfg)
            if accion == "buy":
                invest = eur * invest_pct
                if invest >= MIN_TRADE:
                    buy_fee = invest * fee_por_op
                    total_fees += buy_fee
                    btc = invest / close
                    eur -= invest + buy_fee
                    entry_p1 = close
                    entry_i = i
                    in_pos = True
                    has_second = False
                    btc2 = 0.0

    # Close any remaining position
    if in_pos:
        last_c = float(closes[-1])
        if has_second:
            avg_price = (entry_p1 * btc + entry_p2 * btc2) / (btc + btc2) if (btc + btc2) > 0 else entry_p1
        else:
            avg_price = entry_p1
        pnl_pct = (last_c - avg_price) / avg_price * 100
        fee = (btc + btc2) * last_c * fee_por_op
        total_fees += fee
        trades.append((avg_price, last_c, pnl_pct, len(ohlcv) - entry_i, "END"))
        eur += (btc + btc2) * last_c - fee

    # Build result
    r = ResultadoSim(
        estrategia="MACD Hist -50 (doble entrada)",
        descripcion=" | ".join(f"{k}={v}" for k, v in sorted(cfg.items())),
    )

    num = len(trades)
    total_dias = len(ohlcv) / 24.0
    pnl_neto = eur - capital

    if num == 0:
        r.dias_simulados = total_dias
        r.comisiones = total_fees
        return r

    gains_pct = [t[2] for t in trades]
    wins = [g for g in gains_pct if g > 0]
    losses = [g for g in gains_pct if g <= 0]
    durs = [t[3] for t in trades]
    pnl_bruto = pnl_neto + total_fees

    r.total_ops = num; r.ganadoras = len(wins); r.perdedoras = len(losses)
    r.winrate = len(wins) / num * 100
    r.ganancia_media_por_op = float(np.mean(gains_pct))
    r.ganancia_media_ganadoras = float(np.mean(wins)) if wins else 0.0
    r.perdida_media_perdedoras = float(np.mean(losses)) if losses else 0.0
    r.mejor_operacion = float(max(gains_pct))
    r.peor_operacion = float(min(gains_pct))
    r.pnl_bruto = pnl_bruto; r.pnl_neto = pnl_neto; r.comisiones = total_fees
    r.pnl_por_operacion = pnl_neto / num
    r.pnl_diario = pnl_neto / total_dias
    r.pnl_mensual = r.pnl_diario * 30
    r.roi_total = pnl_neto / capital * 100
    r.roi_diario = r.roi_total / total_dias
    r.roi_mensual = r.roi_diario * 30
    r.tiempo_medio_h = float(np.mean(durs))
    r.tiempo_maximo_h = float(max(durs))
    r.dias_simulados = total_dias
    r.ops_por_mes = num / total_dias * 30

    max_dd = 0.0; peak = capital; bal = capital
    for g in gains_pct:
        bal += bal * (g / 100) * invest_pct
        if bal > peak: peak = bal
        dd = (peak - bal) / peak * 100
        if dd > max_dd: max_dd = dd
    r.max_drawdown = max_dd
    r.score = r.roi_mensual * 0.4 + r.winrate * 0.15 + r.pnl_por_operacion * 0.3 - r.max_drawdown * 0.15

    return r


# ── Random Search: señal genérica multi-indicador ──

# Rangos de los ~30 parámetros más significativos para trading de Bitcoin
RANDOM_PARAM_SPACE: dict[str, tuple | list] = {
    # ── Risk Management ──
    "tp_percent":          (0.5, 5.0),        # Take profit %
    "sl_percent":          (0.5, 5.0),        # Stop loss %

    # ── RSI ──
    "use_rsi":             [True, False],
    "rsi_period":          (7, 21),
    "rsi_oversold":        (20, 45),

    # ── MACD ──
    "use_macd":            [True, False],
    "macd_fast":           (8, 20),
    "macd_slow":           (21, 40),
    "macd_signal_period":  (7, 12),
    "macd_hist_min":       (20, 80),

    # ── Bollinger Bands ──
    "use_bb":              [True, False],
    "bb_period":           (10, 30),
    "bb_std_mult":         (1.5, 3.0),
    "bb_position_pct":     (0.0, 5.0),        # % máximo por encima de la banda inferior

    # ── Volumen ──
    "use_volume":          [True, False],
    "vol_ma_period":       (10, 30),
    "vol_multiplier":      (1.0, 3.0),

    # ── Moving Average Crossover ──
    "use_ma_cross":        [True, False],
    "ma_short":            (5, 20),
    "ma_long":             (21, 50),

    # ── ATR ──
    "use_atr_sl":          [True, False],     # usar ATR para SL dinámico
    "atr_period":          (10, 20),
    "atr_mult_sl":         (1.0, 3.0),

    # ── Posición ──
    "max_position_candles": (6, 24),
    "min_candles_between":  (0, 5),
    "confirmation_candles": (0, 3),

    # ── Momentum / ROC ──
    "use_momentum":         [True, False],
    "momentum_period":      (5, 20),
    "momentum_threshold":   (0.5, 3.0),
}


def _sample_random_params() -> dict:
    """Devuelve un diccionario con una combinación aleatoria de parámetros."""
    cfg: dict = {}
    for key, space in RANDOM_PARAM_SPACE.items():
        if isinstance(space, list):
            cfg[key] = random.choice(space)
        elif isinstance(space, tuple):
            if isinstance(space[0], int) and isinstance(space[1], int):
                cfg[key] = random.randint(space[0], space[1])
            else:
                cfg[key] = round(random.uniform(space[0], space[1]), 2)
    # Ensure invest_percent is always 100% and at least one signal is active
    cfg["invest_percent"] = 100.0
    if not any(cfg.get(k) for k in ("use_rsi", "use_macd", "use_bb", "use_volume", "use_ma_cross", "use_momentum")):
        cfg["use_rsi"] = True
    return cfg


def _build_signal_fn(cfg: dict, precomputed: dict):
    """Construye una función de señal RÁPIDA usando arrays precomputados.

    NO recalcula indicadores por cada vela — usa `precomputed` ya calculado.
    """

    conditions: list[tuple[str, Callable[[int], bool]]] = []

    # ── RSI ──
    if cfg.get("use_rsi"):
        rsi_arr = precomputed["rsi_arr"]
        threshold = cfg["rsi_oversold"]
        conditions.append(("rsi", lambda i, a=rsi_arr, t=threshold: (
            i < len(a) and not np.isnan(a[i]) and float(a[i]) < t
        )))

    # ── MACD valley ──
    if cfg.get("use_macd"):
        hist_arr = precomputed["hist_arr"]
        min_abs = cfg["macd_hist_min"]
        conditions.append(("macd", lambda i, h=hist_arr, m=min_abs: (
            i >= 4
            and not any(np.isnan(h[j]) for j in range(i - 3, i + 1))
            and float(h[i - 3]) >= float(h[i - 2]) > float(h[i - 1]) < float(h[i])
            and float(h[i - 1]) < 0
            and abs(float(h[i - 1])) >= m
        )))

    # ── Bollinger Bands ──
    if cfg.get("use_bb"):
        bb_low = precomputed["bb_low"]
        bb_pos = float(cfg["bb_position_pct"])
        closes_p = precomputed["closes"]
        conditions.append(("bb", lambda i, bl=bb_low, bp=bb_pos, c=closes_p: (
            i < len(bl) and not np.isnan(bl[i])
            and float(c[i]) <= float(bl[i]) * (1.0 + bp / 100.0)
        )))

    # ── Volumen ──
    if cfg.get("use_volume"):
        vol_arr = precomputed["vols"]
        vol_ma = precomputed["vol_ma"]
        vm = float(cfg["vol_multiplier"])
        conditions.append(("vol", lambda i, v=vol_arr, ma=vol_ma, m=vm: (
            i < len(ma) and not np.isnan(ma[i]) and float(ma[i]) > 0
            and float(v[i]) > float(ma[i]) * m
        )))

    # ── MA Crossover ──
    if cfg.get("use_ma_cross"):
        ma_s = precomputed["ma_short_arr"]
        ma_l = precomputed["ma_long_arr"]
        conditions.append(("ma_cross", lambda i, ms=ma_s, ml=ma_l: (
            i >= 1
            and i < len(ms) and i < len(ml)
            and not np.isnan(ms[i - 1]) and not np.isnan(ml[i - 1])
            and not np.isnan(ms[i]) and not np.isnan(ml[i])
            and float(ms[i - 1]) <= float(ml[i - 1])
            and float(ms[i]) > float(ml[i])
        )))

    # ── Momentum / ROC ──
    if cfg.get("use_momentum"):
        closes_m = precomputed["closes"]
        mom_period = int(cfg["momentum_period"])
        mom_th = float(cfg["momentum_threshold"])
        conditions.append(("momentum", lambda i, c=closes_m, p=mom_period, t=mom_th: (
            i >= p
            and (float(c[i]) - float(c[i - p])) / float(c[i - p]) * 100.0 > t
        )))

    if not conditions:
        def _always(_pre: dict, _i: int, _cfg: dict | None = None) -> str | None:
            return "buy"
        return _always

    # OR logic: any active condition triggers buy
    def _signal(_pre: dict, i: int, _cfg: dict | None = None) -> str | None:
        for _name, check in conditions:
            if check(i):
                return "buy"
        return None

    return _signal


RANDOM_SEARCH_CSV = SIM_DIR / "random_search_history.csv"
RANDOM_SEARCH_JSON_OLD = SIM_DIR / "random_search_history.json"

# ── Orden canónico de columnas ──
_CSV_META_COLS = ["global_id", "timestamp", "run_tag", "fee_rate"]
_CSV_PARAM_COLS = ["invest_percent"] + sorted(RANDOM_PARAM_SPACE.keys())
_CSV_RESULT_COLS = [
    "total_ops", "ganadoras", "perdedoras", "winrate",
    "ganancia_media_por_op", "ganancia_media_ganadoras", "perdida_media_perdedoras",
    "mejor_operacion", "peor_operacion",
    "pnl_neto", "pnl_por_operacion",
    "roi_mensual", "pnl_mensual", "max_drawdown", "score",
    "tiempo_medio_h", "ops_por_mes",
]
_CSV_ALL_COLS = _CSV_META_COLS + _CSV_PARAM_COLS + _CSV_RESULT_COLS


def _migrate_json_to_csv() -> int:
    """Migra el JSON antiguo a CSV si existe. Devuelve número de entradas migradas."""
    if not RANDOM_SEARCH_JSON_OLD.exists():
        return 0
    try:
        with open(RANDOM_SEARCH_JSON_OLD) as f:
            old_data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return 0

    rows = []
    for entry in old_data:
        row: dict = {
            "global_id": entry.get("global_id", 0),
            "timestamp": entry.get("timestamp", ""),
            "run_tag": entry.get("run_tag", ""),
            "fee_rate": entry.get("fee_rate", 0.0),
        }
        # Flatten params
        params = entry.get("params", {})
        for k in _CSV_PARAM_COLS:
            row[k] = params.get(k, "")
        # Flatten results
        results = entry.get("results", {})
        for k in _CSV_RESULT_COLS:
            row[k] = results.get(k, 0)
        # Handle nested results dict
        if isinstance(results, dict):
            for k in _CSV_RESULT_COLS:
                if k in results:
                    row[k] = results[k]
        rows.append(row)

    if rows:
        _write_csv(rows)
        # Rename old JSON as backup
        RANDOM_SEARCH_JSON_OLD.rename(RANDOM_SEARCH_JSON_OLD.with_suffix(".json.bak"))
        print(f"  📦 Migradas {len(rows)} entradas de JSON → CSV")

    return len(rows)


def _load_csv_history() -> list[dict]:
    """Carga el histórico CSV. Si no existe, migra desde JSON."""
    # Auto-migrate from old JSON
    if not RANDOM_SEARCH_CSV.exists() and RANDOM_SEARCH_JSON_OLD.exists():
        _migrate_json_to_csv()

    if not RANDOM_SEARCH_CSV.exists():
        return []

    rows = []
    with open(RANDOM_SEARCH_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric strings back to proper types
            parsed: dict = {}
            for k, v in row.items():
                if k in _CSV_PARAM_COLS:
                    # Try int, then float, keep as string if neither
                    if v == "True":
                        parsed[k] = True
                    elif v == "False":
                        parsed[k] = False
                    elif v == "":
                        parsed[k] = ""
                    else:
                        try:
                            parsed[k] = int(v) if "." not in v else float(v)
                        except ValueError:
                            parsed[k] = v
                elif k in _CSV_RESULT_COLS or k in ("global_id", "fee_rate"):
                    try:
                        parsed[k] = float(v) if "." in v else int(v)
                    except (ValueError, TypeError):
                        parsed[k] = v
                else:
                    parsed[k] = v
            rows.append(parsed)
    return rows


def _write_csv(rows: list[dict]) -> None:
    """Escribe (sobrescribe) el CSV con las filas dadas."""
    SIM_DIR.mkdir(parents=True, exist_ok=True)
    with open(RANDOM_SEARCH_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_ALL_COLS)
        writer.writeheader()
        for row in rows:
            # Ensure all columns present
            clean: dict = {}
            for col in _CSV_ALL_COLS:
                val = row.get(col, "")
                if isinstance(val, bool):
                    val = str(val)
                clean[col] = val
            writer.writerow(clean)


def _entry_to_csv_row(cfg: dict, r: ResultadoSim, fee_rate: float, run_tag: str, gid: int) -> dict:
    """Convierte una iteración a fila plana para CSV."""
    row: dict = {
        "global_id": gid,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_tag": run_tag,
        "fee_rate": fee_rate,
    }
    # Params
    for k in _CSV_PARAM_COLS:
        row[k] = cfg.get(k, "")
    # Results
    row["total_ops"] = r.total_ops
    row["ganadoras"] = r.ganadoras
    row["perdedoras"] = r.perdedoras
    row["winrate"] = round(r.winrate, 2)
    row["ganancia_media_por_op"] = round(r.ganancia_media_por_op, 2)
    row["ganancia_media_ganadoras"] = round(r.ganancia_media_ganadoras, 2)
    row["perdida_media_perdedoras"] = round(r.perdida_media_perdedoras, 2)
    row["mejor_operacion"] = round(r.mejor_operacion, 2)
    row["peor_operacion"] = round(r.peor_operacion, 2)
    row["pnl_neto"] = round(r.pnl_neto, 2)
    row["pnl_por_operacion"] = round(r.pnl_por_operacion, 2)
    row["roi_mensual"] = round(r.roi_mensual, 2)
    row["pnl_mensual"] = round(r.pnl_mensual, 2)
    row["max_drawdown"] = round(r.max_drawdown, 2)
    row["score"] = round(r.score, 2)
    row["tiempo_medio_h"] = round(r.tiempo_medio_h, 2)
    row["ops_por_mes"] = round(r.ops_por_mes, 2)
    return row


# ── Worker para multiprocessing (top-level, pickleable) ──

def _simulate_one_iteration(args: tuple) -> dict:
    """Ejecuta UNA iteración del random search. Debe ser top-level para multiprocessing.

    Args:
        args: (ohlcv, closes, highs, lows, vols, total_dias, n, capital, fee_rate, index)

    Returns:
        dict con 'cfg', 'results', 'trades_count', 'index'
    """
    ohlcv, closes, highs, lows, vols, total_dias, n, capital, fee_rate, index = args

    cfg = _sample_random_params()

    invest_pct = cfg["invest_percent"] / 100.0
    tp_rate = cfg["tp_percent"] / 100.0
    sl_rate = cfg["sl_percent"] / 100.0

    max_candles = int(cfg["max_position_candles"])
    min_between = int(cfg["min_candles_between"])
    confirm = int(cfg["confirmation_candles"])

    use_atr_sl = cfg.get("use_atr_sl", False)
    atr_period = int(cfg["atr_period"])
    atr_mult = float(cfg["atr_mult_sl"])

    fee_por_op = fee_rate / 100.0

    # ── Precomputar indicadores ──
    precomp: dict = {"closes": closes, "vols": vols, "highs": highs, "lows": lows}

    if cfg.get("use_rsi"):
        precomp["rsi_arr"] = _rsi(closes, int(cfg["rsi_period"]))

    if cfg.get("use_macd"):
        fast = int(cfg["macd_fast"])
        slow = int(cfg["macd_slow"])
        sig_p = int(cfg["macd_signal_period"])
        ef = _ema(closes, fast)
        es = _ema(closes, slow)
        ml = ef - es
        msig = _ema(ml, sig_p)
        precomp["hist_arr"] = ml - msig

    if cfg.get("use_bb"):
        bp = int(cfg["bb_period"])
        bm = _sma(closes, bp)
        bs = _std(closes, bp)
        precomp["bb_low"] = bm - bs * float(cfg["bb_std_mult"])

    if cfg.get("use_volume"):
        precomp["vol_ma"] = _sma(vols, int(cfg["vol_ma_period"]))

    if cfg.get("use_ma_cross"):
        precomp["ma_short_arr"] = _sma(closes, int(cfg["ma_short"]))
        precomp["ma_long_arr"] = _sma(closes, int(cfg["ma_long"]))

    atr_arr = _atr(ohlcv, atr_period) if use_atr_sl else np.array([])

    senal_fn = _build_signal_fn(cfg, precomp)

    # ── Simulation loop ──
    trades: list[tuple] = []
    eur = capital
    btc = 0.0
    in_pos = False
    entry_p = 0.0
    entry_i = -1
    tp_price = 0.0
    sl_price_val = 0.0
    total_fees = 0.0
    last_sell_i = -999
    confirming = 0
    pending_buy = False

    for i in range(n):
        close_val = float(closes[i])

        if in_pos:
            exit_type = None
            if close_val >= tp_price:
                exit_type = "TP"
            elif close_val <= sl_price_val:
                exit_type = "SL"
            elif i - entry_i >= max_candles:
                exit_type = "MAX_TIME"

            if exit_type:
                pnl_pct = (close_val - entry_p) / entry_p * 100
                fee = btc * close_val * fee_por_op
                total_fees += fee
                trades.append((entry_p, close_val, pnl_pct, i - entry_i, exit_type))
                eur += btc * close_val - fee
                btc = 0.0
                in_pos = False
                last_sell_i = i
                confirming = 0
                pending_buy = False
                continue

        if not in_pos and i - last_sell_i > min_between:
            accion = senal_fn({}, i, {})  # signal fn uses closures, not these args
            if accion == "buy":
                if confirm <= 0:
                    # Execute buy immediately
                    invest = eur * invest_pct
                    if invest >= MIN_TRADE:
                        buy_fee = invest * fee_por_op
                        total_fees += buy_fee
                        btc = invest / close_val
                        eur -= invest + buy_fee
                        entry_p = close_val
                        entry_i = i
                        in_pos = True
                        tp_price = entry_p * (1 + tp_rate)
                        if use_atr_sl and i >= atr_period and len(atr_arr) > i and not np.isnan(atr_arr[i]):
                            sl_price_val = entry_p - atr_arr[i] * atr_mult
                        else:
                            sl_price_val = entry_p * (1 - sl_rate)
                elif not pending_buy:
                    pending_buy = True
                    confirming = 1
                else:
                    confirming += 1
                    if confirming >= confirm:
                        invest = eur * invest_pct
                        if invest >= MIN_TRADE:
                            buy_fee = invest * fee_por_op
                            total_fees += buy_fee
                            btc = invest / close_val
                            eur -= invest + buy_fee
                            entry_p = close_val
                            entry_i = i
                            in_pos = True
                            tp_price = entry_p * (1 + tp_rate)
                            if use_atr_sl and i >= atr_period and len(atr_arr) > i and not np.isnan(atr_arr[i]):
                                sl_price_val = entry_p - atr_arr[i] * atr_mult
                            else:
                                sl_price_val = entry_p * (1 - sl_rate)
                            pending_buy = False
                            confirming = 0
            else:
                pending_buy = False
                confirming = 0

    if in_pos:
        last_c = float(closes[-1])
        pnl_pct = (last_c - entry_p) / entry_p * 100
        fee = btc * last_c * fee_por_op
        total_fees += fee
        trades.append((entry_p, last_c, pnl_pct, n - entry_i, "END"))
        eur += btc * last_c - fee

    # ── Build results ──
    num = len(trades)
    pnl_neto = eur - capital

    results: dict = {
        "total_ops": num, "ganadoras": 0, "perdedoras": 0,
        "winrate": 0.0, "ganancia_media_por_op": 0.0,
        "ganancia_media_ganadoras": 0.0, "perdida_media_perdedoras": 0.0,
        "mejor_operacion": 0.0, "peor_operacion": 0.0,
        "pnl_neto": round(pnl_neto, 2), "comisiones": round(total_fees, 2),
        "pnl_por_operacion": 0.0, "roi_mensual": 0.0, "pnl_mensual": 0.0,
        "max_drawdown": 0.0, "score": 0.0,
        "tiempo_medio_h": 0.0, "ops_por_mes": 0.0,
    }

    if num > 0:
        gains_pct = [t[2] for t in trades]
        wins = [g for g in gains_pct if g > 0]
        losses = [g for g in gains_pct if g <= 0]
        durs = [t[3] for t in trades]

        results["ganadoras"] = len(wins)
        results["perdedoras"] = len(losses)
        results["winrate"] = round(len(wins) / num * 100, 2)
        results["ganancia_media_por_op"] = round(float(np.mean(gains_pct)), 2)
        results["ganancia_media_ganadoras"] = round(float(np.mean(wins)), 2) if wins else 0.0
        results["perdida_media_perdedoras"] = round(float(np.mean(losses)), 2) if losses else 0.0
        results["mejor_operacion"] = round(float(max(gains_pct)), 2)
        results["peor_operacion"] = round(float(min(gains_pct)), 2)
        results["pnl_por_operacion"] = round(pnl_neto / num, 2)
        pnl_diario = pnl_neto / total_dias
        results["pnl_mensual"] = round(pnl_diario * 30, 2)
        results["roi_mensual"] = round((pnl_neto / capital * 100) / total_dias * 30, 2)
        results["tiempo_medio_h"] = round(float(np.mean(durs)), 2)
        results["ops_por_mes"] = round(num / total_dias * 30, 2)

        # Drawdown
        max_dd = 0.0
        peak_bal = capital
        bal = capital
        for g in gains_pct:
            bal += bal * (g / 100) * invest_pct
            if bal > peak_bal:
                peak_bal = bal
            dd = (peak_bal - bal) / peak_bal * 100
            if dd > max_dd:
                max_dd = dd
        results["max_drawdown"] = round(max_dd, 2)
        # Score = € netos al mes (lo que realmente ganas)
        results["score"] = round(results["pnl_mensual"], 2)

    return {"cfg": cfg, "results": results, "index": index}


def simular_random_search(
    ohlcv: list[list[float]],
    pre: dict,
    num_iter: int,
    capital: float = INITIAL_EUR,
    fee_rate: float = FEE_RATE,
    show_progress: bool = True,
    workers: int | None = None,
) -> tuple[list[ResultadoSim], list[dict]]:
    """Prueba N combinaciones aleatorias en paralelo usando todos los cores disponibles.

    Args:
        ohlcv: velas OHLCV
        pre: precalculos comunes
        num_iter: número de iteraciones a ejecutar
        capital: capital inicial en €
        fee_rate: fee en % (ej: 0.08)
        show_progress: mostrar progreso en consola
        workers: número de workers paralelos (default: todos los CPUs)

    Returns:
        (resultados_ordenados_por_score, configs_correspondientes)
    """
    if workers is None:
        workers = os.cpu_count() or 4

    closes = pre["closes"]
    highs = pre["highs"]
    lows = pre["lows"]
    vols = pre["vols"]
    total_dias = len(ohlcv) / 24.0
    n = len(ohlcv)

    # ── Prepare args for each iteration ──
    task_args = [
        (ohlcv, closes, highs, lows, vols, total_dias, n, capital, fee_rate, idx)
        for idx in range(num_iter)
    ]

    if show_progress:
        print(f"  🚀 Ejecutando {num_iter} iteraciones en paralelo "
              f"({workers} workers, {os.cpu_count() or '?'} CPUs)...")
        print(f"  {'─'*55}")

    resultados: list[ResultadoSim] = []
    configs: list[dict] = []
    completed = 0

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_simulate_one_iteration, args): args[-1]
                   for args in task_args}

        for future in as_completed(futures):
            completed += 1
            data = future.result()
            cfg = data["cfg"]
            res = data["results"]
            idx = data["index"]

            # ── Build ResultadoSim ──
            desc_parts = []
            for k in sorted(cfg):
                v = cfg[k]
                desc_parts.append(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}")

            r = ResultadoSim(
                estrategia=f"RandomSearch #{idx + 1}",
                descripcion=" | ".join(desc_parts),
            )
            r.total_ops = res["total_ops"]
            r.ganadoras = res["ganadoras"]
            r.perdedoras = res["perdedoras"]
            r.winrate = res["winrate"]
            r.ganancia_media_por_op = res["ganancia_media_por_op"]
            r.ganancia_media_ganadoras = res["ganancia_media_ganadoras"]
            r.perdida_media_perdedoras = res["perdida_media_perdedoras"]
            r.mejor_operacion = res["mejor_operacion"]
            r.peor_operacion = res["peor_operacion"]
            r.pnl_neto = res["pnl_neto"]
            r.comisiones = res["comisiones"]
            r.pnl_por_operacion = res["pnl_por_operacion"]
            r.roi_mensual = res["roi_mensual"]
            r.pnl_mensual = res["pnl_mensual"]
            r.max_drawdown = res["max_drawdown"]
            r.score = res["score"]
            r.tiempo_medio_h = res["tiempo_medio_h"]
            r.dias_simulados = total_dias
            r.ops_por_mes = res["ops_por_mes"]
            r.pnl_diario = r.pnl_neto / total_dias if total_dias > 0 else 0
            r.roi_total = r.pnl_neto / capital * 100 if capital > 0 else 0
            r.roi_diario = r.roi_total / total_dias if total_dias > 0 else 0
            r.tiempo_maximo_h = 0.0
            r.pnl_bruto = r.pnl_neto + r.comisiones

            resultados.append(r)
            configs.append(cfg)

            if show_progress:
                enabled = [k for k in ("use_rsi", "use_macd", "use_bb",
                                       "use_volume", "use_ma_cross", "use_momentum")
                           if cfg.get(k)]
                print(f"  ✅ [{completed}/{num_iter}] #{idx + 1}  "
                      f"tp={cfg['tp_percent']:.1f}%  sl={cfg['sl_percent']:.1f}%  "
                      f"activos: {', '.join(enabled) if enabled else 'always'}")
                print(f"       {r.total_ops} ops | WR={r.winrate:.1f}% | "
                      f"%/op={r.ganancia_media_por_op:+.2f} | "
                      f"ROI/mes={r.roi_mensual:+.2f}% | Score={r.score:.2f}")

    # ── Persistir en CSV acumulativo ──
    run_tag = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    history = _load_csv_history()
    global_id_start = len(history) + 1

    new_rows = []
    for i, (r, cfg) in enumerate(zip(resultados, configs)):
        row = _entry_to_csv_row(cfg, r, fee_rate, run_tag, global_id_start + i)
        new_rows.append(row)

    # Append new rows to existing CSV
    all_rows = history + new_rows
    _write_csv(all_rows)

    if show_progress:
        size_kb = RANDOM_SEARCH_CSV.stat().st_size / 1024 if RANDOM_SEARCH_CSV.exists() else 0
        print(f"\n  💾 {num_iter} iteraciones guardadas en {RANDOM_SEARCH_CSV.name} ({size_kb:.0f} KB)")
        print(f"     Total acumulado: {len(all_rows)} combinaciones")

    # ── Construir ResultadoSim desde TODO el histórico ──
    all_results, all_configs = _csv_rows_to_resultados(all_rows)

    # Sort by score descending
    sorted_pairs = sorted(zip(all_results, all_configs), key=lambda x: x[0].score, reverse=True)
    sorted_results = [p[0] for p in sorted_pairs]
    sorted_configs = [p[1] for p in sorted_pairs]

    return sorted_results, sorted_configs


def _csv_rows_to_resultados(rows: list[dict]) -> tuple[list[ResultadoSim], list[dict]]:
    """Convierte filas planas de CSV a listas de ResultadoSim y configs."""
    resultados: list[ResultadoSim] = []
    configs: list[dict] = []

    for row in rows:
        gid = row.get("global_id", 0)
        ts = str(row.get("timestamp", ""))[:10]

        # Extract params (only keys in _CSV_PARAM_COLS)
        params = {}
        for k in _CSV_PARAM_COLS:
            val = row.get(k, "")
            if val != "":
                params[k] = val

        r = ResultadoSim(
            estrategia=f"#{gid} ({ts})",
            descripcion=" | ".join(
                f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in sorted(params.items())
            ),
        )
        r.total_ops = int(row.get("total_ops", 0))
        r.ganadoras = int(row.get("ganadoras", 0))
        r.perdedoras = int(row.get("perdedoras", 0))
        r.winrate = float(row.get("winrate", 0))
        r.ganancia_media_por_op = float(row.get("ganancia_media_por_op", 0))
        r.ganancia_media_ganadoras = float(row.get("ganancia_media_ganadoras", 0))
        r.perdida_media_perdedoras = float(row.get("perdida_media_perdedoras", 0))
        r.mejor_operacion = float(row.get("mejor_operacion", 0))
        r.peor_operacion = float(row.get("peor_operacion", 0))
        r.pnl_neto = float(row.get("pnl_neto", 0))
        r.pnl_por_operacion = float(row.get("pnl_por_operacion", 0))
        r.roi_mensual = float(row.get("roi_mensual", 0))
        r.pnl_mensual = float(row.get("pnl_mensual", 0))
        r.max_drawdown = float(row.get("max_drawdown", 0))
        # Score = € netos al mes (lo que realmente ganas), calculado al vuelo
        r.score = r.pnl_mensual
        r.tiempo_medio_h = float(row.get("tiempo_medio_h", 0))
        r.ops_por_mes = float(row.get("ops_por_mes", 0))

        resultados.append(r)
        configs.append(params)

    return resultados, configs


def imprimir_random_top3(resultados: list[ResultadoSim], configs: list[dict]):
    """Imprime las 3 mejores combinaciones (del histórico + sesión actual).

    Solo entran en el ranking las que tienen ≥ MIN_OPS_FOR_RANKING operaciones.
    """
    # Filter by minimum operations
    qualifying = [(r, c) for r, c in zip(resultados, configs)
                  if r.total_ops >= MIN_OPS_FOR_RANKING]

    top3_pairs = qualifying[:3]
    if not top3_pairs:
        print(f"\n  ⚠️  Ninguna combinación alcanza las {MIN_OPS_FOR_RANKING} ops mínimas para el ranking.")
        return

    top3_r = [p[0] for p in top3_pairs]
    top3_c = [p[1] for p in top3_pairs]

    # Count totals from CSV
    total_count = 0
    qualifying_count = len(qualifying)
    if RANDOM_SEARCH_CSV.exists():
        with open(RANDOM_SEARCH_CSV) as f:
            total_count = sum(1 for _ in f) - 1  # subtract header

    print(f"\n{'='*80}")
    print(f"  🏆 TOP 3 MEJORES COMBINACIONES (Random Search)")
    print(f"     {total_count} total | {qualifying_count} con ≥{MIN_OPS_FOR_RANKING} ops")
    print(f"{'='*80}")
    print(f"  {'#':<3} {'ID (fecha)':<16} {'Ops':<5} {'WR%':<6} {'%/op':<7} {'€/op':<8}"
          f" {'ROI/mes':<9} {'€/mes':<9} {'DD%':<7} {'Score':<8}")
    print(f"  {'─'*83}")

    for idx, (r, c) in enumerate(zip(top3_r, top3_c)):
        icon = "🥇" if idx == 0 else "🥈" if idx == 1 else "🥉"
        print(f"  {icon} {r.estrategia:<16} {r.total_ops:<5} {r.winrate:<6.1f}"
              f" {r.ganancia_media_por_op:<+7.2f} {r.pnl_por_operacion:<+8.2f}"
              f" {r.roi_mensual:<+9.2f} {r.pnl_mensual:<+9.2f}"
              f" {r.max_drawdown:<7.2f} {r.score:<8.2f}")

    print(f"\n{'─'*80}")
    print(f"  📋 DETALLE DE PARÁMETROS DEL TOP 3")
    print(f"{'─'*80}")

    for idx, (r, c) in enumerate(zip(top3_r, top3_c)):
        icon = "🥇" if idx == 0 else "🥈" if idx == 1 else "🥉"
        print(f"\n  {icon} {r.estrategia} — Score: {r.score:.2f} | WR: {r.winrate:.1f}% | "
              f"ROI/mes: {r.roi_mensual:+.2f}% | Ops: {r.total_ops}")

        # Group params by category
        risk_keys = ["tp_percent", "sl_percent", "invest_percent", "max_position_candles",
                      "min_candles_between", "confirmation_candles"]
        rsi_keys = ["use_rsi", "rsi_period", "rsi_oversold"]
        macd_keys = ["use_macd", "macd_fast", "macd_slow", "macd_signal_period", "macd_hist_min"]
        bb_keys = ["use_bb", "bb_period", "bb_std_mult", "bb_position_pct"]
        vol_keys = ["use_volume", "vol_ma_period", "vol_multiplier"]
        ma_keys = ["use_ma_cross", "ma_short", "ma_long"]
        atr_keys = ["use_atr_sl", "atr_period", "atr_mult_sl"]
        mom_keys = ["use_momentum", "momentum_period", "momentum_threshold"]

        def _fmt(v):
            if isinstance(v, bool):
                return "SÍ" if v else "NO"
            if isinstance(v, float):
                return f"{v:.2f}"
            return str(v)

        def _print_group(title: str, keys: list[str]):
            vals = []
            for k in keys:
                if k in c:
                    vals.append(f"{k}={_fmt(c[k])}")
            if vals:
                print(f"    {title}: {'  '.join(vals)}")

        _print_group("Riesgo", risk_keys)
        _print_group("RSI", rsi_keys)
        _print_group("MACD", macd_keys)
        _print_group("Bollinger", bb_keys)
        _print_group("Volumen", vol_keys)
        _print_group("MA Cross", ma_keys)
        _print_group("ATR", atr_keys)
        _print_group("Momentum", mom_keys)


# ── Registry de estrategias ──

ESTRATEGIAS: list[dict] = [
    {
        "id": "macd_puro",
        "nombre": "MACD puro",
        "senal": senal_macd,
        "config": {"min_histogram_abs": 60, "sl_percent": 4.95, "trailing_min_gain": 1.05, "fee_percent": 0.0},
    },
    {
        "id": "macd_rsi_filtro",
        "nombre": "MACD+RSI<40",
        "senal": senal_macd_rsi,
        "config": {"min_histogram_abs": 60, "rsi_max": 40, "sl_percent": 4.95, "trailing_min_gain": 1.05, "fee_percent": 0.0},
    },
    {
        "id": "macd_rsi_30",
        "nombre": "MACD+RSI<30",
        "senal": senal_macd_rsi,
        "config": {"min_histogram_abs": 60, "rsi_max": 30, "sl_percent": 4.95, "trailing_min_gain": 1.05, "fee_percent": 0.0},
    },
    {
        "id": "bb_rsi",
        "nombre": "BB+RSI (reversion)",
        "senal": senal_bb_rsi,
        "config": {"rsi_max": 35, "sl_percent": 4.95, "trailing_min_gain": 1.05, "fee_percent": 0.0},
    },
    {
        "id": "macd_volumen",
        "nombre": "MACD+Volumen",
        "senal": senal_macd_vol,
        "config": {"min_histogram_abs": 60, "vol_mult": 1.5, "sl_percent": 4.95, "trailing_min_gain": 1.05, "fee_percent": 0.0},
    },
    {
        "id": "momentum",
        "nombre": "Momentum (EMA20+Vol)",
        "senal": senal_momentum,
        "config": {"vol_mult": 1.5, "sl_percent": 4.95, "trailing_min_gain": 1.05, "fee_percent": 0.0},
    },
    {
        "id": "breakout",
        "nombre": "Breakout",
        "senal": senal_breakout,
        "config": {"lookback": 20, "vol_threshold": 1.5, "sl_percent": 4.95, "trailing_min_gain": 1.05, "fee_percent": 0.0},
    },
    {
        "id": "triple_confirmacion",
        "nombre": "Triple Confirmacion",
        "senal": senal_triple,
        "config": {"min_histogram_abs": 50, "rsi_max": 30, "bb_max_distance_pct": 2.0, "sl_percent": 4.95, "trailing_min_gain": 1.05, "fee_percent": 0.0},
    },
    {
        "id": "macd_hist_50",
        "nombre": "MACD Hist -50 (doble entrada)",
        "senal": None,  # usa simulación dedicada
        "config": {"invest_percent": 45.0, "drop_2nd_entry_pct": 2.0, "tp_percent": 1.0, "sl_percent": 1.0, "max_position_candles": 12, "fee_percent": 0.0},
        "custom_sim": True,
    },
    {
        "id": "random_search",
        "nombre": "🎲 Random Search (N iteraciones)",
        "senal": None,  # usa simulación dedicada
        "config": {"fee_percent": 0.0},
        "custom_sim": True,
        "random_search": True,
    },
]


# ── Motor de simulacion ──

def simular_estrategia(
    ohlcv: list[list[float]],
    pre: dict,
    est: dict,
    capital: float = INITIAL_EUR,
    fee_rate: float = FEE_RATE,
    return_trades: bool = False,
) -> ResultadoSim | tuple[ResultadoSim, list[tuple]]:
    """Ejecuta la simulacion para una estrategia y devuelve ResultadoSim.

    Si return_trades=True, devuelve (ResultadoSim, trades).
    """
    cfg = est["config"]
    senal_fn = est["senal"]
    sl_rate = cfg.get("sl_percent", 4.95) / 100.0
    trail_min = cfg.get("trailing_min_gain", 1.0)
    fee_por_op = fee_rate / 100.0  # fee_rate en % (ej: 0.15 = 0.15%)

    closes = pre["closes"]

    trades: list[tuple] = []  # (entry_price, exit_price, gain_%, duration_h, tipo)
    eur = capital; btc = 0.0
    in_pos = False; entry_p = 0.0; entry_i = 0
    highest = 0.0; sl_price = None
    total_fees = 0.0

    for i in range(len(ohlcv)):
        close = float(closes[i])

        if in_pos:
            exit_type = None
            if sl_price is not None and close <= sl_price:
                exit_type = "SL"
            elif close > highest:
                highest = close
                gain = (close - entry_p) / entry_p * 100
                if gain > trail_min:
                    trail_pct = gain * TRAIL_RETAIN
                    new_sl = entry_p * (1 + trail_pct / 100.0)
                    if sl_price is None or new_sl > sl_price: sl_price = new_sl
            if i - entry_i >= MAX_POSITION_CANDLES:
                if exit_type is None or exit_type == "SL":
                    pass  # MAX_TIME tiene prioridad si no ha saltado SL
                if exit_type is None:
                    exit_type = "MAX_TIME"

            # Realmente comprobamos en orden: SL > MAX_TIME
            if sl_price is not None and close <= sl_price:
                exit_type = "SL"
            elif i - entry_i >= MAX_POSITION_CANDLES:
                exit_type = "MAX_TIME"

            if exit_type:
                pnl_pct = (close - entry_p) / entry_p * 100
                fee = btc * close * fee_por_op
                total_fees += fee
                trades.append((entry_p, close, pnl_pct, i - entry_i, exit_type))
                eur += btc * close - fee; btc = 0.0; in_pos = False
                continue

        # Senal
        if not in_pos:
            accion = senal_fn(pre, i, cfg)
            if accion == "buy":
                invest = eur * INVEST_PCT / 100.0
                if invest >= MIN_TRADE:
                    buy_fee = invest * fee_por_op
                    total_fees += buy_fee
                    btc = invest / close
                    eur -= invest + buy_fee
                    entry_p = close; entry_i = i
                    highest = close; sl_price = entry_p * (1 - sl_rate)
                    in_pos = True

    # Cerrar si quedo abierta
    if in_pos:
        last_c = float(closes[-1])
        pnl_pct = (last_c - entry_p) / entry_p * 100
        fee = btc * last_c * fee_por_op
        total_fees += fee
        trades.append((entry_p, last_c, pnl_pct, len(ohlcv) - entry_i, "END"))
        eur += btc * last_c - fee

    # Construir resultado
    r = ResultadoSim(
        estrategia=est["nombre"],
        descripcion=" | ".join(f"{k}={v}" for k,v in sorted(cfg.items())),
    )

    num = len(trades)
    total_dias = len(ohlcv) / 24.0
    pnl_neto = eur - capital

    if num == 0:
        r.dias_simulados = total_dias
        r.comisiones = total_fees
        return r

    gains_pct = [t[2] for t in trades]
    wins = [g for g in gains_pct if g > 0]
    losses = [g for g in gains_pct if g <= 0]
    durs = [t[3] for t in trades]

    pnl_bruto = pnl_neto + total_fees

    r.total_ops = num
    r.ganadoras = len(wins)
    r.perdedoras = len(losses)
    r.winrate = len(wins) / num * 100
    r.ganancia_media_por_op = float(np.mean(gains_pct))
    r.ganancia_media_ganadoras = float(np.mean(wins)) if wins else 0.0
    r.perdida_media_perdedoras = float(np.mean(losses)) if losses else 0.0
    r.mejor_operacion = float(max(gains_pct))
    r.peor_operacion = float(min(gains_pct))

    r.pnl_bruto = pnl_bruto
    r.pnl_neto = pnl_neto
    r.comisiones = total_fees
    r.pnl_por_operacion = pnl_neto / num
    r.pnl_diario = pnl_neto / total_dias
    r.pnl_mensual = r.pnl_diario * 30

    r.roi_total = pnl_neto / capital * 100
    r.roi_diario = r.roi_total / total_dias
    r.roi_mensual = r.roi_diario * 30

    r.tiempo_medio_h = float(np.mean(durs))
    r.tiempo_maximo_h = float(max(durs))
    r.dias_simulados = total_dias
    r.ops_por_mes = num / total_dias * 30

    # Drawdown
    max_dd = 0.0; peak = capital; bal = capital
    for g in gains_pct:
        bal += bal * (g / 100) * INVEST_PCT / 100
        if bal > peak: peak = bal
        dd = (peak - bal) / peak * 100
        if dd > max_dd: max_dd = dd
    r.max_drawdown = max_dd

    # Score: pondera ROI mensual, winrate, PnL por operacion, penaliza drawdown
    r.score = r.roi_mensual * 0.4 + r.winrate * 0.15 + r.pnl_por_operacion * 0.3 - r.max_drawdown * 0.15

    if return_trades:
        return r, trades
    return r


def simular_estrategia_con_indices(
    ohlcv: list[list[float]],
    pre: dict,
    est: dict,
    capital: float = INITIAL_EUR,
    fee_rate: float = FEE_RATE,
) -> tuple[ResultadoSim, list[tuple]]:
    """Como simular_estrategia pero los trades incluyen (entry_idx, sell_idx)."""
    cfg = est["config"]
    senal_fn = est["senal"]
    sl_rate = cfg.get("sl_percent", 4.95) / 100.0
    trail_min = cfg.get("trailing_min_gain", 1.0)
    fee_por_op = fee_rate / 100.0

    closes = pre["closes"]

    trades: list[tuple] = []  # (entry_price, exit_price, gain_%, duration_h, tipo, buy_idx, sell_idx)
    eur = capital; btc = 0.0
    in_pos = False; entry_p = 0.0; entry_i = -1
    highest = 0.0; sl_price = None
    total_fees = 0.0

    for i in range(len(ohlcv)):
        close = float(closes[i])

        if in_pos:
            exit_type = None
            if sl_price is not None and close <= sl_price:
                exit_type = "SL"
            elif close > highest:
                highest = close
                gain = (close - entry_p) / entry_p * 100
                if gain > trail_min:
                    trail_pct = gain * TRAIL_RETAIN
                    new_sl = entry_p * (1 + trail_pct / 100.0)
                    if sl_price is None or new_sl > sl_price: sl_price = new_sl
            if i - entry_i >= MAX_POSITION_CANDLES:
                exit_type = "MAX_TIME"
            if sl_price is not None and close <= sl_price:
                exit_type = "SL"
            elif i - entry_i >= MAX_POSITION_CANDLES:
                exit_type = "MAX_TIME"

            if exit_type:
                pnl_pct = (close - entry_p) / entry_p * 100
                fee = btc * close * fee_por_op
                total_fees += fee
                trades.append((entry_p, close, pnl_pct, i - entry_i, exit_type, entry_i, i))
                eur += btc * close - fee; btc = 0.0; in_pos = False
                continue

        if not in_pos:
            accion = senal_fn(pre, i, cfg)
            if accion == "buy":
                invest = eur * INVEST_PCT / 100.0
                if invest >= MIN_TRADE:
                    buy_fee = invest * fee_por_op
                    total_fees += buy_fee
                    btc = invest / close
                    eur -= invest + buy_fee
                    entry_p = close; entry_i = i
                    highest = close; sl_price = entry_p * (1 - sl_rate)
                    in_pos = True

    if in_pos:
        last_c = float(closes[-1])
        pnl_pct = (last_c - entry_p) / entry_p * 100
        fee = btc * last_c * fee_por_op
        total_fees += fee
        trades.append((entry_p, last_c, pnl_pct, len(ohlcv) - entry_i, "END", entry_i, len(ohlcv) - 1))
        eur += btc * last_c - fee

    # Construir resultado igual que simular_estrategia
    r = ResultadoSim(
        estrategia=est["nombre"],
        descripcion=" | ".join(f"{k}={v}" for k,v in sorted(cfg.items())),
    )
    num = len(trades)
    total_dias = len(ohlcv) / 24.0
    pnl_neto = eur - capital

    if num == 0:
        r.dias_simulados = total_dias
        r.comisiones = total_fees
        return r, []

    gains_pct = [t[2] for t in trades]
    wins = [g for g in gains_pct if g > 0]
    losses = [g for g in gains_pct if g <= 0]
    durs = [t[3] for t in trades]
    pnl_bruto = pnl_neto + total_fees

    r.total_ops = num; r.ganadoras = len(wins); r.perdedoras = len(losses)
    r.winrate = len(wins) / num * 100
    r.ganancia_media_por_op = float(np.mean(gains_pct))
    r.ganancia_media_ganadoras = float(np.mean(wins)) if wins else 0.0
    r.perdida_media_perdedoras = float(np.mean(losses)) if losses else 0.0
    r.mejor_operacion = float(max(gains_pct)); r.peor_operacion = float(min(gains_pct))
    r.pnl_bruto = pnl_bruto; r.pnl_neto = pnl_neto; r.comisiones = total_fees
    r.pnl_por_operacion = pnl_neto / num; r.pnl_diario = pnl_neto / total_dias
    r.pnl_mensual = r.pnl_diario * 30
    r.roi_total = pnl_neto / capital * 100; r.roi_diario = r.roi_total / total_dias
    r.roi_mensual = r.roi_diario * 30
    r.tiempo_medio_h = float(np.mean(durs)); r.tiempo_maximo_h = float(max(durs))
    r.dias_simulados = total_dias; r.ops_por_mes = num / total_dias * 30

    max_dd = 0.0; peak = capital; bal = capital
    for g in gains_pct:
        bal += bal * (g / 100) * INVEST_PCT / 100
        if bal > peak: peak = bal
        dd = (peak - bal) / peak * 100
        if dd > max_dd: max_dd = dd
    r.max_drawdown = max_dd
    r.score = r.roi_mensual * 0.4 + r.winrate * 0.15 + r.pnl_por_operacion * 0.3 - r.max_drawdown * 0.15

    return r, trades


def imprimir_resultado(r: ResultadoSim):
    """Imprime un resultado con formato limpio."""
    print(f"\n{'─'*50}")
    print(f"  {r.estrategia}")
    print(f"  {r.descripcion}")
    print(f"{'─'*50}")
    print(f"  Operaciones:   {r.total_ops}  ({r.ops_por_mes:.1f}/mes)")
    print(f"  Win rate:      {r.winrate:.1f}%  ({r.ganadoras}G / {r.perdedoras}P)")
    print(f"  Por operacion: {r.ganancia_media_por_op:+.2f}%  ({r.pnl_por_operacion:+.2f}€)")
    print(f"    Ganadoras:   {r.ganancia_media_ganadoras:+.2f}%")
    if r.perdedoras > 0:
        print(f"    Perdedoras:  {r.perdida_media_perdedoras:+.2f}%")
    print(f"    Mejor:       {r.mejor_operacion:+.2f}%  Peor: {r.peor_operacion:+.2f}%")
    print(f"  Tiempo medio:  {r.tiempo_medio_h:.1f}h  max: {r.tiempo_maximo_h:.0f}h")
    print(f"  Comisiones:    {r.comisiones:.2f}€")
    print(f"  PnL neto:      {r.pnl_neto:+.2f}€  (bruto: {r.pnl_bruto:+.2f}€)")
    print(f"  ROI mensual:   {r.roi_mensual:+.2f}%")
    print(f"  ROI diario:    {r.roi_diario:+.2f}%")
    print(f"  PnL/dia:       {r.pnl_diario:+.2f}€")
    print(f"  PnL/mes:       {r.pnl_mensual:+.2f}€")
    print(f"  Max drawdown:  {r.max_drawdown:.2f}%")
    print(f"  Score:         {r.score:.2f}")


def imprimir_top3(resultados: list[ResultadoSim]):
    """Imprime el top 3 con formato limpio."""
    sorted_r = sorted(resultados, key=lambda x: x.score, reverse=True)
    top3 = sorted_r[:3]

    print(f"\n{'='*80}")
    print(f"  🏆 TOP 3 ESTRATEGIAS (score)")
    print(f"{'='*80}")
    print(f"  {'#':<3} {'Estrategia':<20} {'Ops':<5} {'WR%':<6} {'%/op':<7} {'€/op':<8}"
          f" {'ROI/mes':<9} {'€/mes':<9} {'DD%':<7} {'Score':<8}")
    print(f"  {'─'*83}")
    for idx, r in enumerate(top3, 1):
        print(f"  {idx:<3} {r.estrategia:<20} {r.total_ops:<5} {r.winrate:<6.1f}"
              f" {r.ganancia_media_por_op:<+7.2f} {r.pnl_por_operacion:<+8.2f}"
              f" {r.roi_mensual:<+9.2f} {r.pnl_mensual:<+9.2f}"
              f" {r.max_drawdown:<7.2f} {r.score:<8.2f}")
    print(f"  {'─'*83}")
    print(f"  ROI/mes = rentabilidad neta mensual sobre el capital total")
    print(f"  %/op   = ganancia o perdida media POR operacion")
    print(f"  Score  = 0.4*ROI + 0.15*WR + 0.3*€/op - 0.15*DD")

    # Detalle extendido del #1
    print(f"\n{'─'*50}")
    print(f"  🥇 {top3[0].estrategia} — detalle completo")
    print(f"{'─'*50}")
    print(f"  Config: {top3[0].descripcion}")
    print(f"  Operaciones: {top3[0].total_ops} ({top3[0].ops_por_mes:.1f}/mes)")
    print(f"  Win rate: {top3[0].winrate:.1f}%")
    print(f"  Ganancia media/op: {top3[0].ganancia_media_por_op:+.2f}%")
    print(f"  Ganancia media ganadoras: {top3[0].ganancia_media_ganadoras:+.2f}%")
    if top3[0].perdedoras > 0:
        print(f"  Perdida media perdedoras: {top3[0].perdida_media_perdedoras:+.2f}%")
    print(f"  Mejor op: {top3[0].mejor_operacion:+.2f}%  Peor op: {top3[0].peor_operacion:+.2f}%")
    print(f"  Tiempo medio en posicion: {top3[0].tiempo_medio_h:.1f}h")
    print(f"  PnL neto: {top3[0].pnl_neto:+.2f}€")
    print(f"  ROI mensual: {top3[0].roi_mensual:+.2f}%")
    print(f"  PnL/dia: {top3[0].pnl_diario:+.2f}€  PnL/mes: {top3[0].pnl_mensual:+.2f}€")
    print(f"  Max drawdown: {top3[0].max_drawdown:.2f}%")


# ── Punto de entrada principal ──

def ejecutar(capital: float = INITIAL_EUR, fee_rate: float = FEE_RATE) -> list[ResultadoSim]:
    """Carga velas, ejecuta todas las estrategias y devuelve resultados."""
    print(f"\n📊 SIMON — Simulador de estrategias")
    print(f"{'='*50}")
    print(f"  Capital: {capital:.0f}€")
    print(f"  Fee: {fee_rate:.2f}% por operacion")
    print(f"  Velas: {DATA_FILE}")

    ohlcv = cargar_velas()
    print(f"  {len(ohlcv)} velas 1h ({len(ohlcv)/24:.1f} dias)")
    print(f"  Periodo: {ohlcv[0][0]} -> {ohlcv[-1][0]}")

    pre = precalcular(ohlcv)
    print(f"  Estrategias a probar: {len(ESTRATEGIAS)}")

    resultados = []
    for est in ESTRATEGIAS:
        if est.get("random_search"):
            continue  # skip interactive strategies in batch mode
        if est.get("custom_sim"):
            r = simular_macd_hist_50(ohlcv, pre, est["config"], capital=capital, fee_rate=fee_rate)
        else:
            r = simular_estrategia(ohlcv, pre, est, capital=capital, fee_rate=fee_rate)
        resultados.append(r)
        print(f"    ✓ {est['nombre']} ({r.total_ops} ops, WR {r.winrate:.1f}%, "
              f"ROI {r.roi_mensual:+.2f}%)")

    # Guardar resultados en JSON
    SIM_DIR.mkdir(exist_ok=True)
    sf = SIM_DIR / "resultados.json"
    with open(sf, "w") as f:
        json.dump([r.a_dict() for r in sorted(resultados, key=lambda x: x.score, reverse=True)], f, indent=2)
    print(f"\n  Resultados guardados en {sf}")

    return resultados


def main():
    resultados = ejecutar()

    imprimir_top3(resultados)

    print(f"\n  {'─'*50}")
    print(f"  ¿Quieres ver el detalle completo de alguna?")
    print(f"  Usa: uv run python src/simulador.py --ver <nombre>")

    # Guardar en simulaciones/
    for r in resultados:
        sf2 = SIM_DIR / f"resultado_{r.estrategia.lower().replace('+','').replace(' ','_').replace('(','').replace(')','')}.json"
        with open(sf2, "w") as f:
            json.dump(r.a_dict(), f, indent=2)


if __name__ == "__main__":
    # Si pasan --ver <nombre>
    if len(sys.argv) > 2 and sys.argv[1] == "--ver":
        resultados = ejecutar()
        nombre = " ".join(sys.argv[2:]).lower()
        for r in resultados:
            if nombre in r.estrategia.lower():
                imprimir_resultado(r)
                break
        else:
            print(f"No encontre '{nombre}'")
    elif len(sys.argv) > 1 and sys.argv[1] == "--descargar":
        descargar_velas()
    else:
        main()
