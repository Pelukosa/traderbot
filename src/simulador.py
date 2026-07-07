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
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

# ── Config ──
INITIAL_EUR = 120.0        # capital simulado
INVEST_PCT = 95.0          # % del capital por operacion
TRAIL_RETAIN = 0.6         # % de ganancia que retiene el trailing
MAX_POSITION_CANDLES = 12  # maximo velas en posicion (12h)
MIN_TRADE = 5.0            # minimo para operar
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
