#!/usr/bin/env python3
"""
Prueba estrategias NUEVAS sobre las 721 velas 1h.

Busca la mejor relación: PnL/operación alto + buen win rate + buen ROI.

Estrategias a probar:
  1. MOMENTUM:     Compra cuando precio rompe EMA20 al alza con volumen alto
  2. REVERSIÓN:    Compra cuando precio toca banda inferior de Bollinger + RSI < 30
  3. DOBLE SUELO:  Compra cuando precio hace 2 mínimos consecutivos + volumen decreciente
  4. BREAKOUT:     Compra cuando precio supera máximo de N velas anteriores
  5. MACD+VOLUMEN: Compra en valle MACD solo si el volumen es anormalmente alto
  6. ATR_TRAILING: Compra cuando precio supera EMA20, trailing dinámico por ATR

Uso:
    python scripts/nuevas_estrategias.py --iter 3000
    python scripts/nuevas_estrategias.py --list
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

# ── Constants ──
INITIAL_EUR = 120.0  # Su capital real
INVEST_PCT = 95.0
TRAIL_RETAIN = 0.6
MAX_POSITION_CANDLES = 12
MIN_BALANCE = 5.0
FEE_RATE = 0.002  # 0.20% (OKX market) o 0.004 (Kraken)

DATA_FILE = Path(__file__).parent.parent / "historical_1h.csv"
SIMULATIONS_DIR = Path(__file__).parent.parent / "simulations"


# ── Utilities ──

def ema(data: np.ndarray, period: int) -> np.ndarray:
    result = np.empty_like(data)
    result[:] = np.nan
    first = np.where(~np.isnan(data))[0]
    if len(first) == 0:
        return result
    start = first[0] + period - 1
    if start >= len(data):
        return result
    result[start] = np.mean(data[start - period + 1: start + 1])
    mult = 2.0 / (period + 1)
    for i in range(start + 1, len(data)):
        result[i] = (data[i] - result[i - 1]) * mult + result[i - 1]
    return result


def sma(data: np.ndarray, period: int) -> np.ndarray:
    result = np.empty_like(data)
    result[:] = np.nan
    for i in range(period - 1, len(data)):
        result[i] = np.mean(data[i - period + 1: i + 1])
    return result


def std(data: np.ndarray, period: int) -> np.ndarray:
    result = np.empty_like(data)
    result[:] = np.nan
    for i in range(period - 1, len(data)):
        result[i] = np.std(data[i - period + 1: i + 1])
    return result


def atr(ohlcv: list[list[float]], period: int = 14) -> np.ndarray:
    """Average True Range."""
    result = np.full(len(ohlcv), np.nan)
    for i in range(1, len(ohlcv)):
        high, low, prev_close = ohlcv[i][2], ohlcv[i][3], ohlcv[i - 1][4]
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        result[i] = tr
    # Smooth with EMA
    return ema(result, period)


def load_ohlcv(csv_path: str | Path) -> list[list[float]]:
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append([
                int(r["timestamp"]),
                float(r["open"]), float(r["high"]),
                float(r["low"]), float(r["close"]),
                float(r["volume"]),
            ])
    return rows


def rsi_from_closes(closes: np.ndarray, period: int = 14) -> np.ndarray:
    deltas = np.diff(closes)
    rsi = np.full_like(closes, np.nan)
    if len(deltas) < period:
        return rsi
    seed = deltas[:period]
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period
    rs = up / down if down != 0 else 0
    rsi[period] = 100 - 100 / (1 + rs)
    for i in range(period + 1, len(closes)):
        d = deltas[i - 1]
        upval = d if d > 0 else 0
        downval = -d if d < 0 else 0
        up = (up * (period - 1) + upval) / period
        down = (down * (period - 1) + downval) / period
        rs = up / down if down != 0 else 0
        rsi[i] = 100 - 100 / (1 + rs)
    return rsi


# ── Signal generators ──

# 1. MOMENTUM: Compra cuando precio rompe EMA20 al alza con volumen > media
def gen_momentum(closes, indicators, i, cfg):
    ema20 = indicators.get("ema20", [])
    vol_avg = indicators.get("vol_avg", [])
    vol = indicators.get("volumes", [])
    if i < 1 or i >= len(ema20) or np.isnan(ema20[i]) or i >= len(vol) or i >= len(vol_avg):
        return None
    if indicators.get("in_pos", False):
        return None

    lookback = cfg.get("lookback", 5)
    vol_mult = cfg.get("vol_mult", 1.5)
    prev_close = float(closes[i - 1]) if i > 0 else 0
    curr_close = float(closes[i])

    # Buy: precio cruza EMA20 al alza, y volumen > media * multiplicador
    if prev_close <= ema20[i] and curr_close > ema20[i]:
        if not np.isnan(vol_avg[i]) and vol_avg[i] > 0 and vol[i] > vol_avg[i] * vol_mult:
            return {"action": "buy", "confidence": min((curr_close - ema20[i]) / ema20[i] * 100, 1.0)}
    return None


# 2. REVERSIÓN: Precio toca banda inferior Bollinger + RSI < 30
def gen_reversione(closes, indicators, i, cfg):
    if indicators.get("in_pos", False):
        return None
    bb = indicators.get("bb")
    rsi_arr = indicators.get("rsi_arr", [])
    if bb is None or i >= len(bb[0]) or np.isnan(bb[0][i]):
        return None
    _, upper, lower = bb
    close = float(closes[i])
    rsi_val = float(rsi_arr[i]) if i < len(rsi_arr) and not np.isnan(rsi_arr[i]) else 999
    rsi_max = cfg.get("rsi_max", 35)

    if close <= lower[i] and rsi_val < rsi_max:
        confidence = min((lower[i] - close) / lower[i] * 100, 1.0)
        return {"action": "buy", "confidence": max(confidence, 0.3)}
    return None


# 3. DOBLE SUELO: Dos mínimos consecutivos, volumen bajando
def gen_doble_suelo(closes, indicators, i, cfg):
    if indicators.get("in_pos", False):
        return None
    if i < 8:
        return None

    lows = indicators.get("lows", [])
    vol = indicators.get("volumes", [])
    if len(lows) <= i or len(vol) <= i:
        return None

    lookback = cfg.get("lookback", 3)
    vol_pct = cfg.get("vol_decrease_pct", 0.8)  # volumen debe bajar al 80% o menos

    # Detectar 2 valles (mínimos locales) con precio similar
    l4, l3, l2, l1, l0 = lows[i - 4], lows[i - 3], lows[i - 2], lows[i - 1], lows[i]

    # Patrón: valle1 (l2), subida, valle2 (l0) similar o superior
    if l2 < l1 and l2 < l3 and l0 < l1 and l2 > 0 and l0 > 0:
        # Los dos valles deben estar cerca (diferencia < X%)
        diff_pct = abs(l0 - l2) / max(l2, 1) * 100
        if diff_pct < cfg.get("max_valley_diff", 3.0):
            # Volumen debe ser menor en el segundo valle
            vol_avg = np.mean(vol[max(0, i - lookback):i + 1])
            vol_avg2 = np.mean(vol[max(0, i - lookback - 3):i - 2])
            if vol_avg2 > 0 and vol_avg <= vol_avg2 * vol_pct:
                return {"action": "buy", "confidence": min(diff_pct / 5 + 0.3, 1.0)}
    return None


# 4. BREAKOUT: Precio supera máximo de N velas
def gen_breakout(closes, indicators, i, cfg):
    if indicators.get("in_pos", False):
        return None
    if i < cfg.get("lookback", 20):
        return None

    lookback = cfg.get("lookback", 20)
    vol_threshold = cfg.get("vol_threshold", 1.5)
    vol = indicators.get("volumes", [])
    highs = indicators.get("highs", [])
    if len(highs) <= i or len(vol) <= i:
        return None

    current_high = float(highs[i])
    current_vol = float(vol[i])
    max_prev = max(highs[i - lookback:i])
    avg_vol = np.mean(vol[i - lookback:i]) if i >= lookback else 0

    if current_high > max_prev and avg_vol > 0 and current_vol > avg_vol * vol_threshold:
        return {"action": "buy", "confidence": min((current_high - max_prev) / max_prev * 50, 1.0)}
    return None


# 5. MACD + VOLUMEN: Valle MACD solo si volumen es anormalmente alto
def gen_macd_volumen(closes, indicators, i, cfg):
    if indicators.get("in_pos", False):
        return None
    hist_buf = indicators.get("hist_buf", [])
    vol = indicators.get("volumes", [])
    if len(hist_buf) < 4 or len(vol) <= i:
        return None
    b = hist_buf
    min_h = cfg.get("min_histogram_abs", 60)
    vol_mult = cfg.get("vol_mult", 1.5)

    # Valle MACD
    if not (b[-4] >= b[-3] > b[-2] < b[-1] and b[-2] < 0 and b[-1] < 0 and abs(b[-2]) >= min_h):
        return None

    # Volumen: comparar con media de últimas 24h
    lookback = 24
    if i < lookback:
        return None
    avg_vol = np.mean(vol[i - lookback:i])
    if avg_vol > 0 and vol[i] > avg_vol * vol_mult:
        return {"action": "buy", "confidence": min(abs(b[-2]) / 100 + 0.2, 1.0)}
    return None


# 6. ATR_TRAILING: Compra cuando precio supera EMA20, trailing dinámico con ATR
def gen_atr_trailing(closes, indicators, i, cfg):
    if indicators.get("in_pos", False):
        return None
    ema20 = indicators.get("ema20", [])
    atr_vals = indicators.get("atr_arr", [])
    if i < 1 or i >= len(ema20) or np.isnan(ema20[i]) or i >= len(atr_vals) or np.isnan(atr_vals[i]):
        return None

    prev_close = float(closes[i - 1]) if i > 0 else 0
    curr_close = float(closes[i]) if i < len(closes) else 0

    # Buy: precio cruza EMA20 + spread mínimo por ATR
    atr_mult = cfg.get("atr_mult", 0.5)
    if prev_close <= ema20[i] and curr_close > ema20[i]:
        return {"action": "buy", "confidence": min(atr_vals[i] / curr_close * 100, 1.0)}
    return None


# ── Strategy registry ──
STRATEGY_META = {
    "momentum": {
        "name": "MOMENTUM (EMA20+Vol)",
        "gen": gen_momentum,
        "params": {"lookback": (3, 10, "int"), "vol_mult": (1.2, 3.0, "float"),
                   "sl_percent": (1.0, 5.0, "float"), "trailing_min_gain": (0.5, 3.0, "float"),
                   "fee_percent": (0.0, 0.26, "float")}},
    "reversion": {
        "name": "REVERSIÓN (BB+RSI)",
        "gen": gen_reversione,
        "params": {"rsi_max": (20, 40, "int"),
                   "sl_percent": (1.0, 5.0, "float"), "trailing_min_gain": (0.5, 3.0, "float"),
                   "fee_percent": (0.0, 0.26, "float")}},
    "doble_suelo": {
        "name": "DOBLE SUELO",
        "gen": gen_doble_suelo,
        "params": {"lookback": (2, 5, "int"), "vol_decrease_pct": (0.5, 0.9, "float"),
                   "max_valley_diff": (1.0, 5.0, "float"),
                   "sl_percent": (1.0, 5.0, "float"), "trailing_min_gain": (0.5, 3.0, "float"),
                   "fee_percent": (0.0, 0.26, "float")}},
    "breakout": {
        "name": "BREAKOUT",
        "gen": gen_breakout,
        "params": {"lookback": (10, 40, "int"), "vol_threshold": (1.2, 3.0, "float"),
                   "sl_percent": (1.0, 5.0, "float"), "trailing_min_gain": (0.5, 3.0, "float"),
                   "fee_percent": (0.0, 0.26, "float")}},
    "macd_vol": {
        "name": "MACD+VOLUMEN",
        "gen": gen_macd_volumen,
        "params": {"min_histogram_abs": (30, 100, "int"), "vol_mult": (1.2, 3.0, "float"),
                   "sl_percent": (1.0, 5.0, "float"), "trailing_min_gain": (0.5, 3.0, "float"),
                   "fee_percent": (0.0, 0.26, "float")}},
    "atr_trailing": {
        "name": "ATR TRAILING",
        "gen": gen_atr_trailing,
        "params": {"atr_mult": (0.3, 2.0, "float"),
                   "sl_percent": (1.0, 5.0, "float"), "trailing_min_gain": (0.5, 3.0, "float"),
                   "fee_percent": (0.0, 0.26, "float")}},
}


# ── Precompute helpers ──

def precompute_all(ohlcv, closes):
    vol = np.array([c[5] for c in ohlcv], dtype=np.float64)
    highs = np.array([c[2] for c in ohlcv], dtype=np.float64)
    lows = np.array([c[3] for c in ohlcv], dtype=np.float64)
    ema20 = ema(closes, 20)
    vol_sma20 = sma(vol, 20)
    bb = sma(closes, 20), sma(closes, 20) + std(closes, 20) * 2.0, sma(closes, 20) - std(closes, 20) * 2.0
    rsi_arr = rsi_from_closes(closes)
    atr_arr = atr(ohlcv)

    # MACD histogram buffer
    def pre_macd():
        ema_f = ema(closes, 12)
        ema_s = ema(closes, 26)
        macd = ema_f - ema_s
        sig = ema(macd, 9)
        hist = macd - sig
        bufs = []
        b = []
        for v in hist:
            if np.isnan(v):
                b = []
            else:
                b.append(float(v))
                if len(b) > 5:
                    b = b[-5:]
            bufs.append(list(b))
        return bufs

    hist_bufs = pre_macd()

    result = []
    for i in range(len(ohlcv)):
        result.append({
            "ema20": ema20,
            "vol_avg": vol_sma20,
            "volumes": vol,
            "highs": highs,
            "lows": lows,
            "bb": bb,
            "rsi_arr": rsi_arr,
            "atr_arr": atr_arr,
            "hist_buf": hist_bufs[i] if i < len(hist_bufs) else [],
        })
    return result


# ── Simulator ──

def simulate(ohlcv, closes, precomputed, gen_fn, config):
    fee_rate = config.get("fee_percent", 0.0) / 100.0
    sl_rate = config.get("sl_percent", 2.0) / 100.0
    trail_min = config.get("trailing_min_gain", 1.0)

    eur = INITIAL_EUR
    btc = 0.0
    in_pos = False
    entry_p = 0.0
    entry_i = 0
    highest = 0.0
    stop_loss = None

    trades = 0
    wins = 0
    losses = 0
    total_pnl = 0.0
    total_fees = 0.0
    pnls_per_trade = []

    for i in range(len(ohlcv)):
        close = float(closes[i])
        ind = {
            **precomputed[i],
            "in_pos": in_pos,
        }

        if in_pos:
            # SL
            if stop_loss is not None and close <= stop_loss:
                pnl = (close - entry_p) * btc
                fee = btc * close * fee_rate
                total_fees += fee; total_pnl += pnl - fee
                pnls_per_trade.append(pnl - fee)
                eur += btc * close - fee
                btc = 0.0; in_pos = False; trades += 1
                if pnl - fee > 0: wins += 1
                else: losses += 1
                continue

            # Trailing
            if close > highest:
                highest = close
                gain = (close - entry_p) / entry_p * 100
                if gain > trail_min:
                    trail_pct = gain * TRAIL_RETAIN
                    new_sl = entry_p * (1 + trail_pct / 100.0)
                    if stop_loss is None or new_sl > stop_loss:
                        stop_loss = new_sl

            # Max time
            if i - entry_i >= MAX_POSITION_CANDLES:
                pnl = (close - entry_p) * btc
                fee = btc * close * fee_rate
                total_fees += fee; total_pnl += pnl - fee
                pnls_per_trade.append(pnl - fee)
                eur += btc * close - fee
                btc = 0.0; in_pos = False; trades += 1
                if pnl - fee > 0: wins += 1
                else: losses += 1
                continue

        # Signal
        sig = gen_fn(closes, ind, i, config)
        if sig and sig["action"] == "buy" and not in_pos:
            invest = eur * INVEST_PCT / 100.0
            if invest >= MIN_BALANCE:
                buy_fee = invest * fee_rate
                total_fees += buy_fee
                btc = invest / close
                eur -= invest + buy_fee
                entry_p = close
                entry_i = i
                highest = close
                stop_loss = entry_p * (1 - sl_rate)
                in_pos = True

    # Close remaining
    if in_pos and len(ohlcv) > 0:
        last_c = float(closes[-1])
        pnl = (last_c - entry_p) * btc
        fee = btc * last_c * fee_rate
        total_fees += fee; total_pnl += pnl - fee
        pnls_per_trade.append(pnl - fee)
        eur += btc * last_c - fee
        trades += 1
        if pnl - fee > 0: wins += 1
        else: losses += 1

    roi = ((eur - INITIAL_EUR) / INITIAL_EUR) * 100
    wr = (wins / trades * 100) if trades > 0 else 0
    avg_pnl = total_pnl / trades if trades > 0 else 0
    max_drawdown = 0.0
    peak = INITIAL_EUR
    bal = INITIAL_EUR
    for p in pnls_per_trade:
        bal += p
        if bal > peak:
            peak = bal
        dd = (peak - bal) / peak * 100
        if dd > max_drawdown:
            max_drawdown = dd

    # Score: pondera ROI, PnL/trade, WinRate, penaliza drawdown
    score = roi * 0.4 + avg_pnl * 1.0 + wr * 0.1 - max_drawdown * 0.3

    return {
        "trades": trades, "wins": wins, "losses": losses,
        "win_rate": round(wr, 1), "total_pnl": round(total_pnl, 2),
        "total_fees": round(total_fees, 2),
        "final_balance": round(eur, 2), "roi": round(roi, 2),
        "avg_pnl": round(avg_pnl, 2),
        "max_dd": round(max_drawdown, 2),
        "max_pnl_trade": round(max(pnls_per_trade), 2) if pnls_per_trade else 0,
        "score": round(score, 2),
    }


def random_cfg(params):
    cfg = {}
    for key, (lo, hi, typ) in params.items():
        if typ == "int":
            cfg[key] = random.randint(lo, hi)
        else:
            cfg[key] = round(random.uniform(lo, hi), 2)
    return cfg


def cfg_str(sid, cfg):
    parts = [sid]
    for k, v in sorted(cfg.items()):
        if k in ("fee_percent", "sl_percent", "trailing_min_gain"):
            continue
        parts.append(f"{k}={v}")
    parts.append(f"SL={cfg.get('sl_percent','?')}%")
    parts.append(f"trail={cfg.get('trailing_min_gain','?')}%")
    parts.append(f"fee={cfg.get('fee_percent',0)}%")
    return "  ".join(parts)


def run_strategy(sid, meta, ohlcv, closes, precomputed, iters, save_file):
    print(f"\n{'='*55}")
    print(f"  📊 {meta['name']}")
    print(f"{'='*55}")

    best = []
    best_score = float("-inf")
    print(f"   {iters} iteraciones ...")

    for it in range(1, iters + 1):
        cfg = random_cfg(meta["params"])
        res = simulate(ohlcv, closes, precomputed, meta["gen"], cfg)
        score = res["score"]
        entry = {"config": cfg, "summary": res, "score": score}
        best.append(entry)
        best.sort(key=lambda x: x["score"], reverse=True)
        best = best[:10]

        if score > best_score or it == 1 or it % 1000 == 0:
            best_score = max(best_score, score)
            cs = cfg_str(sid, cfg)
            print(f"   {it:<5} {cs:<50} {res['trades']:<5} {res['win_rate']:<5} "
                  f"{res['total_pnl']:<+9} {res['roi']:<+7} {res['avg_pnl']:<+9} {score:<+8}")
            if score >= best_score and it > 1:
                print(f"          ⭐")

    SIMULATIONS_DIR.mkdir(exist_ok=True)
    with open(save_file, "w") as f:
        json.dump({"strategy": meta["name"], "strategy_id": sid,
                   "iterations": iters, "results": best}, f, indent=2)

    print(f"\n   🏆 TOP 3 (con 120€):")
    for idx, e in enumerate(best[:3], 1):
        c, s = e["config"], e["summary"]
        print(f"      #{idx} — Score: {e['score']}")
        print(f"         {cfg_str(sid, c)}")
        print(f"         Trades: {s['trades']}  WR: {s['win_rate']}%  "
              f"PnL: {s['total_pnl']:+.2f}€  ROI: {s['roi']:+.2f}%  "
              f"Avg/Trade: {s['avg_pnl']:+.2f}€  MaxDD: {s['max_dd']:.2f}%")
    return best[:3]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter", type=int, default=3000)
    parser.add_argument("--strategies", type=str, default="all")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        for f in sorted(SIMULATIONS_DIR.glob("best_nuevas_*.json")):
            d = json.load(open(f))
            print(f"\n  {d['strategy']} ({f.name})")
            for e in d["results"][:3]:
                c, s = e["config"], e["summary"]
                print(f"    Score {e['score']}: WR={s['win_rate']}% "
                      f"PnL={s['total_pnl']:+.2f}€ ROI={s['roi']:+.2f}% "
                      f"Avg={s['avg_pnl']:+.2f}€ Trades={s['trades']} DD={s['max_dd']:.1f}%")
        return

    ohlcv = load_ohlcv(DATA_FILE)
    closes = np.array([c[4] for c in ohlcv], dtype=np.float64)
    print(f"\n📥 {len(ohlcv)} velas 1h: {closes[0]:.0f}€ → {closes[-1]:.0f}€")
    print(f"💶 Capital simulado: {INITIAL_EUR}€")

    # Precompute all indicators once
    print("⚙️  Precomputando indicadores...")
    precomputed = precompute_all(ohlcv, closes)
    print(f"   ✓ {len(precomputed)} ticks precomputados")

    if args.strategies == "all":
        selected = list(STRATEGY_META.items())
    else:
        ids = [s.strip() for s in args.strategies.split(",")]
        selected = [(s, STRATEGY_META[s]) for s in ids]

    tops = {}
    for sid, meta in selected:
        sf = SIMULATIONS_DIR / f"best_nuevas_{sid}.json"
        tops[sid] = run_strategy(sid, meta, ohlcv, closes, precomputed, args.iter, sf)

    print(f"\n{'='*65}")
    print(f"  🏆 COMPARATIVA FINAL — NUEVAS ESTRATEGIAS")
    print(f"{'='*65}")
    print(f"  {'Estrategia':<18} {'Score':<8} {'ROI':<10} {'WR':<7} {'Ops':<5} "
          f"{'Avg€':<8} {'PnL':<10} {'DD%':<7}")
    print(f"  {'─'*73}")
    for sid, ts in tops.items():
        if ts:
            s = ts[0]["summary"]
            print(f"  {STRATEGY_META[sid]['name']:<18} {ts[0]['score']:<+8} {s['roi']:<+10} "
                  f"{s['win_rate']:<7} {s['trades']:<5} {s['avg_pnl']:<+8} "
                  f"{s['total_pnl']:<+10} {s['max_dd']:<7}")
    print()


if __name__ == "__main__":
    main()
