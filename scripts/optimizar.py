#!/usr/bin/env python3
"""
Optimizador de estrategia MACD Divergence.
Prueba N combinaciones aleatorias de parámetros sobre datos históricos
y guarda el top 3 global en simulations/best.json.

Uso:
    python scripts/optimizar.py                    # 1000 iteraciones
    python scripts/optimizar.py --iter 5000        # 5000 iteraciones
    python scripts/optimizar.py --list             # ver mejores combinaciones
    python scripts/optimizar.py --reset            # borrar historial
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# ── Parámetros fijos (no se optimizan) ──
FIXED = {
    "fast": 12,
    "slow": 26,
    "signal": 9,
    "invest_percent": 95.0,
    "max_position_hours": 12,  # más amplio para 1h
    "min_balance_eur": 5.0,
    "initial_eur": 5000.0,
    "trail_retain": 0.6,  # retiene 60% de ganancia
}

# ── Auto-detect timeframe ──
# If first two candles differ by ~3600 seconds, it's 1h. If ~86400, it's 1d.
def detect_timeframe(ohlcv):
    if len(ohlcv) < 2:
        return "1h"
    diff = (ohlcv[1][0] - ohlcv[0][0]) / 1000
    if abs(diff - 86400) < 1000:
        return "1d"
    return "1h"

# Convert hours to candle units
def max_position_candles(tf: str, hours: int = 12):
    if tf == "1d":
        return max(1, hours // 24)
    return hours  # 1h = 1 candle per hour

# ── Rangos de búsqueda ──
SEARCH_SPACE = {
    "min_histogram_abs": (15, 120, "int"),       # |histograma| mínimo
    "confirm_velas": (1, 4, "int"),               # velas de confirmación
    "sl_percent": (0.5, 5.0, "float"),            # stop-loss %
    "trailing_min_gain": (0.5, 5.0, "float"),     # desde qué % activa trailing
    "fee_percent": (0.0, 0.5, "float"),           # comisión % (0 = best case)
}

BEST_FILE = Path(__file__).parent.parent / "simulations/best.json"
DATA_FILE = Path(__file__).parent.parent / "historical_1h.csv"
SIMULATIONS_DIR = Path(__file__).parent.parent / "simulations"


# ── Utility functions ──

def ema(data: np.ndarray, period: int) -> np.ndarray:
    result = np.empty_like(data)
    result[:] = np.nan
    first_valid = np.where(~np.isnan(data))[0]
    if len(first_valid) == 0:
        return result
    start = first_valid[0] + period - 1
    if start >= len(data):
        return result
    result[start] = np.mean(data[start - period + 1: start + 1])
    mult = 2.0 / (period + 1)
    for i in range(start + 1, len(data)):
        result[i] = (data[i] - result[i - 1]) * mult + result[i - 1]
    return result


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


def precompute_macd(ohlcv: list[list[float]]) -> np.ndarray:
    """Precompute histogram values for ALL candles. Returns array same length as ohlcv."""
    closes = np.array([c[4] for c in ohlcv], dtype=np.float64)
    ema_fast = ema(closes, FIXED["fast"])
    ema_slow = ema(closes, FIXED["slow"])
    macd_line = ema_fast - ema_slow
    sig_line = ema(macd_line, FIXED["signal"])
    return macd_line - sig_line


# ── Simulador rápido (solo métricas clave) ──

def simulate(ohlcv: list[list[float]], hist: np.ndarray, config: dict) -> dict:
    """
    Simulación rápida. Devuelve dict con métricas clave.
    No guarda trades individuales, solo stats agregadas.
    """
    cfg = {**FIXED, **config}
    fee_rate = cfg["fee_percent"] / 100.0
    sl_rate = cfg["sl_percent"] / 100.0
    tf = detect_timeframe(ohlcv)
    max_candles = max_position_candles(tf, cfg["max_position_hours"])

    eur = cfg["initial_eur"]
    btc = 0.0
    in_pos = False
    entry_price = 0.0
    entry_idx = 0
    highest = 0.0
    stop_loss = None
    awaiting = None
    confirm = 0
    buf = []

    trades = 0
    wins = 0
    losses = 0
    total_pnl = 0.0
    total_fees = 0.0

    for i in range(len(ohlcv)):
        h_val = float(hist[i])
        if np.isnan(h_val):
            continue

        close = float(ohlcv[i][4])
        buf.append(h_val)
        if len(buf) > 5:
            buf = buf[-5:]

        # ── Position management ──
        if in_pos:
            # Stop-loss
            if stop_loss is not None and close <= stop_loss:
                pnl = (close - entry_price) * btc
                fee = btc * close * fee_rate
                total_fees += fee
                total_pnl += pnl - fee
                eur += btc * close - fee
                btc = 0.0
                in_pos = False
                trades += 1
                if pnl - fee > 0: wins += 1
                else: losses += 1
                continue

            # Trailing
            if close > highest:
                highest = close
                gain = (close - entry_price) / entry_price * 100
                if gain > cfg["trailing_min_gain"]:
                    trail_pct = gain * FIXED["trail_retain"]
                    new_sl = entry_price * (1 + trail_pct / 100.0)
                    if stop_loss is None or new_sl > stop_loss:
                        stop_loss = new_sl
                        # Check if trailing triggered sell immediately
                        if close <= stop_loss:
                            pnl = (close - entry_price) * btc
                            fee = btc * close * fee_rate
                            total_fees += fee
                            total_pnl += pnl - fee
                            eur += btc * close - fee
                            btc = 0.0
                            in_pos = False
                            trades += 1
                            if pnl - fee > 0: wins += 1
                            else: losses += 1
                            continue

            # Max time
            hours = (i - entry_idx) / 3600.0 if tf == "1h" else (i - entry_idx)
            if hours >= max_candles:
                pnl = (close - entry_price) * btc
                fee = btc * close * fee_rate
                total_fees += fee
                total_pnl += pnl - fee
                eur += btc * close - fee
                btc = 0.0
                in_pos = False
                trades += 1
                if pnl - fee > 0: wins += 1
                else: losses += 1
                continue

            # Sell signal
            if awaiting == "sell" and len(buf) >= 4:
                confirm += 1
                if confirm > cfg["confirm_velas"]:
                    pnl = (close - entry_price) * btc
                    fee = btc * close * fee_rate
                    total_fees += fee
                    total_pnl += pnl - fee
                    eur += btc * close - fee
                    btc = 0.0
                    in_pos = False
                    trades += 1
                    if pnl - fee > 0: wins += 1
                    else: losses += 1
                    awaiting = None
                    continue

        # ── Signal detection ──
        if len(buf) >= 4:
            b = buf
            if not in_pos:
                # Valley (buy)
                if (b[-4] >= b[-3] > b[-2] < b[-1] and b[-2] < 0 and b[-1] < 0
                        and abs(b[-2]) >= cfg["min_histogram_abs"]):
                    awaiting = "buy"
                    confirm = 0
            if in_pos:
                # Peak (sell)
                if (b[-4] <= b[-3] < b[-2] > b[-1] and b[-2] > 0 and b[-1] > 0
                        and abs(b[-2]) >= cfg["min_histogram_abs"]):
                    awaiting = "sell"
                    confirm = 0

        # ── Execute buy ──
        if awaiting == "buy" and not in_pos:
            confirm += 1
            if confirm > cfg["confirm_velas"]:
                invest = eur * FIXED["invest_percent"] / 100.0
                if invest >= cfg["min_balance_eur"]:
                    buy_fee = invest * fee_rate
                    total_fees += buy_fee
                    btc = invest / close
                    eur -= invest + buy_fee
                    entry_price = close
                    entry_idx = i
                    highest = close
                    stop_loss = entry_price * (1 - sl_rate)
                    in_pos = True
                awaiting = None

    # Close any remaining position at last price
    if in_pos and len(ohlcv) > 0:
        last_close = float(ohlcv[-1][4])
        pnl = (last_close - entry_price) * btc
        fee = btc * last_close * fee_rate
        total_fees += fee
        total_pnl += pnl - fee
        eur += btc * last_close - fee
        trades += 1
        if pnl - fee > 0: wins += 1
        else: losses += 1

    final_balance = eur
    roi = ((final_balance - FIXED["initial_eur"]) / FIXED["initial_eur"]) * 100
    win_rate = (wins / trades * 100) if trades > 0 else 0
    profit_factor = abs(sum([1]) / 1)  # placeholder

    # Sharpe-like ratio: avg_pnl / std_pnl (simplified)
    avg_win = total_pnl / trades if trades > 0 else 0

    return {
        "trades": trades,
        "wins": wins,
        "losses": losses,
        "win_rate": round(win_rate, 1),
        "total_pnl": round(total_pnl, 2),
        "total_fees": round(total_fees, 2),
        "final_balance": round(final_balance, 2),
        "roi": round(roi, 2),
        "avg_pnl_per_trade": round(avg_win, 2),
        "score": round(roi - abs(total_fees) * 0.1, 2),  # score = ROI penalizado por comisiones
    }


def random_config() -> dict:
    """Generate a random configuration within search space."""
    cfg = {}
    for key, (lo, hi, typ) in SEARCH_SPACE.items():
        if typ == "int":
            cfg[key] = random.randint(lo, hi)
        else:
            cfg[key] = round(random.uniform(lo, hi), 2)
    return cfg


def config_to_str(cfg: dict) -> str:
    return (f"|hist|≥{cfg['min_histogram_abs']}  "
            f"conf={cfg['confirm_velas']}  "
            f"SL={cfg['sl_percent']}%  "
            f"trail={cfg['trailing_min_gain']}%  "
            f"fee={cfg['fee_percent']}%")


def load_best() -> list[dict]:
    if BEST_FILE.exists():
        with open(BEST_FILE) as f:
            return json.load(f)
    return []


def save_best(best: list[dict]):
    SIMULATIONS_DIR.mkdir(exist_ok=True)
    with open(BEST_FILE, "w") as f:
        json.dump(best, f, indent=2)


def merge_best(existing: list[dict], new_entries: list[dict], top_n: int = 10):
    """Merge new results into existing best list, keep top N by score."""
    combined = existing + new_entries
    # Deduplicate by config fingerprint
    seen = set()
    unique = []
    for entry in combined:
        # Create fingerprint from config values
        cfg = entry["config"]
        fp = tuple(sorted(cfg.items()))
        if fp not in seen:
            seen.add(fp)
            unique.append(entry)
    unique.sort(key=lambda x: x["score"], reverse=True)
    return unique[:top_n]


def main():
    parser = argparse.ArgumentParser(description="Optimizador MACD Divergence")
    parser.add_argument("--iter", type=int, default=1000, help="Número de iteraciones")
    parser.add_argument("--list", action="store_true", help="Ver mejores combinaciones")
    parser.add_argument("--reset", action="store_true", help="Borrar historial")
    parser.add_argument("--save-every", type=int, default=100, help="Guardar cada N iteraciones")
    args = parser.parse_args()

    if args.list:
        best = load_best()
        if not best:
            print("\n📭 No hay combinaciones guardadas. Ejecuta el optimizador primero.\n")
            return
        print(f"\n{'='*60}")
        print(f"  🏆 TOP {len(best)} MEJORES COMBINACIONES")
        print(f"{'='*60}\n")
        for idx, entry in enumerate(best, 1):
            cfg = entry["config"]
            s = entry["summary"]
            print(f"  #{idx} — Score: {entry['score']}")
            print(f"     {config_to_str(cfg)}")
            print(f"     Trades: {s['trades']}  WR: {s['win_rate']}%  "
                  f"PnL: {s['total_pnl']:+.2f}€  ROI: {s['roi']:+.2f}%  "
                  f"Comisiones: {s['total_fees']:.2f}€")
            print()
        return

    if args.reset:
        if BEST_FILE.exists():
            BEST_FILE.unlink()
            print("🗑️  Historial borrado.")
        return

    # ── Load data ──
    if not DATA_FILE.exists():
        print(f"❌ No se encuentra {DATA_FILE}. Ejecuta primero la descarga de velas.")
        print("   python -c \"import asyncio; from scripts.simular import *; asyncio.run(download_ohlcv())\"")
        sys.exit(1)

    print(f"📥 Cargando datos desde {DATA_FILE} ...")
    ohlcv = load_ohlcv(DATA_FILE)
    print(f"   {len(ohlcv)} velas cargadas")
    print(f"   Desde: {datetime.fromtimestamp(ohlcv[0][0]/1000, tz=timezone.utc).isoformat()}")
    print(f"   Hasta: {datetime.fromtimestamp(ohlcv[-1][0]/1000, tz=timezone.utc).isoformat()}")

    # ── Precompute MACD once ──
    print("🧮 Precomputando MACD ...")
    hist = precompute_macd(ohlcv)
    valid_count = int(np.sum(~np.isnan(hist)))
    print(f"   {valid_count} valores de histograma válidos")
    print(f"   Rango: {float(np.nanmin(hist)):.1f} a {float(np.nanmax(hist)):.1f}")

    # ── Load existing best ──
    best = load_best()
    print(f"\n🏆 Mejores registradas actualmente: {len(best)}")

    # ── Run optimization ──
    print(f"\n🚀 Ejecutando {args.iter} iteraciones ...")
    print(f"{'─'*60}")
    print(f"{'#':<6} {'Config':<45} {'Trades':<7} {'WR':<6} {'PnL':<10} {'ROI':<8} {'Score':<8}")
    print(f"{'─'*60}")

    new_results = []
    best_score_so_far = max([b["score"] for b in best]) if best else float("-inf")

    for iteration in range(1, args.iter + 1):
        config = random_config()
        result = simulate(ohlcv, hist, config)
        score = result["score"]

        entry = {
            "config": config,
            "summary": result,
            "score": score,
            "timestamp": datetime.now().isoformat(),
        }
        new_results.append(entry)

        # Progress every 100 or when new best
        if score > best_score_so_far or iteration % args.save_every == 0 or iteration == 1:
            cfg_str = config_to_str(config)
            print(f"{iteration:<6} {cfg_str:<45} "
                  f"{result['trades']:<7} {result['win_rate']:<6} "
                  f"{result['total_pnl']:<+10} {result['roi']:<+8} {score:<+8}")
            if score > best_score_so_far:
                best_score_so_far = score
                print(f"        ⭐ NUEVO MEJOR SCORE: {score}")

        # Save periodically
        if iteration % args.save_every == 0 or iteration == args.iter:
            best = merge_best(best, new_results, top_n=10)
            save_best(best)
            new_results = []

    # ── Final summary ──
    best = merge_best(best, new_results, top_n=10)
    save_best(best)

    print(f"\n{'='*60}")
    print(f"  🏆 OPTIMIZACIÓN COMPLETADA — TOP 10")
    print(f"{'='*60}\n")
    for idx, entry in enumerate(best, 1):
        cfg = entry["config"]
        s = entry["summary"]
        print(f"  #{idx} — Score: {entry['score']}")
        print(f"     {config_to_str(cfg)}")
        print(f"     Trades: {s['trades']}  WR: {s['win_rate']}%  "
              f"PnL: {s['total_pnl']:+.2f}€  ROI: {s['roi']:+.2f}%  "
              f"Comisiones: {s['total_fees']:.2f}€  "
              f"Avg/Trade: {s['avg_pnl_per_trade']:+.2f}€")
        print()

    print(f"📁 Guardado en: {BEST_FILE}")
    print(f"💡 Para ver los resultados más tarde: python scripts/optimizar.py --list\n")


if __name__ == "__main__":
    main()
