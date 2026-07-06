#!/usr/bin/env python3
"""
Multi-estrategia optimizer — prueba MACD, RSI, MACD+RSI, Bollinger, VWAP.

Uso:
    python scripts/multi_estrategia.py --iter 3000
    python scripts/multi_estrategia.py --strategies macd,rsi --iter 1000
    python scripts/multi_estrategia.py --list
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
INITIAL_EUR = 5000.0
INVEST_PCT = 95.0
TRAIL_RETAIN = 0.6
MAX_POSITION_CANDLES = 12
MIN_BALANCE = 5.0

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


# ── Indicator precomputations ──

def precompute_macd(closes: np.ndarray, fast=12, slow=26, signal=9):
    ema_f = ema(closes, fast)
    ema_s = ema(closes, slow)
    macd = ema_f - ema_s
    sig = ema(macd, signal)
    hist_arr = macd - sig
    buf = []
    hist_vals = []
    for v in hist_arr:
        if np.isnan(v):
            buf = []
        else:
            buf.append(float(v))
            if len(buf) > 5:
                buf = buf[-5:]
        hist_vals.append(list(buf))
    return hist_vals


def precompute_rsi(closes: np.ndarray, period=14):
    deltas = np.diff(closes)
    seed = deltas[:period + 1]
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period
    rs = up / down if down != 0 else 0
    rsi = np.full_like(closes, np.nan)
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


def precompute_bb(closes: np.ndarray, period=20, std_mult=2.0):
    mid = sma(closes, period)
    sd = std(closes, period)
    return mid, mid + sd * std_mult, mid - sd * std_mult


def precompute_vwap(ohlcv: list[list[float]]):
    vwap = np.full(len(ohlcv), np.nan)
    cum_pv = 0.0
    cum_v = 0.0
    for i, c in enumerate(ohlcv):
        typical = (c[2] + c[3] + c[4]) / 3.0
        cum_pv += typical * c[5]
        cum_v += c[5]
        vwap[i] = cum_pv / cum_v if cum_v > 0 else c[4]
    return vwap


# ── Signal generators ──

def gen_macd(closes, indicators, i, cfg):
    buf = indicators.get("hist_buf", [])
    if len(buf) < 4:
        return None
    b = buf[-4:]
    min_h = cfg.get("min_histogram_abs", 50)

    if not indicators.get("in_pos", False):
        if (b[-4] >= b[-3] > b[-2] < b[-1] and b[-2] < 0 and b[-1] < 0
                and abs(b[-2]) >= min_h):
            return {"action": "buy", "confidence": min(abs(b[-2]) / 100, 1.0)}
    else:
        if (b[-4] <= b[-3] < b[-2] > b[-1] and b[-2] > 0 and b[-1] > 0
                and abs(b[-2]) >= min_h):
            return {"action": "sell", "confidence": min(abs(b[-2]) / 100, 1.0)}
    return None


def gen_rsi(closes, indicators, i, cfg):
    rsi = indicators.get("rsi_arr", [])
    if i >= len(rsi) or np.isnan(rsi[i]):
        return None
    r = float(rsi[i])
    buy = cfg.get("rsi_buy", 30)
    sell = cfg.get("rsi_sell", 70)

    if not indicators.get("in_pos", False) and r < buy:
        return {"action": "buy", "confidence": max(0.3, (buy - r) / buy)}
    if indicators.get("in_pos", False) and r > sell:
        return {"action": "sell", "confidence": max(0.3, (r - sell) / (100 - sell))}
    return None


def gen_macd_rsi(closes, indicators, i, cfg):
    macd_sig = gen_macd(closes, indicators, i, cfg)
    rsi_sig = gen_rsi(closes, indicators, i, cfg)

    if macd_sig and rsi_sig and macd_sig["action"] == rsi_sig["action"]:
        return {
            "action": macd_sig["action"],
            "confidence": (macd_sig["confidence"] + rsi_sig["confidence"]) / 2,
        }
    if macd_sig and macd_sig["confidence"] > 0.5:
        return macd_sig
    if rsi_sig and rsi_sig["confidence"] > 0.6:
        return rsi_sig
    return None


def gen_bb(closes, indicators, i, cfg):
    bb = indicators.get("bb")
    if bb is None or i >= len(bb[0]) or np.isnan(bb[0][i]):
        return None
    _, upper, lower = bb
    close = float(closes[i])

    if not indicators.get("in_pos", False) and close <= lower[i]:
        return {"action": "buy", "confidence": max(0.3, (lower[i] - close) / lower[i] * 50)}
    if indicators.get("in_pos", False) and close >= upper[i]:
        return {"action": "sell", "confidence": max(0.3, (close - upper[i]) / upper[i] * 50)}
    return None


def gen_vwap(closes, indicators, i, cfg):
    vwap = indicators.get("vwap_arr")
    if vwap is None or i >= len(vwap) or np.isnan(vwap[i]):
        return None
    close = float(closes[i])
    dev = (close - vwap[i]) / vwap[i] * 100
    buy_th = cfg.get("vwap_buy", -2.0)
    sell_th = cfg.get("vwap_sell", 2.0)

    if not indicators.get("in_pos", False) and dev < buy_th:
        return {"action": "buy", "confidence": min(abs(dev) / 10, 1.0)}
    if indicators.get("in_pos", False) and dev > sell_th:
        return {"action": "sell", "confidence": min(dev / 10, 1.0)}
    return None


def gen_macd_rsi_filtro(closes, indicators, i, cfg):
    """MACD valley para comprar + RSI < umbral (sobreventa) como filtro.
       Vende por MACD peak, igual que la normal."""
    macd_sig = gen_macd(closes, indicators, i, cfg)
    rsi = indicators.get("rsi_arr", [])
    rsi_val = float(rsi[i]) if i < len(rsi) and not np.isnan(rsi[i]) else 999
    rsi_max = cfg.get("rsi_max_buy", 30)

    if macd_sig:
        if macd_sig["action"] == "buy" and rsi_val < rsi_max:
            # MACD valley + RSI en sobreventa → compra
            return macd_sig
        elif macd_sig["action"] == "sell":
            # Vende por MACD peak sin filtro RSI
            return macd_sig
    return None


# ── Strategy registry ──
STRATEGY_META = {
    "macd": {"name": "MACD Divergence", "generator": gen_macd,
             "params": {"min_histogram_abs": (15, 120, "int"),
                        "sl_percent": (0.5, 5.0, "float"),
                        "trailing_min_gain": (0.5, 5.0, "float"),
                        "fee_percent": (0.0, 0.26, "float")}},
    "rsi": {"name": "RSI", "generator": gen_rsi,
            "params": {"rsi_buy": (20, 45, "int"), "rsi_sell": (55, 85, "int"),
                       "sl_percent": (0.5, 5.0, "float"),
                       "trailing_min_gain": (0.5, 5.0, "float"),
                       "fee_percent": (0.0, 0.26, "float")}},
    "macd_rsi": {"name": "MACD+RSI", "generator": gen_macd_rsi,
                 "params": {"min_histogram_abs": (15, 120, "int"),
                            "rsi_buy": (20, 45, "int"), "rsi_sell": (55, 85, "int"),
                            "sl_percent": (0.5, 5.0, "float"),
                            "trailing_min_gain": (0.5, 5.0, "float"),
                            "fee_percent": (0.0, 0.26, "float")}},
    "bb": {"name": "Bollinger Bands", "generator": gen_bb,
           "params": {"bb_lookback": (1, 5, "int"),
                      "sl_percent": (0.5, 5.0, "float"),
                      "trailing_min_gain": (0.5, 5.0, "float"),
                      "fee_percent": (0.0, 0.26, "float")}},
    "vwap": {"name": "VWAP", "generator": gen_vwap,
             "params": {"vwap_buy": (-5.0, -0.5, "float"),
                        "vwap_sell": (0.5, 5.0, "float"),
                        "sl_percent": (0.5, 5.0, "float"),
                        "trailing_min_gain": (0.5, 5.0, "float"),
                        "fee_percent": (0.0, 0.26, "float")}},
    "macd_rsi_filtro": {"name": "MACD + RSI<30 filtro", "generator": gen_macd_rsi_filtro,
                        "params": {"min_histogram_abs": (30, 120, "int"),
                                   "rsi_max_buy": (20, 40, "int"),  # RSI máximo para comprar (sobreventa)
                                   "sl_percent": (0.5, 5.0, "float"),
                                   "trailing_min_gain": (0.5, 5.0, "float"),
                                   "fee_percent": (0.0, 0.26, "float")}},
}


def gen_indicators(sid: str, closes: np.ndarray, ohlcv: list) -> dict:
    if sid == "macd":
        buf = precompute_macd(closes)
        return {"hist_buf": buf[-1] if buf else []}  # will be updated per index
    return {}


# ── Simulator ──

def simulate(ohlcv: list[list[float]], closes: np.ndarray,
             indicators: dict, gen_fn: Callable, config: dict) -> dict:
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

    # Per-index indicators
    hist_buf = precompute_macd(closes)
    rsi_arr = precompute_rsi(closes)
    bb = precompute_bb(closes)
    vwap_arr = precompute_vwap(ohlcv)

    for i in range(len(ohlcv)):
        close = float(closes[i])

        # Build per-index indicators dict
        ind = {
            "in_pos": in_pos,
            "hist_buf": hist_buf[i] if i < len(hist_buf) else [],
            "rsi_arr": rsi_arr,
            "bb": bb,
            "vwap_arr": vwap_arr,
        }

        # ── Position management ──
        if in_pos:
            if stop_loss is not None and close <= stop_loss:
                pnl = (close - entry_p) * btc
                fee = btc * close * fee_rate
                total_fees += fee; total_pnl += pnl - fee
                eur += btc * close - fee
                btc = 0.0; in_pos = False; trades += 1
                if pnl - fee > 0: wins += 1
                else: losses += 1
                continue

            if close > highest:
                highest = close
                gain = (close - entry_p) / entry_p * 100
                if gain > trail_min:
                    trail_pct = gain * TRAIL_RETAIN
                    new_sl = entry_p * (1 + trail_pct / 100.0)
                    if stop_loss is None or new_sl > stop_loss:
                        stop_loss = new_sl

            if i - entry_i >= MAX_POSITION_CANDLES:
                pnl = (close - entry_p) * btc
                fee = btc * close * fee_rate
                total_fees += fee; total_pnl += pnl - fee
                eur += btc * close - fee
                btc = 0.0; in_pos = False; trades += 1
                if pnl - fee > 0: wins += 1
                else: losses += 1
                continue

        # ── Signal generation ──
        signal = gen_fn(closes, ind, i, config)
        if signal:
            action = signal["action"]
            if action == "buy" and not in_pos:
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
            elif action == "sell" and in_pos:
                pnl = (close - entry_p) * btc
                fee = btc * close * fee_rate
                total_fees += fee; total_pnl += pnl - fee
                eur += btc * close - fee
                btc = 0.0; in_pos = False; trades += 1
                if pnl - fee > 0: wins += 1
                else: losses += 1

    # Close remaining
    if in_pos and len(ohlcv) > 0:
        last_c = float(closes[-1])
        pnl = (last_c - entry_p) * btc
        fee = btc * last_c * fee_rate
        total_fees += fee; total_pnl += pnl - fee
        eur += btc * last_c - fee
        trades += 1
        if pnl - fee > 0: wins += 1
        else: losses += 1

    roi = ((eur - INITIAL_EUR) / INITIAL_EUR) * 100
    wr = (wins / trades * 100) if trades > 0 else 0
    return {
        "trades": trades, "wins": wins, "losses": losses,
        "win_rate": round(wr, 1), "total_pnl": round(total_pnl, 2),
        "total_fees": round(total_fees, 2),
        "final_balance": round(eur, 2), "roi": round(roi, 2),
        "avg_pnl": round(total_pnl / trades, 2) if trades > 0 else 0,
        "score": round(roi - abs(total_fees) * 0.1, 2),
    }


def random_cfg(params: dict) -> dict:
    cfg = {}
    for key, (lo, hi, typ) in params.items():
        if typ == "int":
            cfg[key] = random.randint(lo, hi)
        else:
            cfg[key] = round(random.uniform(lo, hi), 2)
    return cfg


def cfg_str(sid: str, cfg: dict) -> str:
    parts = [sid]
    for k, v in sorted(cfg.items()):
        if k in ("fee_percent", "sl_percent", "trailing_min_gain"):
            continue
        parts.append(f"{k}={v}")
    parts.append(f"SL={cfg.get('sl_percent','?')}%")
    parts.append(f"trail={cfg.get('trailing_min_gain','?')}%")
    parts.append(f"fee={cfg.get('fee_percent',0)}%")
    return "  ".join(parts)


def run_strategy(sid: str, meta: dict, ohlcv: list, closes: np.ndarray,
                 iterations: int, save_file: Path) -> list:
    print(f"\n{'='*55}")
    print(f"  📊 {meta['name']}")
    print(f"{'='*55}")

    best = []
    best_score = float("-inf")

    print(f"   {iterations} iteraciones ...")
    for it in range(1, iterations + 1):
        cfg = random_cfg(meta["params"])
        res = simulate(ohlcv, closes, {}, meta["generator"], cfg)
        score = res["score"]

        entry = {"config": cfg, "summary": res, "score": score}
        best.append(entry)
        best.sort(key=lambda x: x["score"], reverse=True)
        best = best[:10]

        if score > best_score or it == 1 or it % 500 == 0:
            best_score = max(best_score, score)
            cs = cfg_str(sid, cfg)
            print(f"   {it:<5} {cs:<55} {res['trades']:<5} {res['win_rate']:<5} "
                  f"{res['total_pnl']:<+9} {res['roi']:<+7} {score:<+7}")
            if score >= best_score and it > 1:
                print(f"          ⭐")

    SIMULATIONS_DIR.mkdir(exist_ok=True)
    with open(save_file, "w") as f:
        json.dump({"strategy": meta["name"], "strategy_id": sid,
                   "iterations": iterations, "results": best}, f, indent=2)

    print(f"\n   🏆 TOP 3:")
    for idx, e in enumerate(best[:3], 1):
        c, s = e["config"], e["summary"]
        print(f"      #{idx} — Score: {e['score']}")
        print(f"         {cfg_str(sid, c)}")
        print(f"         Trades: {s['trades']}  WR: {s['win_rate']}%  "
              f"PnL: {s['total_pnl']:+.2f}€  ROI: {s['roi']:+.2f}%  "
              f"Avg: {s['avg_pnl']:+.2f}€")
    return best[:3]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter", type=int, default=3000)
    parser.add_argument("--strategies", type=str, default="all")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        for f in sorted(SIMULATIONS_DIR.glob("best_*.json")):
            d = json.load(open(f))
            print(f"\n  {d['strategy']} ({f.name})")
            for e in d["results"][:3]:
                c, s = e["config"], e["summary"]
                print(f"    Score {e['score']}: WR={s['win_rate']}% PnL={s['total_pnl']:+.2f}€ "
                      f"ROI={s['roi']:+.2f}% Trades={s['trades']}")
        return

    ohlcv = load_ohlcv(DATA_FILE)
    closes = np.array([c[4] for c in ohlcv], dtype=np.float64)
    print(f"📥 {len(ohlcv)} velas 1h: {closes[0]:.0f} → {closes[-1]:.0f}")

    if args.strategies == "all":
        selected = list(STRATEGY_META.items())
    else:
        ids = [s.strip() for s in args.strategies.split(",")]
        selected = [(s, STRATEGY_META[s]) for s in ids]

    tops = {}
    for sid, meta in selected:
        sf = SIMULATIONS_DIR / f"best_{sid}.json"
        tops[sid] = run_strategy(sid, meta, ohlcv, closes, args.iter, sf)

    print(f"\n{'='*55}")
    print(f"  🏆 COMPARATIVA FINAL")
    print(f"{'='*55}")
    print(f"  {'Estrategia':<18} {'Score':<8} {'ROI':<10} {'WR':<7} {'Ops':<6} {'PnL':<10}")
    print(f"  {'─'*59}")
    for sid, ts in tops.items():
        if ts:
            s = ts[0]["summary"]
            print(f"  {STRATEGY_META[sid]['name']:<18} {ts[0]['score']:<+8} {s['roi']:<+10} "
                  f"{s['win_rate']:<7} {s['trades']:<6} {s['total_pnl']:<+10}")


if __name__ == "__main__":
    main()
