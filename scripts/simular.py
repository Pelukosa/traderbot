#!/usr/bin/env python3
"""
Simulador de estrategia MACD Divergence sobre datos históricos.

Uso:
    python scripts/simular.py                          # parámetros por defecto
    python scripts/simular.py --min-hist 60 --sl 1.5   # parámetros custom

Los resultados se guardan en simulations/ para comparar entre pruebas.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# ── Configuración por defecto (modificable por CLI) ──

DEFAULT_CONFIG = {
    "fast": 12,
    "slow": 26,
    "signal": 9,
    "confirm_velas": 1,
    "min_histogram_abs": 50.0,       # |histograma| mínimo para operar
    "sl_percent": 1.0,               # stop-loss fijo (%)
    "trail_retain": 0.6,            # 60% de ganancia retenida en trailing
    "trailing_min_gain": 1.0,       # % mínimo de ganancia para activar trailing
    "invest_percent": 95.0,          # % del saldo EUR a invertir
    "max_position_hours": 6,         # tiempo máximo en posición (horas)
    "min_balance_eur": 5.0,          # saldo mínimo para comprar
    "fee_percent": 0.26,             # comisión por operación (%) — maker/taker
    "initial_eur": 125.0,            # saldo inicial EUR
}

SIMULATIONS_DIR = Path(__file__).parent.parent / "simulations"


@dataclass
class SimTrade:
    """Una operación completa de compra-venta."""
    buy_time: str = ""
    buy_price: float = 0.0
    buy_hist: float = 0.0
    buy_valley: float = 0.0
    sell_time: str = ""
    sell_price: float = 0.0
    sell_reason: str = ""  # signal | stop_loss | trailing | max_time
    sell_hist: float = 0.0
    sell_peak: float = 0.0
    duration_hours: float = 0.0
    pnl_eur: float = 0.0
    pnl_pct: float = 0.0
    fees_eur: float = 0.0
    net_pnl_eur: float = 0.0
    net_pnl_pct: float = 0.0


@dataclass
class SimResult:
    config: dict = field(default_factory=dict)
    trades: list = field(default_factory=list)
    final_eur: float = 0.0
    total_fees: float = 0.0
    win_count: int = 0
    loss_count: int = 0
    total_pnl_eur: float = 0.0
    total_net_pnl_eur: float = 0.0


def ema(data: np.ndarray, period: int) -> np.ndarray:
    result = np.empty_like(data)
    result[:] = np.nan
    # Find first valid index
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


def load_ohlcv(csv_path: str = "historical_1h.csv") -> list[list[float]]:
    """Load OHLCV from CSV. Returns same format as exchange.fetch_ohlcv."""
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


def simulate(ohlcv: list[list[float]], config: dict | None = None) -> SimResult:
    """Run the MACD Divergence strategy on historical data."""
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    result = SimResult(config=cfg)

    closes = np.array([c[4] for c in ohlcv], dtype=np.float64)
    volumes = np.array([c[5] for c in ohlcv], dtype=np.float64)

    # MACD computation
    ema_fast = ema(closes, cfg["fast"])
    ema_slow = ema(closes, cfg["slow"])
    macd_line = ema_fast - ema_slow
    sig_line = ema(macd_line, cfg["signal"])
    hist = macd_line - sig_line  # histogram values for all candles

    # State
    eur_balance = cfg["initial_eur"]
    btc_balance = 0.0
    in_position = False
    entry_price = 0.0
    entry_idx = 0
    entry_time = ""
    highest_price = 0.0
    stop_loss = None
    valley_detected = None
    peak_detected = None
    awaiting_action = None
    confirm_count = 0
    hist_buffer = []
    trades = []
    total_fees = 0.0

    min_len = cfg["slow"] + cfg["signal"] + 5
    fee_rate = cfg["fee_percent"] / 100.0
    sl_rate = cfg["sl_percent"] / 100.0

    for i in range(len(ohlcv)):
        h = hist[i]
        if np.isnan(h):
            continue

        timestamp_str = datetime.fromtimestamp(ohlcv[i][0] / 1000, tz=timezone.utc).isoformat()
        close_price = float(closes[i])

        # Build histogram buffer (last 5 values)
        hist_buffer.append(float(h))
        if len(hist_buffer) > 5:
            hist_buffer = hist_buffer[-5:]

        # Need at least 4 values for detection
        buf = hist_buffer

        # ── Position management ──
        if in_position:
            # Check SL
            if stop_loss is not None and close_price <= stop_loss:
                # SELL by stop-loss
                sell_amt = btc_balance
                pnl = (close_price - entry_price) * sell_amt
                pnl_pct = ((close_price - entry_price) / entry_price) * 100
                fee = sell_amt * close_price * fee_rate
                total_fees += fee
                net_pnl = pnl - fee
                eur_balance += sell_amt * close_price - fee
                btc_balance = 0.0

                # Deduct buy fee (already accounted in cost basis)
                dur = (i - entry_idx) / 3600.0  # hours

                trades.append(SimTrade(
                    buy_time=entry_time, buy_price=entry_price,
                    buy_hist=0, buy_valley=valley_detected or 0,
                    sell_time=timestamp_str, sell_price=close_price,
                    sell_reason="stop_loss",
                    sell_hist=float(h), sell_peak=peak_detected or 0,
                    duration_hours=dur,
                    pnl_eur=pnl, pnl_pct=pnl_pct,
                    fees_eur=fee, net_pnl_eur=net_pnl, net_pnl_pct=pnl_pct,
                ))
                in_position = False
                stop_loss = None
                continue

            # Trailing stop
            if close_price > highest_price:
                highest_price = close_price
                gain_pct = (close_price - entry_price) / entry_price * 100
                if gain_pct > cfg["trailing_min_gain"]:
                    trail_pct = gain_pct * cfg["trail_retain"]
                    new_sl = entry_price * (1 + trail_pct / 100.0)
                    if stop_loss is None or new_sl > stop_loss:
                        stop_loss = new_sl
                        # SELL by trailing
                        if close_price <= stop_loss:
                            sell_amt = btc_balance
                            pnl = (close_price - entry_price) * sell_amt
                            pnl_pct = ((close_price - entry_price) / entry_price) * 100
                            fee = sell_amt * close_price * fee_rate
                            total_fees += fee
                            net_pnl = pnl - fee
                            eur_balance += sell_amt * close_price - fee
                            btc_balance = 0.0
                            dur = (i - entry_idx) / 3600.0
                            trades.append(SimTrade(
                                buy_time=entry_time, buy_price=entry_price,
                                buy_hist=0, buy_valley=valley_detected or 0,
                                sell_time=timestamp_str, sell_price=close_price,
                                sell_reason="trailing",
                                sell_hist=float(h), sell_peak=peak_detected or 0,
                                duration_hours=dur,
                                pnl_eur=pnl, pnl_pct=pnl_pct,
                                fees_eur=fee, net_pnl_eur=net_pnl, net_pnl_pct=pnl_pct,
                            ))
                            in_position = False
                            stop_loss = None
                            continue

            # Max time
            hours_in_pos = (i - entry_idx) / 3600.0
            if hours_in_pos >= cfg["max_position_hours"]:
                sell_amt = btc_balance
                pnl = (close_price - entry_price) * sell_amt
                pnl_pct = ((close_price - entry_price) / entry_price) * 100
                fee = sell_amt * close_price * fee_rate
                total_fees += fee
                net_pnl = pnl - fee
                eur_balance += sell_amt * close_price - fee
                btc_balance = 0.0
                trades.append(SimTrade(
                    buy_time=entry_time, buy_price=entry_price,
                    buy_hist=0, buy_valley=valley_detected or 0,
                    sell_time=timestamp_str, sell_price=close_price,
                    sell_reason="max_time",
                    sell_hist=float(h), sell_peak=peak_detected or 0,
                    duration_hours=hours_in_pos,
                    pnl_eur=pnl, pnl_pct=pnl_pct,
                    fees_eur=fee, net_pnl_eur=net_pnl, net_pnl_pct=pnl_pct,
                ))
                in_position = False
                stop_loss = None
                continue

            # Sell by signal (peak confirmed)
            if awaiting_action == "sell" and len(buf) >= 4:
                # confirm_count starts at 1 when peak detected.
                # We need confirm_velas MORE candles after detection.
                confirm_count += 1
                if confirm_count > cfg["confirm_velas"]:
                    sell_amt = btc_balance
                    pnl = (close_price - entry_price) * sell_amt
                    pnl_pct = ((close_price - entry_price) / entry_price) * 100
                    fee = sell_amt * close_price * fee_rate
                    total_fees += fee
                    net_pnl = pnl - fee
                    eur_balance += sell_amt * close_price - fee
                    btc_balance = 0.0
                    trades.append(SimTrade(
                        buy_time=entry_time, buy_price=entry_price,
                        buy_hist=0, buy_valley=valley_detected or 0,
                        sell_time=timestamp_str, sell_price=close_price,
                        sell_reason="signal",
                        sell_hist=float(h), sell_peak=peak_detected or 0,
                        duration_hours=(i - entry_idx) / 3600.0,
                        pnl_eur=pnl, pnl_pct=pnl_pct,
                        fees_eur=fee, net_pnl_eur=net_pnl, net_pnl_pct=pnl_pct,
                    ))
                    in_position = False
                    stop_loss = None
                    awaiting_action = None
                    confirm_count = 0
                    peak_detected = None
                    continue

        # ── Signal detection (both buy and sell, regardless of position) ──
        if len(buf) >= 4:
            # Valley detection (buy) — only if NOT in position
            if not in_position:
                if (buf[-4] >= buf[-3] > buf[-2] < buf[-1]
                        and buf[-2] < 0 and buf[-1] < 0
                        and abs(buf[-2]) >= cfg["min_histogram_abs"]):
                    valley_detected = buf[-2]
                    awaiting_action = "buy"
                    confirm_count = 0

            # Peak detection (sell) — only if IN position
            if in_position:
                if (buf[-4] <= buf[-3] < buf[-2] > buf[-1]
                        and buf[-2] > 0 and buf[-1] > 0
                        and abs(buf[-2]) >= cfg["min_histogram_abs"]):
                    peak_detected = buf[-2]
                    awaiting_action = "sell"
                    confirm_count = 0

        # ── Execute buy on confirmation ──
        if awaiting_action == "buy" and not in_position:
            confirm_count += 1
            if confirm_count > cfg["confirm_velas"]:
                invest_eur = eur_balance * cfg["invest_percent"] / 100.0
                if invest_eur >= cfg["min_balance_eur"]:
                    buy_amount = invest_eur / close_price
                    buy_fee = invest_eur * fee_rate
                    total_fees += buy_fee
                    btc_balance = buy_amount
                    eur_balance -= invest_eur + buy_fee
                    entry_price = close_price
                    entry_idx = i
                    entry_time = timestamp_str
                    highest_price = close_price
                    stop_loss = entry_price * (1 - sl_rate)
                    in_position = True
                    valley_detected = buf[-2]
                awaiting_action = None
                confirm_count = 0

        # ── Execute sell on confirmation (if not already sold above) ──
        if awaiting_action == "sell" and in_position:
            pass  # handled above in position management

    # Summary stats
    wins = sum(1 for t in trades if t.net_pnl_eur > 0)
    losses = sum(1 for t in trades if t.net_pnl_eur <= 0)
    total_pnl = sum(t.pnl_eur for t in trades)
    total_net = sum(t.net_pnl_eur for t in trades)

    result.trades = trades
    result.final_eur = round(eur_balance + btc_balance * (closes[-1] if len(closes) > 0 else 0), 2)
    result.total_fees = round(total_fees, 2)
    result.win_count = wins
    result.loss_count = losses
    result.total_pnl_eur = round(total_pnl, 2)
    result.total_net_pnl_eur = round(total_net, 2)
    return result


def print_result(result: SimResult):
    """Print a readable summary of simulation results."""
    cfg = result.config
    trades = result.trades
    total_ops = len(trades)
    wr = (result.win_count / total_ops * 100) if total_ops > 0 else 0
    avg_win = (sum(t.net_pnl_eur for t in trades if t.net_pnl_eur > 0) / result.win_count) if result.win_count > 0 else 0
    avg_loss = (sum(t.net_pnl_eur for t in trades if t.net_pnl_eur <= 0) / result.loss_count) if result.loss_count > 0 else 0
    profit_factor = abs(sum(t.net_pnl_eur for t in trades if t.net_pnl_eur > 0) / 
                        min(sum(t.net_pnl_eur for t in trades if t.net_pnl_eur < 0), -0.01)) if result.loss_count > 0 else float("inf")
    avg_dur = sum(t.duration_hours for t in trades) / total_ops if total_ops > 0 else 0
    initial = cfg["initial_eur"]
    final = result.final_eur
    roi = ((final - initial) / initial * 100) if initial > 0 else 0

    lines = [
        f"\n{'='*55}",
        f"  📊 SIMULACIÓN MACD DIVERGENCE",
        f"{'='*55}",
        f"",
        f"  ⚙️  PARÁMETROS:",
        f"     MACD fast/slow/signal: {cfg['fast']}/{cfg['slow']}/{cfg['signal']}",
        f"     |histograma| mínimo:  {cfg['min_histogram_abs']}",
        f"     Confirmación velas:   {cfg['confirm_velas']}",
        f"     Stop-loss:            {cfg['sl_percent']}%",
        f"     Trailing:             desde {cfg['trailing_min_gain']}%, retiene {cfg['trail_retain']*100:.0f}%",
        f"     Máx posición:         {cfg['max_position_hours']}h",
        f"     Comisión:             {cfg['fee_percent']}%",
        f"     Inversión inicial:    {initial:.2f}€",
        f"",
        f"  📈 RESULTADOS:",
        f"     Operaciones:          {total_ops}",
        f"     Ganadas:              {result.win_count}",
        f"     Perdidas:             {result.loss_count}",
        f"     Win Rate:             {wr:.1f}%",
        f"     Beneficio medio:      {avg_win:+.2f}€",
        f"     Pérdida media:        {avg_loss:+.2f}€",
        f"     Profit Factor:        {profit_factor:.2f}x",
        f"     Duración media:       {avg_dur:.1f}h",
        f"     Comisiones totales:   {result.total_fees:.2f}€",
        f"     PnL bruto:            {result.total_pnl_eur:+.2f}€",
        f"     PnL neto:             {result.total_net_pnl_eur:+.2f}€",
        f"     Balance final:        {final:.2f}€",
        f"     ROI:                  {roi:+.2f}%",
        f"",
    ]

    if total_ops > 0:
        lines.append(f"  📋 ÚLTIMAS 10 OPERACIONES:")
        lines.append(f"     {'#':<4} {'Compra':<20} {'Venta':<20} {'Dur':<5} {'PnL':<10} {'Motivo':<12}")
        lines.append(f"     {'-'*73}")
        for idx, t in enumerate(trades[-10:], 1):
            buy_short = t.buy_time[11:16] if t.buy_time else "?"
            sell_short = t.sell_time[11:16] if t.sell_time else "?"
            lines.append(
                f"     {idx:<4} {buy_short:<20} {sell_short:<20} "
                f"{t.duration_hours:.1f}h {t.net_pnl_eur:+.2f}€ {t.sell_reason:<12}"
            )
        lines.append("")

    lines.append(f"{'='*55}\n")
    print("\n".join(lines))

    # Save to file
    SIMULATIONS_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = SIMULATIONS_DIR / f"sim_{ts}.json"
    with open(out, "w") as f:
        json.dump({
            "config": cfg,
            "trades": [
                {"buy_time": t.buy_time, "buy_price": t.buy_price,
                 "sell_time": t.sell_time, "sell_price": t.sell_price,
                 "sell_reason": t.sell_reason,
                 "duration_hours": round(t.duration_hours, 2),
                 "pnl_eur": round(t.pnl_eur, 2), "net_pnl_eur": round(t.net_pnl_eur, 2),
                 "pnl_pct": round(t.pnl_pct, 2)}
                for t in trades
            ],
            "summary": {
                "trades": total_ops, "wins": result.win_count, "losses": result.loss_count,
                "win_rate": round(wr, 1),
                "total_fees": result.total_fees,
                "total_pnl_bruto": result.total_pnl_eur,
                "total_pnl_neto": result.total_net_pnl_eur,
                "initial_eur": initial,
                "final_eur": final,
                "roi_pct": round(roi, 2),
            }
        }, f, indent=2)
    print(f"  📁 Resultados guardados en: {out}\n")


def main():
    parser = argparse.ArgumentParser(description="Simulador MACD Divergence")
    parser.add_argument("--min-hist", type=float, default=None, help="|histograma| mínimo")
    parser.add_argument("--sl", type=float, default=None, help="Stop-loss %")
    parser.add_argument("--trail-start", type=float, default=None, help="Trailing desde %")
    parser.add_argument("--trail-retain", type=float, default=None, help="Trailing retiene % (0-1)")
    parser.add_argument("--confirm", type=int, default=None, help="Velas de confirmación")
    parser.add_argument("--max-hours", type=float, default=None, help="Máx horas en posición")
    parser.add_argument("--fee", type=float, default=None, help="Comisión %")
    parser.add_argument("--invest", type=float, default=None, help="% del saldo a invertir")
    parser.add_argument("--initial", type=float, default=None, help="Saldo inicial EUR")
    parser.add_argument("--data", type=str, default="historical_1h.csv", help="CSV de velas")
    parser.add_argument("--list", action="store_true", help="Listar simulaciones anteriores")
    args = parser.parse_args()

    if args.list:
        SIMULATIONS_DIR.mkdir(exist_ok=True)
        files = sorted(SIMULATIONS_DIR.glob("*.json"))
        if not files:
            print("No hay simulaciones guardadas.")
            return
        print(f"\n📁 Simulaciones disponibles ({len(files)}):")
        for f in files[-10:]:
            with open(f) as fh:
                data = json.load(fh)
            s = data["summary"]
            cfg = data["config"]
            print(f"  {f.name}")
            print(f"     |hist|≥{cfg['min_histogram_abs']}  SL={cfg['sl_percent']}%  "
                  f"Trail={cfg['trailing_min_gain']}%→{cfg['trail_retain']*100:.0f}%  "
                  f"Ops={s['trades']}  WR={s['win_rate']}%  ROI={s['roi_pct']:+.1f}%  "
                  f"PnL={s['total_pnl_neto']:+.2f}€")
        return

    # Load data
    csv_path = args.data
    if not os.path.exists(csv_path):
        print(f"❌ No se encuentra {csv_path}. Ejecuta primero la descarga de velas.")
        sys.exit(1)
    ohlcv = load_ohlcv(csv_path)
    print(f"📥 {len(ohlcv)} velas cargadas desde {csv_path}")

    # Build config from defaults + overrides
    config = dict(DEFAULT_CONFIG)
    if args.min_hist is not None:
        config["min_histogram_abs"] = args.min_hist
    if args.sl is not None:
        config["sl_percent"] = args.sl
    if args.trail_start is not None:
        config["trailing_min_gain"] = args.trail_start
    if args.trail_retain is not None:
        config["trail_retain"] = args.trail_retain
    if args.confirm is not None:
        config["confirm_velas"] = args.confirm
    if args.max_hours is not None:
        config["max_position_hours"] = args.max_hours
    if args.fee is not None:
        config["fee_percent"] = args.fee
    if args.invest is not None:
        config["invest_percent"] = args.invest
    if args.initial is not None:
        config["initial_eur"] = args.initial

    result = simulate(ohlcv, config)
    print_result(result)


if __name__ == "__main__":
    main()
