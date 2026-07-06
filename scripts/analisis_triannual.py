#!/usr/bin/env python3
"""
Análisis de rendimiento de trading — ejecutar cada 3 días.

Lee performance.csv, analiza patrones de operaciones ganadoras vs perdedoras,
y genera un informe con propuestas de ajuste para la estrategia MACDDivergence.
"""
from __future__ import annotations

import csv
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

PERF_LOG = Path(__file__).parent.parent / "performance.csv"


def load_trades() -> list[dict]:
    if not PERF_LOG.exists():
        return []
    with open(PERF_LOG) as f:
        return list(csv.DictReader(f))


def _parse_float(val: str, default: float = 0.0) -> float:
    try:
        return float(val) if val else default
    except (ValueError, TypeError):
        return default


def _safe_div(a: float, b: float) -> float:
    return a / b if b != 0 else 0.0


def analyze() -> str:
    trades = load_trades()
    if not trades:
        return "📊 *No hay operaciones todavía.*\n\nEl bot está corriendo y recogiendo datos. Vuelve a pedir el informe en 3 días."

    now = datetime.now(timezone.utc)

    # Filter completed sells (they have both entry and exit prices)
    sells = [t for t in trades if t["side"] == "sell" and t.get("exit_price")]
    if not sells:
        return "📊 *No hay operaciones cerradas todavía.*\n\nHay {} registros pero ninguna venta completada.".format(len(trades))

    total_ops = len(sells)
    wins = [t for t in sells if _parse_float(t["pnl_eur"]) > 0]
    losses = [t for t in sells if _parse_float(t["pnl_eur"]) <= 0]
    win_count = len(wins)
    loss_count = len(losses)
    total_pnl = sum(_parse_float(t["pnl_eur"]) for t in sells)
    total_fees = sum(_parse_float(t["fee_eur"]) for t in trades)
    win_rate = _safe_div(win_count, total_ops) * 100

    durations = [_parse_float(t.get("duration_minutes")) for t in sells if _parse_float(t.get("duration_minutes")) > 0]
    avg_duration = statistics.mean(durations) if durations else 0.0

    avg_win = statistics.mean(_parse_float(t["pnl_eur"]) for t in wins) if wins else 0.0
    avg_loss = statistics.mean(_parse_float(t["pnl_eur"]) for t in losses) if losses else 0.0
    profit_factor = abs(_safe_div(sum(_parse_float(t["pnl_eur"]) for t in wins), sum(_parse_float(t["pnl_eur"]) for t in losses))) if losses else float("inf")

    # ── Patrones: analizar variables de contexto ──
    # Fields to analyze for patterns
    context_fields = [
        ("buy_valley_value", "Valor del valle (entrada)"),
        ("buy_valley_to_entry_diff", "Diferencia valle→entrada"),
        ("buy_reversal_slope", "Pendiente de reversión"),
        ("market_trend_20", "Tendencia 20 velas (%)"),
        ("market_volatility_20", "Volatilidad 20 velas"),
        ("market_volume_ratio", "Ratio de volumen"),
        ("hour_of_day", "Hora del día"),
        ("duration_minutes", "Duración (min)"),
        ("buy_histogram", "Histograma en compra"),
    ]

    patterns: list[str] = []
    recommendations: list[str] = []

    for field, label in context_fields:
        win_vals = [_parse_float(t.get(field)) for t in wins if t.get(field)]
        loss_vals = [_parse_float(t.get(field)) for t in losses if t.get(field)]

        if len(win_vals) >= 3 and len(loss_vals) >= 3:
            w_mean = statistics.mean(win_vals)
            l_mean = statistics.mean(loss_vals)
            w_std = statistics.stdev(win_vals) if len(win_vals) > 1 else 0.0
            l_std = statistics.stdev(loss_vals) if len(loss_vals) > 1 else 0.0

            # Effect size (Cohen's d approximation)
            pooled_std = math.sqrt((w_std**2 + l_std**2) / 2) if (w_std + l_std) > 0 else 1.0
            effect = abs(w_mean - l_mean) / max(pooled_std, 0.01)

            if effect > 0.5:  # Medium or larger effect
                direction = "↑ mayor" if w_mean > l_mean else "↓ menor"
                patterns.append(
                    f"  • *{label}*: ganadoras {w_mean:.2f} vs perdedoras {l_mean:.2f}  "
                    f"(d={effect:.2f}, {direction})"
                )

                # Generate recommendation
                if field == "buy_valley_value" and w_mean < l_mean:
                    recommendations.append(
                        f"📌 *Filtrar valles débiles*: Las ganadoras tienen valle más profundo "
                        f"(media {w_mean:.1f}) que las perdedoras ({l_mean:.1f}). "
                        f"Propongo ignorar valles con valor > {w_mean + w_std:.1f}."
                    )
                elif field == "buy_reversal_slope" and w_mean > l_mean:
                    recommendations.append(
                        f"📌 *Pendiente mínima*: Las ganadoras tienen pendiente de reversión "
                        f"de {w_mean:.2f} vs {l_mean:.2f} en perdedoras. "
                        f"Propongo requerir pendiente mínima de {w_mean - w_std:.2f}."
                    )
                elif field == "market_trend_20" and w_mean > l_mean:
                    recommendations.append(
                        f"📌 *Tendencia favorable*: Las ganadoras tienen tendencia "
                        f"{w_mean:+.1f}% vs {l_mean:+.1f}% en perdedoras. "
                        f"Considerar solo entradas con tendencia > {max(0, w_mean - w_std):.1f}%."
                    )
                elif field == "market_volatility_20" and w_mean < l_mean:
                    recommendations.append(
                        f"📌 *Evitar alta volatilidad*: Perdedoras tienen volatilidad "
                        f"{l_mean:.2f} vs {w_mean:.2f} en ganadoras. "
                        f"Propongo ignorar entradas con volatilidad > {w_mean + w_std:.2f}."
                    )
                elif field == "duration_minutes" and w_mean > l_mean:
                    recommendations.append(
                        f"📌 *Duración mínima*: Ganadoras duraron {w_mean:.0f}min de media, "
                        f"perdedoras {l_mean:.0f}min. "
                        f"Considerar no vender antes de {w_mean - w_std:.0f}min."
                    )
                elif field == "hour_of_day":
                    # Find best hours
                    win_hours = defaultdict(int)
                    loss_hours = defaultdict(int)
                    for t in wins:
                        win_hours[int(_parse_float(t.get("hour_of_day")))] += 1
                    for t in losses:
                        loss_hours[int(_parse_float(t.get("hour_of_day")))] += 1
                    best_hours = []
                    for h in range(24):
                        total_h = win_hours[h] + loss_hours[h]
                        if total_h >= 2:
                            wr_h = win_hours[h] / total_h * 100
                            if wr_h > 60:
                                best_hours.append(f"{h:02d}:00 ({wr_h:.0f}%)")
                    if best_hours:
                        recommendations.append(
                            f"📌 *Mejores horas*: {', '.join(best_hours[:5])}. "
                            f"Considerar operar solo en estas franjas."
                        )

    # ── Historical comparison ──
    # Track win rate per day / period
    trades_by_day = defaultdict(list)
    for t in sells:
        ts = t.get("timestamp", "")
        day = ts[:10] if ts else "unknown"
        trades_by_day[day].append(t)

    # Per-day win rates
    daily_wr = []
    for day, day_trades in sorted(trades_by_day.items()):
        day_sells = [t for t in day_trades if t["side"] == "sell"]
        if day_sells:
            day_wins = sum(1 for t in day_sells if _parse_float(t["pnl_eur"]) > 0)
            daily_wr.append((day, _safe_div(day_wins, len(day_sells)) * 100))

    # ── Build report ──
    lines = [
        f"📊 *INFORME DE TRADING — {now.strftime('%d/%m/%Y %H:%M')} UTC*",
        f"",
        f"━ *RESUMEN GENERAL*",
        f"  Operaciones: {total_ops}",
        f"  Ganadas: {win_count}  Perdidas: {loss_count}",
        f"  Win Rate: {win_rate:.1f}%",
        f"  PnL Total: {total_pnl:+.2f}€",
        f"  Comisiones: {total_fees:.2f}€",
        f"  Ganancia media: {avg_win:+.2f}€  Pérdida media: {avg_loss:+.2f}€",
        f"  Profit Factor: {profit_factor:.2f}x",
        f"  Duración media: {avg_duration:.0f} min",
        f"",
    ]

    if daily_wr:
        lines.append("━ *EVOLUCIÓN DIARIA*")
        for day, wr in daily_wr:
            bar = "█" * max(1, int(wr / 5))
            lines.append(f"  {day}: {wr:.0f}% {bar}")
        lines.append("")

    if patterns:
        lines.append("━ *PATRONES DETECTADOS*")
        lines.extend(patterns)
        lines.append("")
    else:
        lines.append("━ *PATRONES* (necesitas más datos para detectar patrones significativos)")
        lines.append("")

    if recommendations:
        lines.append("━ *PROPUESTAS DE AJUSTE*")
        for r in recommendations:
            lines.append(r)
        lines.append("")
        lines.append("_Responde OK para aplicar los cambios, o dime qué ajustar._")
    else:
        lines.append("━ *PROPUESTAS* (sin datos suficientes aún para recomendar cambios)")
        lines.append("")

    lines.append("──")
    lines.append(f"📈 *MACDDivergence*  ·  {total_ops} ops  ·  {win_rate:.0f}% WR  ·  {total_pnl:+.2f}€")

    return "\n".join(lines)


if __name__ == "__main__":
    print(analyze())
