#!/usr/bin/env python3
"""
SIMON — Simulador interactivo de estrategias de trading.

Pide el % de fee, luego muestra un menú para elegir qué estrategia
simular (1 iteración) y muestra el resultado detallado.

El fee se aplica en compra Y en venta (el simulador lo hace automáticamente).

Uso:
    python scripts/simular.py
"""
import sys
from src.simulador import (
    cargar_velas, precalcular, imprimir_resultado,
    simular_estrategia, simular_macd_hist_50,
    simular_random_search, imprimir_random_top3,
    ESTRATEGIAS, DATA_FILE, INITIAL_EUR,
)


def menu(fee_pct: float):
    print(f"\n{'='*60}")
    print(f"  📊 SIMON — Simulador de estrategias")
    print(f"{'='*60}")
    print(f"  Datos: {DATA_FILE.name}")
    print(f"  Capital: {INITIAL_EUR}€  |  Fee: {fee_pct:.2f}% (compra + venta)")
    print(f"\n  Estrategias disponibles:")
    print(f"  {'─'*55}")

    for idx, est in enumerate(ESTRATEGIAS, 1):
        cfg = est["config"]
        parts = []
        for k, v in cfg.items():
            if k == "fee_percent":
                continue
            elif k == "min_histogram_abs" and v:
                parts.append(f"|hist|≥{v}")
            elif k == "sl_percent" and v:
                parts.append(f"SL={v}%")
            elif k == "tp_percent" and v:
                parts.append(f"TP={v}%")
            elif k == "invest_percent" and v:
                parts.append(f"inv={v}%")
        cfg_str = "  ".join(parts[-3:])
        print(f"  {idx:>2}. {est['nombre']:<28} {cfg_str}")

    print(f"  {'─'*55}")
    print(f"   0. Salir")
    print(f"   A. TODAS (una pasada a cada estrategia)")


def main():
    print(f"\n{'='*60}")
    print(f"  📊 SIMON — Simulador de estrategias")
    print(f"{'='*60}")

    # ── Pedir fee ──
    try:
        fee_input = input("  Fee por operación (%) [0.15]: ").strip()
        fee_pct = float(fee_input) if fee_input else 0.15
    except (EOFError, KeyboardInterrupt):
        print("\n👋 ¡Hasta luego!")
        return
    except ValueError:
        print("  Valor inválido, usando 0.15%")
        fee_pct = 0.15

    print(f"  Fee: {fee_pct:.2f}% (el simulador lo aplica en compra y en venta)")

    # ── Cargar datos ──
    print("📥 Cargando velas ...", end=" ", flush=True)
    ohlcv = cargar_velas()
    print(f"{len(ohlcv)} velas ({len(ohlcv) / 24:.0f} días)")
    pre = precalcular(ohlcv)

    while True:
        menu(fee_pct)
        try:
            choice = input("\n  Elige estrategia (número o A/0): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 ¡Hasta luego!")
            break

        if choice == "0":
            print("👋 ¡Hasta luego!")
            break

        if choice.upper() == "A":
            print(f"\n  Ejecutando TODAS (fee={fee_pct:.2f}%)...")
            print(f"  {'─'*55}")
            resultados = []
            for est in ESTRATEGIAS:
                if est.get("random_search"):
                    continue  # skip interactive strategies in batch mode
                print(f"  ⏳ {est['nombre']} ...", end=" ", flush=True)
                if est.get("custom_sim"):
                    r = simular_macd_hist_50(ohlcv, pre, est["config"],
                                             capital=INITIAL_EUR, fee_rate=fee_pct)
                else:
                    r = simular_estrategia(ohlcv, pre, est,
                                           capital=INITIAL_EUR, fee_rate=fee_pct)
                resultados.append(r)
                print(f"✓ {r.total_ops} ops, WR {r.winrate:.1f}%, ROI {r.roi_mensual:+.2f}%/mes")

            resultados.sort(key=lambda x: x.score, reverse=True)
            print(f"\n{'='*70}")
            print(f"  🏆 RANKING FINAL  |  fee={fee_pct:.2f}%")
            print(f"{'='*70}")
            print(f"  {'#':<3} {'Estrategia':<30} {'Ops':<5} {'WR%':<6} {'%/op':<7} {'€/op':<8} {'ROI/mes':<9} {'Score':<8}")
            print(f"  {'─'*70}")
            for idx, r in enumerate(resultados, 1):
                icon = "🥇" if idx == 1 else "🥈" if idx == 2 else "🥉" if idx == 3 else "  "
                print(f"  {icon} {r.estrategia:<28} {r.total_ops:<5} {r.winrate:<6.1f}"
                      f" {r.ganancia_media_por_op:<+7.2f} {r.pnl_por_operacion:<+8.2f}"
                      f" {r.roi_mensual:<+9.2f} {r.score:<8.2f}")
            continue

        try:
            idx = int(choice) - 1
            if idx < 0 or idx >= len(ESTRATEGIAS):
                print(f"  ❌ Número fuera de rango (1-{len(ESTRATEGIAS)})")
                continue
        except ValueError:
            print(f"  ❌ Opción no válida: '{choice}'")
            continue

        est = ESTRATEGIAS[idx]
        print(f"\n  ⏳ Simulando: {est['nombre']} (fee={fee_pct:.2f}%) ...")

        # ── 🎲 Random Search ──
        if est.get("random_search"):
            try:
                n_input = input(f"  Número de iteraciones (combinaciones aleatorias): ").strip()
                num_iter = int(n_input)
                if num_iter <= 0:
                    print("  ❌ Debe ser > 0")
                    continue
            except (EOFError, KeyboardInterrupt):
                print("\n👋 ¡Hasta luego!")
                break
            except ValueError:
                print("  ❌ Número no válido")
                continue

            resultados, configs = simular_random_search(
                ohlcv, pre, num_iter,
                capital=INITIAL_EUR, fee_rate=fee_pct,
            )
            imprimir_random_top3(resultados, configs)

        # ── Estrategia con simulación dedicada (ej: MACD Hist -50) ──
        elif est.get("custom_sim"):
            r = simular_macd_hist_50(ohlcv, pre, est["config"],
                                     capital=INITIAL_EUR, fee_rate=fee_pct)
            imprimir_resultado(r)

        # ── Estrategia estándar ──
        else:
            r = simular_estrategia(ohlcv, pre, est, capital=INITIAL_EUR, fee_rate=fee_pct)
            imprimir_resultado(r)


if __name__ == "__main__":
    main()
