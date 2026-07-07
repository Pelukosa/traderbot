#!/usr/bin/env python3
"""
SIMON — Simulador de estrategias de trading.

Carga las 720 velas 1h, prueba todas las estrategias registradas,
y devuelve el TOP 3 con el mismo formato siempre.

Uso:
    uv run python scripts/simular.py                        # probar todas
    uv run python scripts/simular.py --ver "MACD+RSI"        # detalle de una
    uv run python scripts/simular.py --descargar             # descargar velas frescas
"""
from src.simulador import main, ejecutar, imprimir_resultado, imprimir_top3, descargar_velas
import sys

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--descargar":
        descargar_velas()
    elif len(sys.argv) > 2 and sys.argv[1] == "--ver":
        resultados = ejecutar()
        nombre = " ".join(sys.argv[2:]).lower()
        for r in resultados:
            if nombre in r.estrategia.lower():
                imprimir_resultado(r)
                break
        else:
            print(f"No encontre '{nombre}'")
            print(f"Estrategias disponibles:")
            for r in resultados:
                print(f"  - {r.estrategia}")
    else:
        main()
