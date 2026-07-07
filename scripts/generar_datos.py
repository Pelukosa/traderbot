#!/usr/bin/env python3
"""
Genera data.json para el frontend de SIMON.

Ejecuta esto cada vez que cambien las estrategias o los datos de velas:
    python scripts/generar_datos.py

El frontend (cd frontend && npm run dev) cargará el JSON automáticamente.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.simulador import (
    cargar_velas, precalcular, simular_estrategia_con_indices, ESTRATEGIAS,
    INITIAL_EUR, FEE_RATE,
)


def generar_datos() -> dict:
    print("Cargando velas...")
    ohlcv = cargar_velas()
    print(f"  {len(ohlcv)} velas 1h ({len(ohlcv)/24:.1f} dias)")

    print("Precalculando indicadores...")
    pre = precalcular(ohlcv)

    closes = pre["closes"].tolist()
    opens = [c[1] for c in ohlcv]
    highs = pre["highs"].tolist()
    lows = pre["lows"].tolist()
    vols = pre["vols"].tolist()
    timestamps = [c[0] for c in ohlcv]

    rsi = [None if np.isnan(v) else round(float(v), 2) for v in pre["rsi_arr"]]
    macd_line = [None if np.isnan(v) else round(float(v), 2) for v in pre["macd_line"]]
    macd_sig = [None if np.isnan(v) else round(float(v), 2) for v in pre["macd_sig"]]
    hist = [None if np.isnan(v) else round(float(v), 2) for v in pre["hist_arr"]]

    velas_json = [{
        "t": timestamps[i], "o": round(opens[i], 2), "h": round(highs[i], 2),
        "l": round(lows[i], 2), "c": round(closes[i], 2), "v": round(vols[i], 4),
        "rsi": rsi[i], "macd": macd_line[i], "macd_sig": macd_sig[i], "hist": hist[i],
    } for i in range(len(ohlcv))]

    print("Simulando estrategias...")
    estrategias_json = []
    for est in ESTRATEGIAS:
        r, trades_con_idx = simular_estrategia_con_indices(
            ohlcv, pre, est, capital=INITIAL_EUR, fee_rate=FEE_RATE
        )
        datos = r.a_dict()
        trades_json = []
        for t in trades_con_idx:
            trades_json.append({
                "entry_price": round(t[0], 2), "exit_price": round(t[1], 2),
                "gain_pct": round(t[2], 2), "duration_h": round(t[3], 1),
                "exit_type": t[4],
                "buy_idx": t[5], "sell_idx": t[6],
            })
        estrategias_json.append({
            "id": est["id"],
            "nombre": est["nombre"],
            "resultado": datos,
            "trades": trades_json,
        })
        print(f"  ✓ {est['nombre']}: {datos['total_ops']} ops, "
              f"WR {datos['winrate']}%, ROI {datos['roi_mensual']}%")

    return {"velas": velas_json, "estrategias": estrategias_json}


def main():
    datos = generar_datos()
    out = Path(__file__).parent.parent / "frontend" / "public" / "data.json"
    out.write_text(json.dumps(datos, ensure_ascii=False))
    print(f"\n✅ Generado {out.absolute()}")
    print(f"   cd frontend && npm run dev")


if __name__ == "__main__":
    main()
