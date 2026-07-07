#!/usr/bin/env python3
"""
Genera un frontend HTML con grafico de velas + RSI + MACD + selector de estrategias.

Las velas se cargan del CSV local (no llama a la API de Kraken).
Al seleccionar una estrategia y pulsar "Generar", muestra:
  - En el grafico: flechas verdes (compra) y rojas (venta)
  - En un card: todos los datos de la estrategia

Uso:
    uv run python scripts/frontend.py
    # Abre http://localhost:8050
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template_string

# Anyadir src al path para importar el simulador
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.simulador import (
    cargar_velas, precalcular, simular_estrategia, ESTRATEGIAS, DATA_FILE,
    INITIAL_EUR, FEE_RATE,
)

app = Flask(__name__)

# Cache de datos
_ohlcv = None
_pre = None
_resultados = {}
_html_template = None


def get_data():
    global _ohlcv, _pre
    if _ohlcv is None:
        _ohlcv = cargar_velas()
        _pre = precalcular(_ohlcv)
    return _ohlcv, _pre


def generar_resultados():
    global _resultados
    ohlcv, pre = get_data()
    for est in ESTRATEGIAS:
        rid = est["id"]
        if rid not in _resultados:
            r = simular_estrategia(ohlcv, pre, est, capital=INITIAL_EUR, fee_rate=FEE_RATE)
            _resultados[rid] = r.a_dict()
    return _resultados


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/velas")
def api_velas():
    ohlcv, pre = get_data()
    closes = pre["closes"].tolist()
    highs = pre["highs"].tolist()
    lows = pre["lows"].tolist()
    vols = pre["vols"].tolist()
    rsi = np.where(np.isnan(pre["rsi_arr"]), None, pre["rsi_arr"]).tolist()
    macd = np.where(np.isnan(pre["macd_line"]), None, pre["macd_line"]).tolist()
    macd_sig = np.where(np.isnan(pre["macd_sig"]), None, pre["macd_sig"]).tolist()
    hist = np.where(np.isnan(pre["hist_arr"]), None, pre["hist_arr"]).tolist()

    velas = []
    for i, c in enumerate(ohlcv):
        velas.append({
            "ts": c[0], "open": c[1], "high": c[2], "low": c[3],
            "close": c[4], "volume": c[5],
            "rsi": rsi[i] if i < len(rsi) else None,
            "macd": macd[i] if i < len(macd) else None,
            "macd_sig": macd_sig[i] if i < len(macd_sig) else None,
            "hist": hist[i] if i < len(hist) else None,
        })

    return jsonify({"velas": velas, "total": len(velas), "dias": round(len(velas)/24, 1)})


@app.route("/api/estrategias")
def api_estrategias():
    return jsonify([{"id": e["id"], "nombre": e["nombre"]} for e in ESTRATEGIAS])


@app.route("/api/simular/<estrategia_id>")
def api_simular(estrategia_id: str):
    ohlcv, pre = get_data()
    resultados = generar_resultados()

    if estrategia_id not in resultados:
        # Simular solo esta
        for est in ESTRATEGIAS:
            if est["id"] == estrategia_id:
                r = simular_estrategia(ohlcv, pre, est, capital=INITIAL_EUR, fee_rate=FEE_RATE)
                resultados[estrategia_id] = r.a_dict()
                break

    data = resultados.get(estrategia_id)
    if not data:
        return jsonify({"error": "Estrategia no encontrada"})

    # Generar trades para las flechas en el grafico
    est = next((e for e in ESTRATEGIAS if e["id"] == estrategia_id), None)
    if est:
        r, trades = simular_estrategia(ohlcv, pre, est, capital=INITIAL_EUR, fee_rate=FEE_RATE, return_trades=True)
        trades_data = []
        for t in trades:
            trades_data.append({
                "entry_price": t[0], "exit_price": t[1],
                "gain_pct": round(t[2], 2), "duration_h": t[3],
                "exit_type": t[4],
            })
        return jsonify({"resultado": data, "trades": trades_data})
    
    return jsonify({"resultado": data, "trades": []})


# ── HTML Template ──

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SIMON — Simulador de Estrategias</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0d1117; color: #c9d1d9; padding: 20px; }}
h1 {{ color: #58a6ff; margin-bottom: 10px; font-size: 24px; }}
.subtitle {{ color: #8b949e; margin-bottom: 20px; font-size: 14px; }}
.controls {{ display: flex; align-items: center; gap: 12px; margin-bottom: 20px; flex-wrap: wrap; }}
.controls label {{ color: #c9d1d9; font-size: 14px; }}
.controls select {{ background: #21262d; color: #c9d1d9; border: 1px solid #30363d; padding: 8px 12px; border-radius: 6px; font-size: 14px; min-width: 220px; cursor: pointer; }}
.controls select:focus {{ border-color: #58a6ff; outline: none; }}
.controls button {{ background: #238636; color: #fff; border: none; padding: 8px 20px; border-radius: 6px; font-size: 14px; cursor: pointer; font-weight: 500; }}
.controls button:hover {{ background: #2ea043; }}
.controls button:disabled {{ background: #21262d; color: #484f58; cursor: not-allowed; }}
#loading {{ color: #8b949e; font-size: 13px; margin-left: 10px; display: none; }}
.chart-container {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 15px; margin-bottom: 20px; }}
#chart {{ width: 100%; height: 700px; }}
#card-container {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px; }}
.card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; }}
.card h3 {{ color: #58a6ff; font-size: 14px; margin-bottom: 12px; border-bottom: 1px solid #21262d; padding-bottom: 8px; }}
.card .row {{ display: flex; justify-content: space-between; padding: 4px 0; font-size: 13px; }}
.card .label {{ color: #8b949e; }}
.card .value {{ color: #c9d1d9; font-weight: 500; }}
.card .positive {{ color: #3fb950; }}
.card .negative {{ color: #f85149; }}
.card .neutral {{ color: #d29922; }}
.card-highlight {{ border-color: #238636; }}
.empty-state {{ text-align: center; padding: 40px; color: #484f58; }}
.empty-state p {{ font-size: 16px; margin-bottom: 8px; }}
.empty-state .sub {{ font-size: 13px; }}
</style>
</head>
<body>

<h1>📊 SIMON</h1>
<p class="subtitle">Simulador de estrategias — <span id="info-velas">cargando...</span></p>

<div class="controls">
    <label for="estrategia">Estrategia:</label>
    <select id="estrategia">
        <option value="">-- Seleccionar --</option>
    </select>
    <button id="btn-generar" onclick="generar()">▶ Generar</button>
    <span id="loading">⏳ Simulando...</span>
</div>

<div class="chart-container">
    <div id="chart"></div>
</div>

<div id="card-container">
    <div class="empty-state" id="card-placeholder">
        <p>Selecciona una estrategia y pulsa "Generar"</p>
        <p class="sub">Se mostraran las operaciones en el grafico y los resultados aqui</p>
    </div>
</div>

<script>
let velasData = [];
let estrategiasList = [];
let chartLayout = {};

// Cargar velas al inicio
fetch('/api/velas')
    .then(r => r.json())
    .then(data => {{
        velasData = data.velas;
        document.getElementById('info-velas').textContent =
            data.total + ' velas 1h (' + data.dias + ' dias)';
        cargarEstrategias();
        dibujarVacio();
    }});

function cargarEstrategias() {{
    fetch('/api/estrategias')
        .then(r => r.json())
        .then(data => {{
            estrategiasList = data;
            const sel = document.getElementById('estrategia');
            data.forEach(e => {{
                const opt = document.createElement('option');
                opt.value = e.id;
                opt.textContent = e.nombre;
                sel.appendChild(opt);
            }});
        }});
}}

function dibujarVacio() {{
    const timestamps = velasData.map(v => new Date(v.ts));
    const closes = velasData.map(v => v.close);

    const candle = {{
        x: timestamps, open: velasData.map(v => v.open),
        high: velasData.map(v => v.high), low: velasData.map(v => v.low),
        close: closes, type: 'candlestick', name: 'BTC/EUR',
        yaxis: 'y', increasing: {{line: {{color: '#3fb950'}}}},
        decreasing: {{line: {{color: '#f85149'}}}},
    }};

    const volColors = velasData.map(v => v.close >= v.open ? '#3fb95044' : '#f8514944');
    const volume = {{
        x: timestamps, y: velasData.map(v => v.volume),
        type: 'bar', name: 'Volumen', yaxis: 'y3',
        marker: {{color: volColors}}, opacity: 0.4,
    }};

    const layout = {{
        paper_bgcolor: '#161b22', plot_bgcolor: '#161b22',
        font: {{color: '#c9d1d9', size: 11}},
        margin: {{l: 50, r: 20, b: 40, t: 20, pad: 0}},
        dragmode: 'zoom',
        hovermode: 'x unified',
        xaxis: {{type: 'date', showgrid: true, gridcolor: '#21262d', rangeslider: {{visible: false}}, domain: [0, 1]}},
        yaxis: {{title: 'Precio (€)', domain: [0.45, 1], showgrid: true, gridcolor: '#21262d', side: 'right'}},
        yaxis3: {{title: 'Volumen', domain: [0.35, 0.43], showgrid: false, side: 'right'}},
        legend: {{orientation: 'h', y: 1.02, x: 0, font: {{size: 10}}}},
        shapes: [],
        annotations: [],
    }};

    chartLayout = layout;
    Plotly.newPlot('chart', [candle, volume], layout, {{responsive: true, displayModeBar: false}});
}}

function generar() {{
    const sel = document.getElementById('estrategia');
    const id = sel.value;
    if (!id) {{ alert('Selecciona una estrategia'); return; }}

    document.getElementById('loading').style.display = 'inline';
    document.getElementById('btn-generar').disabled = true;

    fetch('/api/simular/' + id)
        .then(r => r.json())
        .then(data => {{
            document.getElementById('loading').style.display = 'none';
            document.getElementById('btn-generar').disabled = false;
            if (data.error) {{ alert(data.error); return; }}
            dibujarConTrades(data.resultado, data.trades);
            mostrarCard(data.resultado);
        }})
        .catch(err => {{
            document.getElementById('loading').style.display = 'none';
            document.getElementById('btn-generar').disabled = false;
            alert('Error: ' + err);
        }});
}}

function dibujarConTrades(resultado, trades) {{
    const timestamps = velasData.map(v => new Date(v.ts));
    const closes = velasData.map(v => v.close);
    const rsi = velasData.map(v => v.rsi);
    const macd = velasData.map(v => v.macd);
    const macdSig = velasData.map(v => v.macd_sig);
    const hist = velasData.map(v => v.hist);

    const volColors = velasData.map(v => v.close >= v.open ? '#3fb95044' : '#f8514944');

    // Candle
    const candle = {{
        x: timestamps, open: velasData.map(v => v.open),
        high: velasData.map(v => v.high), low: velasData.map(v => v.low),
        close: closes, type: 'candlestick', name: 'BTC/EUR',
        yaxis: 'y', increasing: {{line: {{color: '#3fb950'}}}},
        decreasing: {{line: {{color: '#f85149'}}}},
    }};

    // Volumen
    const volume = {{
        x: timestamps, y: velasData.map(v => v.volume),
        type: 'bar', name: 'Volumen', yaxis: 'y5',
        marker: {{color: volColors}}, opacity: 0.4,
    }};

    // RSI
    const rsiTrace = {{
        x: timestamps, y: rsi, type: 'scatter', mode: 'lines',
        name: 'RSI (14)', yaxis: 'y2', line: {{color: '#d29922', width: 1.5}},
    }};
    const rsi30 = {{
        x: [timestamps[0], timestamps[timestamps.length-1]], y: [30, 30],
        type: 'scatter', mode: 'lines', name: 'RSI 30', yaxis: 'y2',
        line: {{color: '#f8514988', width: 1, dash: 'dash'}},
        showlegend: true,
    }};
    const rsi70 = {{
        x: [timestamps[0], timestamps[timestamps.length-1]], y: [70, 70],
        type: 'scatter', mode: 'lines', name: 'RSI 70', yaxis: 'y2',
        line: {{color: '#3fb95088', width: 1, dash: 'dash'}},
        showlegend: true,
    }};

    // MACD
    const macdTrace = {{
        x: timestamps, y: macd, type: 'scatter', mode: 'lines',
        name: 'MACD', yaxis: 'y4', line: {{color: '#58a6ff', width: 1.5}},
    }};
    const macdSigTrace = {{
        x: timestamps, y: macdSig, type: 'scatter', mode: 'lines',
        name: 'Señal', yaxis: 'y4', line: {{color: '#f0883e', width: 1.5}},
    }};
    const histTrace = {{
        x: timestamps, y: hist, type: 'bar', name: 'Histograma',
        yaxis: 'y4',
        marker: {{color: hist.map(v => v >= 0 ? '#3fb95088' : '#f8514988')}},
        opacity: 0.5,
    }};

    // Flechas de trades
    const shapes = [];
    const annotations = [];

    trades.forEach((t, idx) => {{
        // Buscar indices en el timeline
        const buyTs = new Date(velasData[0].ts + idx * 3600000); // aproximacion
        // Buscar velas cercanas al precio de entrada/salida
        let buyIdx = -1, sellIdx = -1;
        for (let i = 0; i < velasData.length; i++) {{
            if (buyIdx < 0 && Math.abs(velasData[i].close - t.entry_price) / t.entry_price < 0.02) {{
                buyIdx = i;
            }}
            if (sellIdx < 0 && Math.abs(velasData[i].close - t.exit_price) / t.exit_price < 0.02) {{
                sellIdx = i;
            }}
        }}
        if (buyIdx < 0) buyIdx = Math.floor(idx * velasData.length / trades.length);
        if (sellIdx < 0) sellIdx = Math.min(buyIdx + Math.max(Math.round(t.duration_h), 1), velasData.length - 1);

        // Flecha verde de compra
        shapes.push({{
            type: 'line', xref: 'x', yref: 'y',
            x0: timestamps[buyIdx], y0: t.entry_price * 0.97,
            x1: timestamps[buyIdx], y1: t.entry_price * 1.005,
            line: {{color: '#3fb950', width: 2}},
        }});
        annotations.push({{
            x: timestamps[buyIdx], y: t.entry_price * 0.965,
            text: '▲ ' + t.gain_pct.toFixed(2) + '%',
            showarrow: false, font: {{color: '#3fb950', size: 9}},
            xanchor: 'center',
        }});

        // Flecha roja de venta
        shapes.push({{
            type: 'line', xref: 'x', yref: 'y',
            x0: timestamps[sellIdx], y0: t.exit_price * 1.03,
            x1: timestamps[sellIdx], y1: t.exit_price * 0.995,
            line: {{color: '#f85149', width: 2}},
        }});
        annotations.push({{
            x: timestamps[sellIdx], y: t.exit_price * 1.045,
            text: '▼',
            showarrow: false, font: {{color: '#f85149', size: 11}},
            xanchor: 'center',
        }});
    }});

    const layout = {{
        paper_bgcolor: '#161b22', plot_bgcolor: '#161b22',
        font: {{color: '#c9d1d9', size: 11}},
        margin: {{l: 50, r: 20, b: 40, t: 20, pad: 0}},
        dragmode: 'zoom',
        hovermode: 'x unified',
        xaxis: {{type: 'date', showgrid: true, gridcolor: '#21262d', rangeslider: {{visible: false}}, domain: [0, 1]}},
        yaxis: {{title: 'Precio (€)', domain: [0.45, 1], showgrid: true, gridcolor: '#21262d', side: 'right'}},
        yaxis2: {{domain: [0.30, 0.42], showgrid: true, gridcolor: '#21262d', side: 'right', range: [0, 100]}},
        yaxis4: {{domain: [0.15, 0.27], showgrid: true, gridcolor: '#21262d', side: 'right'}},
        yaxis5: {{domain: [0, 0.12], showgrid: false, side: 'right'}},
        legend: {{orientation: 'h', y: 1.02, x: 0, font: {{size: 10}}}},
        shapes: shapes,
        annotations: annotations,
    }};

    Plotly.newPlot('chart', [candle, volume, rsiTrace, rsi30, rsi70, histTrace, macdTrace, macdSigTrace], layout,
        {{responsive: true, displayModeBar: false}});
}}

function mostrarCard(resultado) {{
    const cont = document.getElementById('card-container');
    const fmt = (v) => {{
        if (v === null || v === undefined) return '-';
        if (typeof v === 'number') {{
            if (Math.abs(v) < 0.01) return '0.00';
            if (Number.isInteger(v)) return v.toString();
            return v.toFixed(2);
        }}
        return v;
    }};
    const cls = (v) => v > 0 ? 'positive' : (v < 0 ? 'negative' : 'neutral');

    cont.innerHTML = `
        <div class="card card-highlight">
            <h3>${resultado.estrategia || 'Estrategia'}</h3>
            <div class="row"><span class="label">Operaciones</span><span class="value">${resultado.total_ops} (${fmt(resultado.ops_por_mes)}/mes)</span></div>
            <div class="row"><span class="label">Win rate</span><span class="value ${cls(resultado.winrate)}">${fmt(resultado.winrate)}% (${resultado.ganadoras}G / ${resultado.perdedoras}P)</span></div>
            <div class="row"><span class="label">Ganancia media/op</span><span class="value ${cls(resultado.ganancia_media_por_op)}">${fmt(resultado.ganancia_media_por_op)}%</span></div>
            <div class="row"><span class="label">Ganadoras media</span><span class="value positive">${fmt(resultado.ganancia_media_ganadoras)}%</span></div>
            ${resultado.perdedoras > 0 ? `<div class="row"><span class="label">Perdedoras media</span><span class="value negative">${fmt(resultado.perdida_media_perdedoras)}%</span></div>` : ''}
            <div class="row"><span class="label">Mejor/Peor op</span><span class="value">${fmt(resultado.mejor_operacion)}% / ${fmt(resultado.peor_operacion)}%</span></div>
            <div class="row"><span class="label">Tiempo medio</span><span class="value">${fmt(resultado.tiempo_medio_h)}h (max ${fmt(resultado.tiempo_maximo_h)}h)</span></div>
            <div class="row"><span class="label">Comisiones</span><span class="value">${fmt(resultado.comisiones)}€</span></div>
        </div>
        <div class="card card-highlight">
            <h3>Rentabilidad</h3>
            <div class="row"><span class="label">PnL neto</span><span class="value ${cls(resultado.pnl_neto)}">${fmt(resultado.pnl_neto)}€ (bruto ${fmt(resultado.pnl_bruto)}€)</span></div>
            <div class="row"><span class="label">PnL por operacion</span><span class="value ${cls(resultado.pnl_por_operacion)}">${fmt(resultado.pnl_por_operacion)}€</span></div>
            <div class="row"><span class="label">PnL/dia</span><span class="value ${cls(resultado.pnl_diario)}">${fmt(resultado.pnl_diario)}€</span></div>
            <div class="row"><span class="label">PnL/mes</span><span class="value ${cls(resultado.pnl_mensual)}">${fmt(resultado.pnl_mensual)}€</span></div>
            <div class="row"><span class="label">ROI total</span><span class="value ${cls(resultado.roi_total)}">${fmt(resultado.roi_total)}%</span></div>
            <div class="row"><span class="label">ROI diario</span><span class="value ${cls(resultado.roi_diario)}">${fmt(resultado.roi_diario)}%</span></div>
            <div class="row"><span class="label">ROI mensual</span><span class="value ${cls(resultado.roi_mensual)}">${fmt(resultado.roi_mensual)}%</span></div>
            <div class="row"><span class="label">Max drawdown</span><span class="value negative">${fmt(resultado.max_drawdown)}%</span></div>
            <div class="row"><span class="label">Score</span><span class="value ${cls(resultado.score)}">${fmt(resultado.score)}</span></div>
        </div>
        <div class="card">
            <h3>Config</h3>
            <div style="font-size: 12px; color: #8b949e; white-space: pre-wrap;">${resultado.descripcion || '-'}</div>
        </div>
    `;
}}
</script>
</body>
</html>
"""


if __name__ == "__main__":
    import webbrowser
    print("📊 SIMON Frontend")
    print(f"   Velas: {DATA_FILE}")
    print(f"   Abre: http://localhost:8050")
    webbrowser.open("http://localhost:8050")
    app.run(host="0.0.0.0", port=8050, debug=False)
