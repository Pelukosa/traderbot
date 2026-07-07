#!/usr/bin/env python3
"""
Genera frontend HTML auto-contenido con grafico de velas + RSI + MACD + simulaciones.

Uso:
    uv run python scripts/frontend.py            # (en remoto)
    python scripts/frontend.py                   # (en local sin uv)

Genera frontend.html en la carpeta actual. Abrelo con doble click.

No necesita servidor, ni Flask, ni npm. Todo va en un solo HTML
con los datos de las velas y las simulaciones embebidos.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.simulador import (
    cargar_velas, precalcular, simular_estrategia, ESTRATEGIAS,
    INITIAL_EUR, FEE_RATE,
)


def generar_html() -> str:
    print("Cargando velas...")
    ohlcv = cargar_velas()
    print(f"  {len(ohlcv)} velas 1h ({len(ohlcv)/24:.1f} dias)")

    print("Precalculando indicadores...")
    pre = precalcular(ohlcv)

    # Preparar datos JSON
    closes = pre["closes"].tolist()
    highs = pre["highs"].tolist()
    lows = pre["lows"].tolist()
    opens = [c[1] for c in ohlcv]
    vols = pre["vols"].tolist()
    timestamps = [c[0] for c in ohlcv]

    rsi = [None if np.isnan(v) else round(v, 2) for v in pre["rsi_arr"]]
    macd = [None if np.isnan(v) else round(v, 2) for v in pre["macd_line"]]
    macd_sig = [None if np.isnan(v) else round(v, 2) for v in pre["macd_sig"]]
    hist = [None if np.isnan(v) else round(v, 2) for v in pre["hist_arr"]]

    velas_json = [{
        "t": timestamps[i], "o": round(opens[i], 2), "h": round(highs[i], 2),
        "l": round(lows[i], 2), "c": round(closes[i], 2), "v": round(vols[i], 4),
        "rsi": rsi[i], "macd": macd[i], "macd_sig": macd_sig[i], "hist": hist[i],
    } for i in range(len(ohlcv))]

    print("Simulando estrategias...")
    estrategias_json = []
    for est in ESTRATEGIAS:
        r, trades = simular_estrategia(
            ohlcv, pre, est, capital=INITIAL_EUR, fee_rate=FEE_RATE, return_trades=True
        )
        datos = r.a_dict()
        trades_json = [{
            "entry_price": round(t[0], 2), "exit_price": round(t[1], 2),
            "gain_pct": round(t[2], 2), "duration_h": round(t[3], 1),
            "exit_type": t[4],
        } for t in trades]
        estrategias_json.append({
            "id": est["id"],
            "nombre": est["nombre"],
            "resultado": datos,
            "trades": trades_json,
        })
        print(f"  ✓ {est['nombre']}: {datos['total_ops']} ops, "
              f"WR {datos['winrate']}%, ROI {datos['roi_mensual']}%")

    # Para encontrar indices de trades en las velas
    trades_idx = {}
    for est in ESTRATEGIAS:
        _, trades = simular_estrategia(
            ohlcv, pre, est, capital=INITIAL_EUR, fee_rate=FEE_RATE, return_trades=True
        )
        idxs = []
        for t in trades:
            buy_i, sell_i = -1, -1
            for i in range(len(ohlcv)):
                if buy_i < 0 and abs(closes[i] - t[0]) / t[0] < 0.015:
                    buy_i = i
                if sell_i < 0 and abs(closes[i] - t[1]) / t[1] < 0.015:
                    sell_i = i
            if buy_i < 0: buy_i = 0
            if sell_i < 0: sell_i = len(ohlcv) - 1
            idxs.append({"buy": buy_i, "sell": sell_i})
        trades_idx[est["id"]] = idxs

    HTML = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SIMON — Simulador de Estrategias</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
*{{margin:0;padding:0;box-sizing:border-box;}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#0d1117;color:#c9d1d9;padding:20px;}}
h1{{color:#58a6ff;margin-bottom:5px;font-size:22px;}}
.subtitle{{color:#8b949e;margin-bottom:15px;font-size:13px;}}
.controls{{display:flex;align-items:center;gap:10px;margin-bottom:15px;flex-wrap:wrap;}}
.controls select{{background:#21262d;color:#c9d1d9;border:1px solid #30363d;padding:8px 12px;border-radius:6px;font-size:13px;min-width:200px;}}
.controls select:focus{{border-color:#58a6ff;outline:none;}}
.controls button{{background:#238636;color:#fff;border:none;padding:8px 18px;border-radius:6px;font-size:13px;cursor:pointer;}}
.controls button:hover{{background:#2ea043;}}
#chart{{width:100%;height:600px;background:#161b22;border:1px solid #30363d;border-radius:8px;margin-bottom:15px;}}
#cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px;}}
.card{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:14px;}}
.card h3{{color:#58a6ff;font-size:13px;margin-bottom:10px;border-bottom:1px solid #21262d;padding-bottom:6px;}}
.card .row{{display:flex;justify-content:space-between;padding:3px 0;font-size:12px;}}
.card .l{{color:#8b949e;}}
.card .v{{font-weight:500;}}
.pos{{color:#3fb950;}}
.neg{{color:#f85149;}}
.neu{{color:#d29922;}}
.top{{border-color:#238636;}}
.empty{{text-align:center;padding:30px;color:#484f58;font-size:14px;}}
</style>
</head>
<body>

<h1>📊 SIMON</h1>
<p class="subtitle">{len(ohlcv)} velas 1h ({len(ohlcv)/24:.0f} dias)</p>

<div class="controls">
    <select id="sel">
        <option value="">-- Seleccionar estrategia --</option>
    </select>
    <button onclick="generar()">▶ Generar</button>
</div>

<div id="chart"></div>
<div id="cards"><div class="empty">Selecciona una estrategia y pulsa Generar</div></div>

<script>
const VELAS = {json.dumps(velas_json)};
const ESTRATEGIAS = {json.dumps(estrategias_json)};
const TRADES_IDX = {json.dumps(trades_idx)};

// Llenar selector
const sel = document.getElementById('sel');
ESTRATEGIAS.forEach(e => {{
    const o = document.createElement('option');
    o.value = e.id; o.textContent = e.nombre;
    sel.appendChild(o);
}});

// Dibujar grafico vacio
function dibujarVacio() {{
    const ts = VELAS.map(v => new Date(v.t));
    const cdl = {{x:ts,open:VELAS.map(v=>v.o),high:VELAS.map(v=>v.h),low:VELAS.map(v=>v.l),close:VELAS.map(v=>v.c),type:'candlestick',name:'BTC/EUR',yaxis:'y',increasing:{{line:{{color:'#3fb950'}}}},decreasing:{{line:{{color:'#f85149'}}}}}};
    Plotly.newPlot('chart',[cdl],{{paper_bgcolor:'#161b22',plot_bgcolor:'#161b22',font:{{color:'#c9d1d9',size:10}},margin:{{l:50,r:20,b:30,t:10}},dragmode:'zoom',hovermode:'x unified',xaxis:{{type:'date',gridcolor:'#21262d',rangeslider:{{visible:false}}}},yaxis:{{domain:[0.3,1],gridcolor:'#21262d',side:'right'}},yaxis2:{{domain:[0.15,0.27],gridcolor:'#21262d',side:'right'}},yaxis3:{{domain:[0,0.12],gridcolor:'#21262d',side:'right'}},legend:{{orientation:'h',y:1.02,x:0,font:{{size:9}}}}}},{{responsive:true,displayModeBar:false}});
}}

function generar() {{
    const id = sel.value;
    if (!id) return;
    const est = ESTRATEGIAS.find(e => e.id === id);
    if (!est) return;
    const r = est.resultado;
    const trades = est.trades;
    const idxs = TRADES_IDX[id] || [];

    const ts = VELAS.map(v => new Date(v.t));
    const cc = VELAS.map(v => v.c);
    const volColors = VELAS.map(v => v.c >= v.o ? '#3fb95044' : '#f8514944');

    // Traces
    const candle = {{x:ts,open:VELAS.map(v=>v.o),high:VELAS.map(v=>v.h),low:VELAS.map(v=>v.l),close:cc,type:'candlestick',name:'BTC/EUR',yaxis:'y',increasing:{{line:{{color:'#3fb950'}}}},decreasing:{{line:{{color:'#f85149'}}}}}};
    const volume = {{x:ts,y:VELAS.map(v=>v.v),type:'bar',name:'Volumen',yaxis:'y3',marker:{{color:volColors}},opacity:0.3}};
    const rsi = {{x:ts,y:VELAS.map(v=>v.rsi),type:'scatter',mode:'lines',name:'RSI(14)',yaxis:'y2',line:{{color:'#d29922',width:1.5}}}};
    const rsi30 = {{x:[ts[0],ts[ts.length-1]],y:[30,30],type:'scatter',mode:'lines',name:'RSI30',yaxis:'y2',line:{{color:'#f8514988',width:1,dash:'dash'}},showlegend:true}};
    const rsi70 = {{x:[ts[0],ts[ts.length-1]],y:[70,70],type:'scatter',mode:'lines',name:'RSI70',yaxis:'y2',line:{{color:'#3fb95088',width:1,dash:'dash'}},showlegend:true}};

    // Flechas de trades
    const shapes = [];
    const annots = [];
    idxs.forEach((ix,i)=>{{
        if (i >= trades.length) return;
        const t = trades[i];
        // Compra - flecha verde arriba
        shapes.push({{type:'line',xref:'x',yref:'y',x0:ts[ix.buy],y0:t.entry_price*0.97,x1:ts[ix.buy],y1:t.entry_price*1.005,line:{{color:'#3fb950',width:2}}}});
        annots.push({{x:ts[ix.buy],y:t.entry_price*0.965,text:'▲ '+t.gain_pct.toFixed(2)+'%',showarrow:false,font:{{color:'#3fb950',size:9}},xanchor:'center'}});
        // Venta - flecha roja abajo
        shapes.push({{type:'line',xref:'x',yref:'y',x0:ts[ix.sell],y0:t.exit_price*1.03,x1:ts[ix.sell],y1:t.exit_price*0.995,line:{{color:'#f85149',width:2}}}});
        annots.push({{x:ts[ix.sell],y:t.exit_price*1.045,text:'▼',showarrow:false,font:{{color:'#f85149',size:11}},xanchor:'center'}});
    }});

    Plotly.newPlot('chart',[candle,volume,rsi,rsi30,rsi70],{{
        paper_bgcolor:'#161b22',plot_bgcolor:'#161b22',font:{{color:'#c9d1d9',size:10}},
        margin:{{l:50,r:20,b:30,t:10}},dragmode:'zoom',hovermode:'x unified',
        xaxis:{{type:'date',gridcolor:'#21262d',rangeslider:{{visible:false}}}},
        yaxis:{{domain:[0.3,1],gridcolor:'#21262d',side:'right'}},
        yaxis2:{{domain:[0.15,0.27],gridcolor:'#21262d',side:'right',range:[0,100]}},
        yaxis3:{{domain:[0,0.12],gridcolor:'#21262d',side:'right'}},
        legend:{{orientation:'h',y:1.02,x:0,font:{{size:9}}}},
        shapes,annotations:annots,
    }},{{responsive:true,displayModeBar:false}});

    // Cards
    const fm = (v)=>{{
        if(v===null||v===undefined)return'-';
        if(typeof v==='number'){{if(Math.abs(v)<0.01)return'0.00';if(Number.isInteger(v))return v;return v.toFixed(2);}}
        return v;
    }};
    const cl = (v)=>v>0?'pos':(v<0?'neg':'neu');
    document.getElementById('cards').innerHTML=`
        <div class="card top">
            <h3>${{r.estrategia}}</h3>
            <div class="row"><span class="l">Operaciones</span><span class="v">${{r.total_ops}} (${{fm(r.ops_por_mes)}}/mes)</span></div>
            <div class="row"><span class="l">Win rate</span><span class="v ${{cl(r.winrate)}}">${{fm(r.winrate)}}% (${{r.ganadoras}}G/${{r.perdedoras}}P)</span></div>
            <div class="row"><span class="l">%/op media</span><span class="v ${{cl(r.ganancia_media_por_op)}}">${{fm(r.ganancia_media_por_op)}}%</span></div>
            <div class="row"><span class="l">Ganadoras media</span><span class="v pos">${{fm(r.ganancia_media_ganadoras)}}%</span></div>
            ${{r.perdedoras>0?'<div class=\"row\"><span class=\"l\">Perdedoras media</span><span class=\"v neg\">'+fm(r.perdida_media_perdedoras)+'%</span></div>':''}}
            <div class="row"><span class="l">Mejor / Peor</span><span class="v">${{fm(r.mejor_operacion)}}% / ${{fm(r.peor_operacion)}}%</span></div>
            <div class="row"><span class="l">Tiempo medio</span><span class="v">${{fm(r.tiempo_medio_h)}}h</span></div>
        </div>
        <div class="card top">
            <h3>Rentabilidad</h3>
            <div class="row"><span class="l">PnL neto</span><span class="v ${{cl(r.pnl_neto)}}">${{fm(r.pnl_neto)}}€</span></div>
            <div class="row"><span class="l">PnL/op</span><span class="v ${{cl(r.pnl_por_operacion)}}">${{fm(r.pnl_por_operacion)}}€</span></div>
            <div class="row"><span class="l">PnL/dia</span><span class="v ${{cl(r.pnl_diario)}}">${{fm(r.pnl_diario)}}€</span></div>
            <div class="row"><span class="l">PnL/mes</span><span class="v ${{cl(r.pnl_mensual)}}">${{fm(r.pnl_mensual)}}€</span></div>
            <div class="row"><span class="l">ROI mensual</span><span class="v ${{cl(r.roi_mensual)}}">${{fm(r.roi_mensual)}}%</span></div>
            <div class="row"><span class="l">ROI diario</span><span class="v ${{cl(r.roi_diario)}}">${{fm(r.roi_diario)}}%</span></div>
            <div class="row"><span class="l">Max drawdown</span><span class="v neg">${{fm(r.max_drawdown)}}%</span></div>
            <div class="row"><span class="l">Score</span><span class="v ${{cl(r.score)}}">${{fm(r.score)}}</span></div>
        </div>
        <div class="card">
            <h3>Config</h3>
            <div style="font-size:11px;color:#8b949e;">${{r.descripcion||'-'}}</div>
            <div style="margin-top:8px;font-size:11px;color:#8b949e;">
                Capital simulado: ${{r.capital}}€<br>
                ${{r.total_ops*2*0.4 > 0 ? 'Comisiones estimadas: '+fm(r.comisiones)+'€': ''}}
            </div>
        </div>
    `;
}}

dibujarVacio();
</script>
</body>
</html>"""

    return HTML


def main():
    html = generar_html()
    out = Path("frontend.html")
    out.write_text(html)
    print(f"\n✅ Generado {out.absolute()}")
    print("   Abrelo con doble click en tu navegador")


if __name__ == "__main__":
    main()
