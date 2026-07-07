/**
 * SIMON — Simulador de Estrategias (Frontend)
 *
 * Carga datos desde /data.json (generado por scripts/generar_datos.py)
 * y renderiza el gráfico de velas + RSI + MACD + trades.
 */

// ── Estado global ──
let velas = [];
let estrategias = [];

// ── Elementos DOM ──
const sel = document.getElementById("sel");
const btn = document.getElementById("btnGenerar");
const feeTakerEl = document.getElementById("feeTaker");
const capitalEl = document.getElementById("capital");
const subtitle = document.getElementById("subtitle");
const cardsEl = document.getElementById("cards");

// ── Helpers ──
const fm = (v) => {
  if (v === null || v === undefined) return "-";
  if (typeof v === "number") {
    if (Math.abs(v) < 0.01) return "0.00";
    if (Number.isInteger(v)) return String(v);
    return v.toFixed(2);
  }
  return v;
};

const cl = (v) => (v > 0 ? "pos" : v < 0 ? "neg" : "neu");

// ── Inicialización ──
btn.addEventListener("click", generar);

async function init() {
  try {
    const resp = await fetch("/data.json");
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();
    velas = data.velas;
    estrategias = data.estrategias;

    // Llenar select
    sel.innerHTML = '<option value="">-- Seleccionar estrategia --</option>';
    estrategias.forEach((e) => {
      const o = document.createElement("option");
      o.value = e.id;
      o.textContent = e.nombre;
      sel.appendChild(o);
    });

    subtitle.textContent = `${velas.length} velas 1h (${(velas.length / 24).toFixed(0)} dias) — Estrategias: ${estrategias.length}`;
    dibujarVacio();
  } catch (err) {
    console.error("Error cargando datos:", err);
    subtitle.textContent = "❌ Error al cargar data.json. Ejecuta: python scripts/generar_datos.py";
    subtitle.style.color = "#f85149";
  }
}

// ── Gráfico vacío (solo velas) ──
function dibujarVacio() {
  const ts = velas.map((v) => new Date(v.t));
  const cdl = {
    x: ts,
    open: velas.map((v) => v.o),
    high: velas.map((v) => v.h),
    low: velas.map((v) => v.l),
    close: velas.map((v) => v.c),
    type: "candlestick",
    name: "BTC/EUR",
    yaxis: "y",
    increasing: { line: { color: "#3fb950" } },
    decreasing: { line: { color: "#f85149" } },
  };

  Plotly.newPlot(
    "chart",
    [cdl],
    {
      paper_bgcolor: "#161b22",
      plot_bgcolor: "#161b22",
      font: { color: "#c9d1d9", size: 10 },
      margin: { l: 50, r: 20, b: 30, t: 10 },
      dragmode: "zoom",
      hovermode: "x unified",
      xaxis: { type: "date", gridcolor: "#21262d", rangeslider: { visible: true, thickness: 0.05 } },
      yaxis: { domain: [0.28, 1], gridcolor: "#21262d", side: "right", fixedrange: false },
      yaxis2: { domain: [0.14, 0.25], gridcolor: "#21262d", side: "right", range: [0, 100], fixedrange: false },
      yaxis3: { domain: [0, 0.11], gridcolor: "#21262d", side: "right", fixedrange: false },
      legend: { orientation: "h", y: 1.02, x: 0, font: { size: 9 } },
    },
    { responsive: true, scrollZoom: true, displayModeBar: true, modeBarButtonsToRemove: ["lasso2d", "select2d"], displaylogo: false },
  );
}

// ── Generar gráfico + cards ──
function generar() {
  const id = sel.value;
  if (!id) return;

  const est = estrategias.find((e) => e.id === id);
  if (!est) return;

  const r = est.resultado;
  const trades = est.trades;

  const ts = velas.map((v) => new Date(v.t));
  const volColors = velas.map((v) =>
    v.c >= v.o ? "#3fb95044" : "#f8514944",
  );

  // ── Puntos de compra/venta y líneas horizontales ──
  const buyTimes = [];
  const buyPrices = [];
  const buyTexts = [];
  const sellTimes = [];
  const sellPrices = [];
  const sellTexts = [];
  const shapes = [];

  trades.forEach((t) => {
    const bi = t.buy_idx;
    const si = t.sell_idx;
    if (bi < 0 || si < 0 || bi >= ts.length || si >= ts.length) return;

    buyTimes.push(ts[bi]);
    buyPrices.push(t.entry_price);
    buyTexts.push("Compra " + t.entry_price.toFixed(0) + "€");

    sellTimes.push(ts[si]);
    sellPrices.push(t.exit_price);
    sellTexts.push("Venta " + t.exit_price.toFixed(0) + "€ → " + (t.gain_pct >= 0 ? "+" : "") + t.gain_pct.toFixed(2) + "%");

    // Línea horizontal discontinua al nivel de entrada
    shapes.push({
      type: "line", xref: "x", yref: "y",
      x0: ts[bi], y0: t.entry_price,
      x1: ts[si], y1: t.entry_price,
      line: { color: "#3fb95066", width: 1, dash: "dot" },
    });
  });

  // ── Trazas ──
  const traces = [
    {
      x: ts,
      open: velas.map((v) => v.o),
      high: velas.map((v) => v.h),
      low: velas.map((v) => v.l),
      close: velas.map((v) => v.c),
      type: "candlestick",
      name: "BTC/EUR",
      yaxis: "y",
      increasing: { line: { color: "#3fb950" } },
      decreasing: { line: { color: "#f85149" } },
    },
    // Marcadores de compra (triángulo verde hacia arriba)
    {
      x: buyTimes,
      y: buyPrices,
      type: "scatter",
      mode: "markers+text",
      name: "Compra",
      yaxis: "y",
      marker: { symbol: "triangle-up", size: 12, color: "#3fb950", line: { width: 1, color: "#fff" } },
      text: buyTexts,
      textposition: "top center",
      textfont: { size: 9, color: "#3fb950" },
      showlegend: true,
    },
    // Marcadores de venta (triángulo rojo hacia abajo)
    {
      x: sellTimes,
      y: sellPrices,
      type: "scatter",
      mode: "markers+text",
      name: "Venta",
      yaxis: "y",
      marker: { symbol: "triangle-down", size: 12, color: "#f85149", line: { width: 1, color: "#fff" } },
      text: sellTexts,
      textposition: "bottom center",
      textfont: { size: 9, color: "#f85149" },
      showlegend: true,
    },
    {
      x: ts,
      y: velas.map((v) => v.v),
      type: "bar",
      name: "Volumen",
      yaxis: "y3",
      marker: { color: volColors },
      opacity: 0.3,
    },
    {
      x: ts,
      y: velas.map((v) => v.rsi),
      type: "scatter",
      mode: "lines",
      name: "RSI(14)",
      yaxis: "y2",
      line: { color: "#d29922", width: 1.5 },
    },
    {
      x: [ts[0], ts[ts.length - 1]],
      y: [30, 30],
      type: "scatter",
      mode: "lines",
      name: "RSI30",
      yaxis: "y2",
      line: { color: "#f8514988", width: 1, dash: "dash" },
      showlegend: true,
    },
    {
      x: [ts[0], ts[ts.length - 1]],
      y: [70, 70],
      type: "scatter",
      mode: "lines",
      name: "RSI70",
      yaxis: "y2",
      line: { color: "#3fb95088", width: 1, dash: "dash" },
      showlegend: true,
    },
    {
      x: ts,
      y: velas.map((v) => v.macd),
      type: "scatter",
      mode: "lines",
      name: "MACD",
      yaxis: "y3",
      line: { color: "#58a6ff", width: 1.5 },
    },
    {
      x: ts,
      y: velas.map((v) => v.macd_sig),
      type: "scatter",
      mode: "lines",
      name: "Señal",
      yaxis: "y3",
      line: { color: "#f0883e", width: 1.5 },
    },
    {
      x: ts,
      y: velas.map((v) => v.hist),
      type: "bar",
      name: "Histograma",
      yaxis: "y3",
      marker: {
        color: (() => {
          // Coloreado dinámico del histograma MACD:
          //   - Barras positivas: verde fuerte si crece, verde sutil si decrece
          //   - Barras negativas: rojo fuerte si cae más, rojo sutil si remonta
          const GREEN_STRONG = "#3fb950";
          const GREEN_FADE = "#3fb95033";
          const RED_STRONG = "#f85149";
          const RED_FADE = "#f8514933";
          return velas.map((v, i) => {
            if (v.hist === null || v.hist === undefined) return "transparent";
            const prev = i > 0 ? velas[i - 1].hist : null;
            if (v.hist >= 0) {
              // Territorio positivo
              if (prev === null) return GREEN_STRONG;
              return v.hist >= prev ? GREEN_STRONG : GREEN_FADE;
            } else {
              // Territorio negativo
              if (prev === null) return RED_STRONG;
              return v.hist <= prev ? RED_STRONG : RED_FADE;
            }
          });
        })(),
      },
    },
  ];

  Plotly.newPlot(
    "chart",
    traces,
    {
      paper_bgcolor: "#161b22",
      plot_bgcolor: "#161b22",
      font: { color: "#c9d1d9", size: 10 },
      margin: { l: 50, r: 20, b: 30, t: 10 },
      dragmode: "zoom",
      hovermode: "x unified",
      xaxis: { type: "date", gridcolor: "#21262d", rangeslider: { visible: true, thickness: 0.05 } },
      yaxis: { domain: [0.28, 1], gridcolor: "#21262d", side: "right", fixedrange: false },
      yaxis2: { domain: [0.14, 0.25], gridcolor: "#21262d", side: "right", range: [0, 100], fixedrange: false },
      yaxis3: { domain: [0, 0.11], gridcolor: "#21262d", side: "right", fixedrange: false },
      legend: { orientation: "h", y: 1.02, x: 0, font: { size: 9 } },
      shapes,
    },
    { responsive: true, scrollZoom: true, displayModeBar: true, modeBarButtonsToRemove: ["lasso2d", "select2d"], displaylogo: false },
  );

  // ── Cards con cálculo de comisiones ──
  const SIM_CAPITAL = 120; // capital con el que se ejecutó la simulación
  const CAPITAL = parseFloat(capitalEl.value) || 120;
  const capitalPorTrade = CAPITAL; // 100% del capital por operación
  const escala = CAPITAL / SIM_CAPITAL; // factor de escala para el PnL bruto
  const takerPct = parseFloat(feeTakerEl.value) || 0;

  // Calcular comisiones por trade (taker en entrada y salida)
  let totalComision = 0;
  let tradesNetos = 0;
  let ganadorasNetas = 0;
  let perdedorasNetas = 0;
  let sumaGananciaNeta = 0;
  let mejorNeta = -Infinity;
  let peorNeta = Infinity;

  trades.forEach((t) => {
    const entradaFeeEur = capitalPorTrade * (takerPct / 100);
    const salidaFeeEur = capitalPorTrade * (takerPct / 100);
    const comisionTrade = entradaFeeEur + salidaFeeEur;
    totalComision += comisionTrade;

    const gananciaBrutaEur = capitalPorTrade * (t.gain_pct / 100);
    const gananciaNetaEur = gananciaBrutaEur - comisionTrade;
    const gananciaNetaPct = (gananciaNetaEur / capitalPorTrade) * 100;

    sumaGananciaNeta += gananciaNetaPct;
    if (gananciaNetaPct >= 0) {
      ganadorasNetas++;
    } else {
      perdedorasNetas++;
    }
    tradesNetos++;
    if (gananciaNetaPct > mejorNeta) mejorNeta = gananciaNetaPct;
    if (gananciaNetaPct < peorNeta) peorNeta = gananciaNetaPct;
  });

  const pnlBrutoEscalado = r.pnl_neto * escala;
  const pnlNetoComision = pnlBrutoEscalado - totalComision;
  const pnlMensualComision = pnlNetoComision * (30 / r.dias_simulados);
  const roiMensualComision = (pnlNetoComision / CAPITAL) * 100;
  const roiDiarioComision = roiMensualComision / 30;
  const winrateNeto = tradesNetos > 0 ? (ganadorasNetas / tradesNetos) * 100 : 0;

  cardsEl.innerHTML = `
    <div class="card top">
      <h3>${r.estrategia}</h3>
      <div class="row"><span class="l">Operaciones</span><span class="v">${r.total_ops} (${fm(r.ops_por_mes)}/mes)</span></div>
      <div class="row"><span class="l">Win rate bruto</span><span class="v ${cl(r.winrate)}">${fm(r.winrate)}% (${r.ganadoras}G/${r.perdedoras}P)</span></div>
      <div class="row"><span class="l">Win rate neto</span><span class="v ${cl(winrateNeto)}">${fm(winrateNeto)}% (${ganadorasNetas}G/${perdedorasNetas}P)</span></div>
      <div class="row"><span class="l">%/op media bruta</span><span class="v ${cl(r.ganancia_media_por_op)}">${fm(r.ganancia_media_por_op)}%</span></div>
      <div class="row"><span class="l">%/op media neta</span><span class="v ${cl(sumaGananciaNeta/tradesNetos)}">${fm(sumaGananciaNeta/tradesNetos)}%</span></div>
      <div class="row"><span class="l">Ganadoras media</span><span class="v pos">${fm(r.ganancia_media_ganadoras)}%</span></div>
      ${r.perdedoras > 0 ? '<div class="row"><span class="l">Perdedoras media</span><span class="v neg">' + fm(r.perdida_media_perdedoras) + '%</span></div>' : ""}
      <div class="row"><span class="l">Mejor / Peor (neto)</span><span class="v">${fm(mejorNeta)}% / ${fm(peorNeta)}%</span></div>
      <div class="row"><span class="l">Tiempo medio</span><span class="v">${fm(r.tiempo_medio_h)}h</span></div>
    </div>
    <div class="card top">
      <h3>Rentabilidad</h3>
      <div class="row"><span class="l">PnL bruto</span><span class="v ${cl(pnlBrutoEscalado)}">${fm(pnlBrutoEscalado)}€</span></div>
      <div class="row"><span class="l">Comisiones</span><span class="v neg">-${fm(totalComision)}€</span></div>
      <div class="row" style="border-top:1px solid #21262d;margin-top:4px;padding-top:6px"><span class="l" style="font-weight:bold">PnL neto</span><span class="v ${cl(pnlNetoComision)}" style="font-weight:bold">${fm(pnlNetoComision)}€</span></div>
      <div class="row"><span class="l">PnL/mes (neto)</span><span class="v ${cl(pnlMensualComision)}">${fm(pnlMensualComision)}€</span></div>
      <div class="row"><span class="l">ROI mensual (neto)</span><span class="v ${cl(roiMensualComision)}">${fm(roiMensualComision)}%</span></div>
      <div class="row"><span class="l">ROI diario (neto)</span><span class="v ${cl(roiDiarioComision)}">${fm(roiDiarioComision)}%</span></div>
      <div class="row"><span class="l">Max drawdown</span><span class="v neg">${fm(r.max_drawdown)}%</span></div>
      <div class="row"><span class="l">Score</span><span class="v ${cl(r.score)}">${fm(r.score)}</span></div>
    </div>
    <div class="card">
      <h3>Config</h3>
      <div style="font-size:11px;color:#8b949e;white-space:pre-wrap;">${r.descripcion || "-"}</div>
      <div style="margin-top:8px;font-size:11px;color:#8b949e;">
        Capital: ${CAPITAL}€ — Inversión/op: ${fm(capitalPorTrade)}€<br>
        Comisión: ${fm(takerPct)}% (entrada + salida)<br>
        Comisiones totales: ${fm(totalComision)}€<br>
        ${r.total_ops} trades en ${fm(r.dias_simulados)} dias
      </div>
    </div>
  `;
}

// ── Arranque ──
init();
