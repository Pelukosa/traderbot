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
      xaxis: { type: "date", gridcolor: "#21262d", rangeslider: { visible: false } },
      yaxis: { domain: [0.28, 1], gridcolor: "#21262d", side: "right" },
      yaxis2: { domain: [0.14, 0.25], gridcolor: "#21262d", side: "right", range: [0, 100] },
      yaxis3: { domain: [0, 0.11], gridcolor: "#21262d", side: "right" },
      legend: { orientation: "h", y: 1.02, x: 0, font: { size: 9 } },
    },
    { responsive: true, displayModeBar: false },
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

  // ── Shapes y anotaciones de trades ──
  const shapes = [];
  const annots = [];

  trades.forEach((t) => {
    const bi = t.buy_idx;
    const si = t.sell_idx;
    if (bi < 0 || si < 0 || bi >= ts.length || si >= ts.length) return;

    // Flecha compra (verde hacia arriba)
    shapes.push({
      type: "line",
      xref: "x",
      yref: "y",
      x0: ts[bi],
      y0: t.entry_price * 0.97,
      x1: ts[bi],
      y1: t.entry_price * 1.005,
      line: { color: "#3fb950", width: 2 },
    });
    annots.push({
      x: ts[bi],
      y: t.entry_price * 0.955,
      text: "COMPRA " + t.gain_pct.toFixed(2) + "%",
      showarrow: false,
      font: { color: "#3fb950", size: 9 },
      xanchor: "center",
    });

    // Flecha venta (roja hacia abajo)
    shapes.push({
      type: "line",
      xref: "x",
      yref: "y",
      x0: ts[si],
      y0: t.exit_price * 1.03,
      x1: ts[si],
      y1: t.exit_price * 0.995,
      line: { color: "#f85149", width: 2 },
    });
    annots.push({
      x: ts[si],
      y: t.exit_price * 1.045,
      text: "VENTA (" + t.exit_type + ")",
      showarrow: false,
      font: { color: "#f85149", size: 9 },
      xanchor: "center",
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
        color: velas.map((v) =>
          v.hist && v.hist >= 0 ? "#3fb95066" : "#f8514966",
        ),
      },
      opacity: 0.4,
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
      xaxis: { type: "date", gridcolor: "#21262d", rangeslider: { visible: false } },
      yaxis: { domain: [0.28, 1], gridcolor: "#21262d", side: "right" },
      yaxis2: { domain: [0.14, 0.25], gridcolor: "#21262d", side: "right", range: [0, 100] },
      yaxis3: { domain: [0, 0.11], gridcolor: "#21262d", side: "right" },
      legend: { orientation: "h", y: 1.02, x: 0, font: { size: 9 } },
      shapes,
      annotations: annots,
    },
    { responsive: true, displayModeBar: false },
  );

  // ── Cards ──
  cardsEl.innerHTML = `
    <div class="card top">
      <h3>${r.estrategia}</h3>
      <div class="row"><span class="l">Operaciones</span><span class="v">${r.total_ops} (${fm(r.ops_por_mes)}/mes)</span></div>
      <div class="row"><span class="l">Win rate</span><span class="v ${cl(r.winrate)}">${fm(r.winrate)}% (${r.ganadoras}G/${r.perdedoras}P)</span></div>
      <div class="row"><span class="l">%/op media</span><span class="v ${cl(r.ganancia_media_por_op)}">${fm(r.ganancia_media_por_op)}%</span></div>
      <div class="row"><span class="l">Ganadoras media</span><span class="v pos">${fm(r.ganancia_media_ganadoras)}%</span></div>
      ${r.perdedoras > 0 ? '<div class="row"><span class="l">Perdedoras media</span><span class="v neg">' + fm(r.perdida_media_perdedoras) + '%</span></div>' : ""}
      <div class="row"><span class="l">Mejor / Peor</span><span class="v">${fm(r.mejor_operacion)}% / ${fm(r.peor_operacion)}%</span></div>
      <div class="row"><span class="l">Tiempo medio</span><span class="v">${fm(r.tiempo_medio_h)}h</span></div>
    </div>
    <div class="card top">
      <h3>Rentabilidad</h3>
      <div class="row"><span class="l">PnL neto</span><span class="v ${cl(r.pnl_neto)}">${fm(r.pnl_neto)}€</span></div>
      <div class="row"><span class="l">PnL/op</span><span class="v ${cl(r.pnl_por_operacion)}">${fm(r.pnl_por_operacion)}€</span></div>
      <div class="row"><span class="l">PnL/dia</span><span class="v ${cl(r.pnl_diario)}">${fm(r.pnl_diario)}€</span></div>
      <div class="row"><span class="l">PnL/mes</span><span class="v ${cl(r.pnl_mensual)}">${fm(r.pnl_mensual)}€</span></div>
      <div class="row"><span class="l">ROI mensual</span><span class="v ${cl(r.roi_mensual)}">${fm(r.roi_mensual)}%</span></div>
      <div class="row"><span class="l">ROI diario</span><span class="v ${cl(r.roi_diario)}">${fm(r.roi_diario)}%</span></div>
      <div class="row"><span class="l">Max drawdown</span><span class="v neg">${fm(r.max_drawdown)}%</span></div>
      <div class="row"><span class="l">Score</span><span class="v ${cl(r.score)}">${fm(r.score)}</span></div>
    </div>
    <div class="card">
      <h3>Config</h3>
      <div style="font-size:11px;color:#8b949e;white-space:pre-wrap;">${r.descripcion || "-"}</div>
      <div style="margin-top:8px;font-size:11px;color:#8b949e;">
        Capital: 120€<br>
        Comisiones: ${fm(r.comisiones)}€<br>
        ${r.total_ops} trades en ${r.dias_simulados} dias
      </div>
    </div>
  `;
}

// ── Arranque ──
init();
