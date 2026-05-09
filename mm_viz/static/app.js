/* global Chart, Plotly */

const TAIL = 500;
const WS_PING_MS = 20000;
const MAX_LOCAL = Math.min(
  2000,
  parseInt(
    new URLSearchParams(window.location.search).get("max") || "2000",
    10
  ) || 2000
);

const COLORS = [
  "#3b82f6",
  "#ef4444",
  "#a855f7",
  "#22c55e",
  "#000000",
  "#f97316",
  "#06b6d4",
  "#eab308",
];

const charts = {};
const order = [];
const byStep = new Map();
let lastFrame = null;
let followLatest = true;
let selectedStep = null;
let ws = null;
let pingTimer = null;

function el(id) {
  return document.getElementById(id);
}

/** Tab panels use ``display: none`` when inactive; Plotly would measure 0×0 and render blank. */
function isPanelVisible(panelId) {
  const p = el(panelId);
  return !!(p && p.classList.contains("active"));
}

function tail(arr, n = TAIL) {
  if (!Array.isArray(arr)) return [];
  return arr.slice(-n);
}

function indexLabels(n) {
  return Array.from({ length: n }, (_, i) => i);
}

function lineSeriesMapToDatasets(obj) {
  const entries = Object.entries(obj).filter(
    ([, v]) => Array.isArray(v) && v.length
  );
  const n = Math.max(
    0,
    ...entries.map(([, v]) => (Array.isArray(v) ? v.length : 0))
  );
  return {
    labels: indexLabels(n),
    datasets: entries.map(([name, data], i) => ({
      label: name,
      data: data.slice(-TAIL),
      borderColor: COLORS[i % COLORS.length],
      backgroundColor: "transparent",
      borderWidth: 1.2,
      pointRadius: 0,
      tension: 0,
    })),
  };
}

function destroyChart(id) {
  if (charts[id]) {
    charts[id].destroy();
    delete charts[id];
  }
}

function makeLineChart(canvasId, title, seriesMap) {
  const ctx = el(canvasId);
  if (!ctx) return;
  destroyChart(canvasId);
  const { labels, datasets } = lineSeriesMapToDatasets(seriesMap);
  if (!datasets.length) {
    return;
  }
  charts[canvasId] = new Chart(ctx, {
    type: "line",
    data: { labels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      plugins: { title: { display: !!title, text: title } },
      scales: {
        x: { display: true },
        y: { display: true },
      },
    },
  });
}

function rememberFrame(f) {
  const s = f && f.point ? f.point.step_counter : null;
  if (s == null) return;
  const isNew = !byStep.has(s);
  byStep.set(s, f);
  if (isNew) {
    order.push(s);
    while (order.length > MAX_LOCAL) {
      const old = order.shift();
      byStep.delete(old);
    }
  }
  rebuildStepSelect();
}

function rebuildStepSelect() {
  const sel = el("stepSelect");
  const was = selectedStep;
  sel.innerHTML = "";
  const opt0 = document.createElement("option");
  opt0.value = "";
  opt0.textContent = "Latest (live)";
  sel.appendChild(opt0);
  for (let i = order.length - 1; i >= 0; i--) {
    const s = order[i];
    const o = document.createElement("option");
    o.value = String(s);
    o.textContent = "Step " + s;
    sel.appendChild(o);
  }
  if (was != null && [...sel.options].some((o) => o.value === String(was))) {
    sel.value = String(was);
  }
}

function getDisplayFrame() {
  if (followLatest) return lastFrame;
  if (selectedStep != null && selectedStep !== "") {
    return byStep.get(Number(selectedStep)) || lastFrame;
  }
  return lastFrame;
}

function frameTitle(f) {
  if (!f) return "—";
  const ex = f.exchange || "";
  const sym = f.symbol || "";
  const sc = f.point && f.point.step_counter;
  return `${ex} ${sym} step=${sc}`;
}

const plotlyCfg = { responsive: true, displayModeBar: true, scrollZoom: true };

function updatePlotlyFigures(f) {
  if (typeof Plotly === "undefined" || typeof window.buildActionDepthPlotly !== "function") {
    return;
  }
  const divAd = el("plotActionDepth");
  const divLf = el("plotLof");
  if (!f) return;

  if (divAd) {
    try {
      const built = window.buildActionDepthPlotly(f);
      const mustDeferFirst =
        divAd.dataset.plotlyInit !== "1" && !isPanelVisible("t1");
      if (mustDeferFirst) {
        divAd.dataset.plotlyDeferred = "1";
      } else {
        divAd.dataset.plotlyDeferred = "0";
        if (divAd.dataset.plotlyInit !== "1") {
          Plotly.newPlot(divAd, built.data, built.layout, plotlyCfg).then(() => {
            divAd.dataset.plotlyInit = "1";
            requestAnimationFrame(() => {
              try {
                Plotly.Plots.resize(divAd);
              } catch {
                /* ignore */
              }
            });
          });
        } else {
          Plotly.react(divAd, built.data, built.layout, plotlyCfg);
          if (isPanelVisible("t1")) {
            requestAnimationFrame(() => {
              try {
                Plotly.Plots.resize(divAd);
              } catch {
                /* ignore */
              }
            });
          }
        }
      }
    } catch (err) {
      console.error("buildActionDepthPlotly / Plotly", err);
      divAd.innerHTML =
        '<p class="plot-empty">Action &amp; Depth plot error (see browser console).</p>';
    }
  }

  if (divLf) {
    try {
      const lf = window.buildLofPlotly(f);
      if (!lf) {
        Plotly.purge(divLf);
        divLf.dataset.plotlyInit = "0";
        divLf.innerHTML =
          '<p class="plot-empty">No LOF panels in this frame (need ≥2 steps for observed path and/or an LOF model).</p>';
      } else {
        const mustDeferFirst =
          divLf.dataset.plotlyInit !== "1" && !isPanelVisible("t2");
        if (mustDeferFirst) {
          divLf.dataset.plotlyDeferred = "1";
        } else {
          divLf.dataset.plotlyDeferred = "0";
          if (divLf.dataset.plotlyInit !== "1") {
            divLf.innerHTML = "";
            Plotly.newPlot(divLf, lf.data, lf.layout, plotlyCfg).then(() => {
              divLf.dataset.plotlyInit = "1";
              requestAnimationFrame(() => {
                try {
                  Plotly.Plots.resize(divLf);
                } catch {
                  /* ignore */
                }
              });
            });
          } else {
            Plotly.react(divLf, lf.data, lf.layout, plotlyCfg);
            if (isPanelVisible("t2")) {
              requestAnimationFrame(() => {
                try {
                  Plotly.Plots.resize(divLf);
                } catch {
                  /* ignore */
                }
              });
            }
          }
        }
      }
    } catch (err) {
      console.error("buildLofPlotly / Plotly", err);
      divLf.innerHTML =
        '<p class="plot-empty">LOF plot error (see browser console).</p>';
    }
  }
}

function renderAll() {
  const f = getDisplayFrame();
  el("frameTitle").textContent = frameTitle(f);
  if (!f) {
    el("pointJson").textContent = "—";
    return;
  }

  el("pointJson").textContent = JSON.stringify(f.point || {}, null, 2);
  const rs = f.runner_series || {};
  makeLineChart("cWealth", "Wealth", {
    wealth_quote: tail(rs.wealth_quote),
    wealth_tick_lot: tail(rs.wealth_tick_lot),
  });
  makeLineChart("cInv", "Inventory / cash", {
    inventory_lots: tail(rs.inventory_lots),
    cash_tick_lot: tail(rs.cash_tick_lot),
  });

  const pd = f.price_diag || {};
  const rep = pd.rep_prices || {};
  makeLineChart("cRep", "Rep prices", {
    buy_realized: tail(rep.buy_realized),
    buy_computed: tail(rep.buy_computed),
    sell_realized: tail(rep.sell_realized),
    sell_computed: tail(rep.sell_computed),
  });
  const sig = pd.residual_sigma || {};
  const mu = pd.residual_mu || {};
  makeLineChart("cSig", "Residual σ", {
    sigma_buy: tail(sig.buy),
    sigma_sell: tail(sig.sell),
  });
  makeLineChart("cMu", "Residual μ", {
    mu_buy: tail(mu.buy),
    mu_sell: tail(mu.sell),
  });

  const pe = pd.pred_rep_errors || {};
  makeLineChart("cPe", "Pred rep errors", {
    buy: tail(pe.buy),
    sell: tail(pe.sell),
    buy_prev: tail(pe.buy_prev),
    sell_prev: tail(pe.sell_prev),
  });

  const qq = f.volume_qq || {};
  el("qqMeta").textContent = JSON.stringify(
    {
      present: qq.present,
      reason: qq.reason,
      n_obs: qq.n_obs,
      n_sim_buy: qq.n_sim_buy,
      n_sim_sell: qq.n_sim_sell,
    },
    null,
    2
  );
  if (qq.present) {
    const buy = qq.buy || {};
    const sell = qq.sell || {};
    makeLineChart("cQqBuy", "QQ buy (model x vs obs y)", {
      buy_x_model: tail(buy.x || []),
      buy_y_obs: tail(buy.y || []),
    });
    makeLineChart("cQqSell", "QQ sell", {
      sell_x_model: tail(sell.x || []),
      sell_y_obs: tail(sell.y || []),
    });
  } else {
    destroyChart("cQqBuy");
    destroyChart("cQqSell");
  }

  const ad = f.hedge_diag || {};
  makeLineChart("cHedgeRef", "Ref prices", {
    main_buy: tail((ad.main_ref || {}).buy || []),
    main_sell: tail((ad.main_ref || {}).sell || []),
    hedge_buy: tail((ad.hedge_ref || {}).buy || []),
    hedge_sell: tail((ad.hedge_ref || {}).sell || []),
  });
  const ap = ad.hedge_params || {};
  makeLineChart("cHedgeParams", "Hedge δ params", {
    mu_buy: tail(ap.mu_buy),
    mu_sell: tail(ap.mu_sell),
    sigma_buy: tail(ap.sigma_buy),
    sigma_sell: tail(ap.sigma_sell),
  });
  const port = ad.hedge_portfolio || {};
  makeLineChart("cHedgePort", "Hedge portfolio", {
    hedge_inventory: tail(port.inventory),
    hedge_cash: tail(port.cash),
  });

  el("priceMeta").textContent = JSON.stringify(
    { present: pd.present },
    null,
    2
  );
  el("lofParams").textContent = JSON.stringify(f.lof_params || {}, null, 2);
  el("lossJson").textContent = JSON.stringify(f.loss_subplots || [], null, 2);
  el("rawFrame").textContent = JSON.stringify(f, null, 2);

  updatePlotlyFigures(f);
}

function connectWs() {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  ws = new WebSocket(proto + "//" + location.host + "/ws");
  el("connState").textContent = "connecting…";
  el("connState").className = "badge warn";

  ws.onopen = () => {
    el("connState").textContent = "connected";
    el("connState").className = "badge ok";
    ws.send("hi");
    if (pingTimer) clearInterval(pingTimer);
    pingTimer = setInterval(() => {
      if (ws && ws.readyState === WebSocket.OPEN) ws.send("ping");
    }, WS_PING_MS);
  };

  ws.onclose = () => {
    el("connState").textContent = "disconnected (retry in 2s)";
    el("connState").className = "badge bad";
    if (pingTimer) clearInterval(pingTimer);
    setTimeout(connectWs, 2000);
  };

  ws.onmessage = (ev) => {
    let msg;
    try {
      msg = JSON.parse(ev.data);
    } catch {
      return;
    }
    if (msg.type === "frame" && msg.frame) {
      lastFrame = msg.frame;
      rememberFrame(lastFrame);
      if (followLatest) renderAll();
      else if (
        selectedStep != null &&
        String(msg.frame.point?.step_counter) === String(selectedStep)
      ) {
        renderAll();
      }
    }
    if (msg.type === "frame_delta" && msg.patch != null) {
      if (!lastFrame || typeof applyDiagnosticsPatch !== "function") {
        return;
      }
      lastFrame = applyDiagnosticsPatch(lastFrame, msg.patch);
      rememberFrame(lastFrame);
      const step = lastFrame.point?.step_counter;
      if (followLatest) renderAll();
      else if (selectedStep != null && String(step) === String(selectedStep)) {
        renderAll();
      }
    }
    if (msg.type === "history_meta" && msg.steps) {
      rebuildStepSelect();
    }
  };
}

function setupTabs() {
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document
        .querySelectorAll(".tab-btn")
        .forEach((b) => b.classList.remove("active"));
      document
        .querySelectorAll(".tab-panel")
        .forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      const id = btn.getAttribute("data-tab");
      el(id).classList.add("active");
      // Panel is visible now; create deferred Plotly charts or resize after layout.
      updatePlotlyFigures(getDisplayFrame());
      if (typeof Plotly !== "undefined") {
        requestAnimationFrame(() => {
          requestAnimationFrame(() => {
            const pA = el("plotActionDepth");
            const pL = el("plotLof");
            try {
              if (pA?.dataset.plotlyInit === "1") Plotly.Plots.resize(pA);
              if (pL?.dataset.plotlyInit === "1") Plotly.Plots.resize(pL);
            } catch {
              /* ignore */
            }
          });
        });
      }
    });
  });
}

window.addEventListener("resize", () => {
  if (typeof Plotly === "undefined") return;
  const pA = el("plotActionDepth");
  const pL = el("plotLof");
  try {
    if (pA?.dataset.plotlyInit === "1") Plotly.Plots.resize(pA);
    if (pL?.dataset.plotlyInit === "1") Plotly.Plots.resize(pL);
  } catch {
    /* ignore */
  }
});

async function loadConfig() {
  try {
    const r = await fetch("/api/config");
    const c = await r.json();
    el("configHint").textContent = `${c.ui_backend || "mm_viz"} · Redis ${c.redis_url} · channel ${c.channel} · latest key ${c.latest_key} · server buffer ${c.max_frames} frames`;
  } catch {
    el("configHint").textContent = "";
  }
}

async function bootstrapHttp() {
  try {
    const r = await fetch("/api/latest");
    if (r.ok) {
      lastFrame = await r.json();
      rememberFrame(lastFrame);
      renderAll();
    }
  } catch {
    /* empty */
  }
}

el("stepSelect").disabled = el("followLatest").checked;

el("followLatest").addEventListener("change", (e) => {
  followLatest = e.target.checked;
  if (followLatest) selectedStep = null;
  el("stepSelect").disabled = followLatest;
  renderAll();
});

el("stepSelect").addEventListener("change", (e) => {
  selectedStep = e.target.value === "" ? null : e.target.value;
  if (selectedStep != null) followLatest = false;
  el("followLatest").checked = followLatest;
  renderAll();
});

setupTabs();
loadConfig();
bootstrapHttp().then(() => {
  connectWs();
  renderAll();
});
