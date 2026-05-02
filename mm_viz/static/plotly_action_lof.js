/**
 * Plotly traces/layout mirroring ``mm_viz/mpl_figures.py`` (Action & Depth, LOF).
 * Used for live updates without server-rendered PNGs.
 */

/* global Plotly */

function num(v) {
  const x = Number(v);
  return Number.isFinite(x) ? x : NaN;
}

function finiteSeriesYlim(seriesList, padMin = 0.5, padFrac = 0.1, defLo = -1, defHi = 1) {
  const vals = [];
  for (const ser of seriesList) {
    if (!Array.isArray(ser)) continue;
    for (const x of ser) {
      const v = num(x);
      if (Number.isFinite(v)) vals.push(v);
    }
  }
  if (!vals.length) return [defLo, defHi];
  let lo = Math.min(...vals);
  let hi = Math.max(...vals);
  const span = hi - lo;
  const pad = span > 0 ? Math.max(span * padFrac, padMin) : Math.max(Math.abs(lo) * padFrac, padMin);
  return [lo - pad, hi + pad];
}

/** Five stacked row domains (top row = J / index 0). */
function rowDomains5(gap = 0.025) {
  const cell = (1 - 4 * gap) / 5;
  const out = [];
  for (let i = 0; i < 5; i++) {
    const y1 = 1 - i * (cell + gap);
    const y0 = y1 - cell;
    out.push([y0, y1]);
  }
  return out;
}

function rowDomains2(gap = 0.06) {
  const cell = (1 - gap) / 2;
  return [
    [cell + gap, 1],
    [0, cell],
  ];
}

/** Trace / xref subplot name: x, y, x2, y2, … */
function axisName(i, axis) {
  if (i === 0) return axis;
  return axis + (i + 1);
}

/** Plotly *layout* key for subplot i: xaxis, yaxis, xaxis2, yaxis2, … (not x / y). */
function layoutXAxisKey(i) {
  return i === 0 ? "xaxis" : `xaxis${i + 1}`;
}
function layoutYAxisKey(i) {
  return i === 0 ? "yaxis" : `yaxis${i + 1}`;
}

function scatterLine(tr, subplotIndex) {
  const xa = axisName(subplotIndex, "x");
  const ya = axisName(subplotIndex, "y");
  return { ...tr, xaxis: xa, yaxis: ya };
}

/**
 * @returns {{ data: object[], layout: object }}
 */
function buildActionDepthPlotly(frame) {
  const domains = rowDomains5();
  const data = [];
  const layout = {
    height: 920,
    margin: { l: 56, r: 16, t: 48, b: 36 },
    paper_bgcolor: "#161b22",
    plot_bgcolor: "#0d1117",
    font: { color: "#e6edf3", size: 10 },
    showlegend: true,
    legend: {
      font: { size: 8 },
      orientation: "h",
      y: 1.02,
      x: 0,
      xanchor: "left",
    },
    title: { text: "Action & Depth diagnostics (Plotly)", font: { size: 13 } },
    shapes: [],
    annotations: [],
  };

  for (let i = 0; i < 5; i++) {
    const sx = axisName(i, "x");
    const sy = axisName(i, "y");
    const lx = layoutXAxisKey(i);
    const ly = layoutYAxisKey(i);
    layout[lx] = {
      domain: [0.12, 0.99],
      anchor: sy,
      showgrid: true,
      gridcolor: "#30363d",
      zeroline: false,
      ...(i < 2 ? {} : { title: { text: "Spread from mid (ticks)", font: { size: 9 } } }),
    };
    layout[ly] = {
      domain: domains[i],
      anchor: sx,
      showgrid: true,
      gridcolor: "#30363d",
      zeroline: false,
    };
  }

  const rs = (frame && frame.runner_series) || {};
  const jc = rs.J_components || {};
  const rw = rs.rewards || {};

  // Row 0 — J components
  const yMean = Array.isArray(jc.mean) ? jc.mean.map(num) : [];
  const yInv = Array.isArray(jc.inv) ? jc.inv.map(num) : [];
  const yStd = Array.isArray(jc.std) ? jc.std.map(num) : [];
  const yUni = Array.isArray(jc.uniformity) ? jc.uniformity.map(num) : [];
  const ySpr = Array.isArray(jc.spread) ? jc.spread.map(num) : [];
  const nComp = Math.min(
    yMean.length,
    yInv.length,
    yStd.length,
    yUni.length,
    ySpr.length
  );

  const yMi = Array.isArray(jc.mean_init) ? jc.mean_init.map(num) : [];
  const yIi = Array.isArray(jc.inv_init) ? jc.inv_init.map(num) : [];
  const ySi = Array.isArray(jc.std_init) ? jc.std_init.map(num) : [];
  const yUi = Array.isArray(jc.uniformity_init) ? jc.uniformity_init.map(num) : [];
  const ySpi = Array.isArray(jc.spread_init) ? jc.spread_init.map(num) : [];
  const nInit = Math.min(nComp, yMi.length, yIi.length, ySi.length, yUi.length, ySpi.length);

  if (nComp > 0) {
    const t = Array.from({ length: nComp }, (_, j) => j);
    const add = (yy, color, name, dash) => {
      data.push(
        scatterLine(
          {
            type: "scatter",
            mode: "lines",
            name,
            x: t,
            y: yy.slice(0, nComp),
            line: { color, width: 1.2, dash: dash || "solid" },
            legendgroup: name,
          },
          0
        )
      );
    };
    add(yMean, "blue", "reward_coef * J_mean (opt)");
    add(yInv, "red", "-inv_coef * J_inv_penalty (opt)");
    add(yStd, "purple", "-risk_coef * J_std (opt)");
    add(yUni, "green", "-uniformity_coef * J_uniformity (opt)");
    add(ySpr, "black", "-spread_coef * J_spread (opt)");
    if (nInit > 0) {
      const ti = Array.from({ length: nInit }, (_, j) => j);
      const addI = (yy, color, name) => {
        data.push(
          scatterLine(
            {
              type: "scatter",
              mode: "lines",
              name,
              x: ti,
              y: yy.slice(0, nInit),
              line: { color, width: 1.1, dash: "dot" },
            },
            0
          )
        );
      };
      addI(yMi, "blue", "reward_coef * J_mean (init)");
      addI(yIi, "red", "-inv_coef * J_inv_penalty (init)");
      addI(ySi, "purple", "-risk_coef * J_std (init)");
      addI(yUi, "green", "-uniformity_coef * J_uniformity (init)");
      addI(ySpi, "black", "-spread_coef * J_spread (init)");
    }
    layout.yaxis.title = { text: "Objective components", font: { size: 10 } };
    layout.xaxis.title = { text: "Decision index", font: { size: 10 } };
    const allY = []
      .concat(yMean.slice(0, nComp), yInv.slice(0, nComp), yStd.slice(0, nComp), yUni.slice(0, nComp), ySpr.slice(0, nComp));
    if (nInit > 0) {
      allY.push(
        ...yMi.slice(0, nInit),
        ...yIi.slice(0, nInit),
        ...ySi.slice(0, nInit),
        ...yUi.slice(0, nInit),
        ...ySpi.slice(0, nInit)
      );
    }
    const [lo, hi] = finiteSeriesYlim([allY], 1.0, 0.1);
    layout.yaxis.range = [lo, hi];
    layout.xaxis.range = nComp > 1 ? [0, nComp - 1] : [0, 1];
  } else {
    layout.xaxis.range = [0, 1];
    layout.yaxis.range = [-1, 1];
  }

  layout.shapes.push({
    type: "line",
    xref: "x",
    yref: "y",
    x0: layout.xaxis.range[0],
    x1: layout.xaxis.range[1],
    y0: 0,
    y1: 0,
    line: { color: "gray", width: 1 },
    layer: "below",
  });

  // Row 1 — rewards
  const mainOpt = Array.isArray(rw.main_opt) ? rw.main_opt.map(num) : [];
  const anchorOpt = Array.isArray(rw.anchor_opt) ? rw.anchor_opt.map(num) : [];
  const mainInit = Array.isArray(rw.main_init) ? rw.main_init.map(num) : [];
  const anchorInit = Array.isArray(rw.anchor_init) ? rw.anchor_init.map(num) : [];
  const nOpt = Math.min(mainOpt.length, anchorOpt.length);
  if (nOpt > 0) {
    const t = Array.from({ length: nOpt }, (_, j) => j);
    data.push(
      scatterLine(
        {
          type: "scatter",
          mode: "lines",
          name: "Main reward (opt)",
          x: t,
          y: mainOpt.slice(0, nOpt),
          line: { color: "blue", width: 1.2 },
        },
        1
      ),
      scatterLine(
        {
          type: "scatter",
          mode: "lines",
          name: "Anchor reward (opt)",
          x: t,
          y: anchorOpt.slice(0, nOpt),
          line: { color: "orange", width: 1.2 },
        },
        1
      )
    );
    const nI = Math.min(mainInit.length, anchorInit.length);
    if (nI > 0) {
      const ti = Array.from({ length: nI }, (_, j) => j);
      data.push(
        scatterLine(
          {
            type: "scatter",
            mode: "lines",
            name: "Main reward (init)",
            x: ti,
            y: mainInit.slice(0, nI),
            line: { color: "blue", width: 1.1, dash: "dot" },
          },
          1
        ),
        scatterLine(
          {
            type: "scatter",
            mode: "lines",
            name: "Anchor reward (init)",
            x: ti,
            y: anchorInit.slice(0, nI),
            line: { color: "orange", width: 1.1, dash: "dot" },
          },
          1
        )
      );
    }
    layout.yaxis2.title = { text: "Reward (tick×lot)", font: { size: 10 } };
    layout.xaxis2.title = { text: "Decision index", font: { size: 10 } };
    const ally = mainOpt.slice(0, nOpt).concat(anchorOpt.slice(0, nOpt));
    if (nI > 0) ally.push(...mainInit.slice(0, nI), ...anchorInit.slice(0, nI));
    const [lo, hi] = finiteSeriesYlim([ally], 1.0, 0.1);
    layout.yaxis2.range = [lo, hi];
    layout.xaxis2.range = nOpt > 1 ? [0, nOpt - 1] : [0, 1];
  } else {
    layout.xaxis2.range = [0, 1];
    layout.yaxis2.range = [-1, 1];
  }
  layout.shapes.push({
    type: "line",
    xref: "x2",
    yref: "y2",
    x0: layout.xaxis2.range[0],
    x1: layout.xaxis2.range[1],
    y0: 0,
    y1: 0,
    line: { color: "gray", width: 1 },
    layer: "below",
  });

  const depth = (frame && frame.depth) || {};
  const xMax = num(depth.x_max) || 1;
  const pad = xMax * 0.05;
  const xlim = [-xMax - pad, xMax + pad];

  const b = depth.bid || {};
  const a = depth.ask || {};
  const xb = Float64Array.from(b.x || [0]);
  const yb = Float64Array.from(b.y || [0]);
  const xa = Float64Array.from(a.x || [0]);
  const ya = Float64Array.from(a.y || [0]);

  if (xb.length >= 2) {
    data.push(
      scatterLine(
        {
          type: "scatter",
          mode: "lines",
          name: "Bid depth",
          x: Array.from(xb),
          y: Array.from(yb),
          line: { color: "green", width: 1.5, shape: "hv" },
        },
        2
      )
    );
  }
  if (xa.length >= 2) {
    data.push(
      scatterLine(
        {
          type: "scatter",
          mode: "lines",
          name: "Ask depth",
          x: Array.from(xa),
          y: Array.from(ya),
          line: { color: "red", width: 1.5, shape: "hv" },
        },
        2
      )
    );
  }

  const fit = depth.fit || {};
  if (fit.present && fit.bid && typeof fit.bid === "object") {
    const fxb = Float64Array.from(fit.bid.x || []);
    const fyb = Float64Array.from(fit.bid.y || []);
    if (fxb.length && fyb.length) {
      data.push(
        scatterLine(
          {
            type: "scatter",
            mode: "lines",
            name: "Bid fit",
            x: Array.from(fxb),
            y: Array.from(fyb),
            line: { color: "green", width: 1.2, dash: "dash" },
          },
          2
        )
      );
    }
  }
  if (fit.present && fit.ask && typeof fit.ask === "object") {
    const fxa = Float64Array.from(fit.ask.x || []);
    const fya = Float64Array.from(fit.ask.y || []);
    if (fxa.length && fya.length) {
      data.push(
        scatterLine(
          {
            type: "scatter",
            mode: "lines",
            name: "Ask fit",
            x: Array.from(fxa),
            y: Array.from(fya),
            line: { color: "red", width: 1.2, dash: "dash" },
          },
          2
        )
      );
    }
  }

  layout.yaxis3.title = { text: "Cumulative volume", font: { size: 10 } };
  const yMaxD = Math.max(
    yb.length > 0 ? Math.max(...Array.from(yb)) : 0,
    ya.length > 0 ? Math.max(...Array.from(ya)) : 0
  );
  layout.yaxis3.range = [0, yMaxD > 0 ? yMaxD * 1.1 : 1];
  layout.xaxis3.range = xlim;
  layout.shapes.push({
    type: "line",
    xref: "x3",
    yref: "y3",
    x0: 0,
    x1: 0,
    y0: 0,
    y1: 1,
    line: { color: "gray", width: 1 },
    layer: "below",
  });

  const ad = (frame && frame.action_depth) || {};
  let hasData = false;
  if (ad.present) {
    const x_b = Float64Array.from(ad.x_bid || []);
    const q_b = Float64Array.from(ad.q_bid || []);
    const x_a = Float64Array.from(ad.x_ask || []);
    const q_a = Float64Array.from(ad.q_ask || []);
    const cum_b = Float64Array.from(ad.cum_bid || []);
    const cum_a = Float64Array.from(ad.cum_ask || []);
    hasData = x_b.length > 0 || q_a.length > 0;

    if (hasData) {
      if (x_b.length > 0) {
        const xs = [x_b[0] - 0.5, ...Array.from(x_b)];
        const ys = [0, ...Array.from(q_b)];
        data.push(
          scatterLine(
            {
              type: "scatter",
              mode: "lines",
              name: "Bid (marginal)",
              x: xs,
              y: ys,
              line: { color: "green", width: 1.5, shape: "hv" },
            },
            3
          )
        );
      }
      if (x_a.length > 0) {
        const xs = [x_a[0] - 0.5, ...Array.from(x_a)];
        const ys = [0, ...Array.from(q_a)];
        data.push(
          scatterLine(
            {
              type: "scatter",
              mode: "lines",
              name: "Ask (marginal)",
              x: xs,
              y: ys,
              line: { color: "red", width: 1.5, shape: "hv" },
            },
            3
          )
        );
      }
      const yMaxM = Math.max(
        q_b.length ? Math.max(...q_b) : 0,
        q_a.length ? Math.max(...q_a) : 0
      );
      layout.yaxis4.range = [0, yMaxM > 0 ? yMaxM * 1.1 : 1];
    } else {
      layout.yaxis4.range = [0, 1];
    }
    layout.xaxis4.range = xlim;
    layout.yaxis4.title = { text: "Marginal volume", font: { size: 10 } };
    layout.shapes.push({
      type: "line",
      xref: "x4",
      yref: "y4",
      x0: 0,
      x1: 0,
      y0: 0,
      y1: 1,
      line: { color: "gray", width: 1 },
      layer: "below",
    });

    if (hasData) {
      if (x_b.length > 0 && cum_b.length > 0) {
        const xs = [x_b[0] - 0.5, ...Array.from(x_b)];
        const ys = [0, ...Array.from(cum_b)];
        data.push(
          scatterLine(
            {
              type: "scatter",
              mode: "lines",
              name: "Bid (cum)",
              x: xs,
              y: ys,
              line: { color: "green", width: 1.5, shape: "hv" },
            },
            4
          )
        );
      }
      if (x_a.length > 0 && cum_a.length > 0) {
        const xs = [x_a[0] - 0.5, ...Array.from(x_a)];
        const ys = [0, ...Array.from(cum_a)];
        data.push(
          scatterLine(
            {
              type: "scatter",
              mode: "lines",
              name: "Ask (cum)",
              x: xs,
              y: ys,
              line: { color: "red", width: 1.5, shape: "hv" },
            },
            4
          )
        );
      }
      const yMaxC = Math.max(
        cum_b.length ? Math.max(...cum_b) : 0,
        cum_a.length ? Math.max(...cum_a) : 0
      );
      layout.yaxis5.range = [0, yMaxC > 0 ? yMaxC * 1.1 : 1];
    } else {
      layout.yaxis5.range = [0, 1];
    }
    layout.xaxis5.range = xlim;
    layout.yaxis5.title = { text: "Cumulative volume", font: { size: 10 } };
    layout.shapes.push({
      type: "line",
      xref: "x5",
      yref: "y5",
      x0: 0,
      x1: 0,
      y0: 0,
      y1: 1,
      line: { color: "gray", width: 1 },
      layer: "below",
    });
  } else {
    layout.yaxis4.range = [0, 1];
    layout.yaxis5.range = [0, 1];
    layout.xaxis4.range = xlim;
    layout.xaxis5.range = xlim;
    layout.shapes.push(
      {
        type: "line",
        xref: "x4",
        yref: "y4",
        x0: 0,
        x1: 0,
        y0: 0,
        y1: 1,
        line: { color: "gray", width: 1 },
        layer: "below",
      },
      {
        type: "line",
        xref: "x5",
        yref: "y5",
        x0: 0,
        x1: 0,
        y0: 0,
        y1: 1,
        line: { color: "gray", width: 1 },
        layer: "below",
      }
    );
  }

  return { data, layout };
}

function buildLofPlotly(frame) {
  const lofObs = (frame && frame.lof_observed) || {};
  const lofS = (frame && frame.lof_sample) || {};
  if (!lofObs.present && !lofS.present) {
    return null;
  }

  const domains = rowDomains2();
  const layout = {
    height: 560,
    margin: { l: 56, r: 16, t: 44, b: 40 },
    paper_bgcolor: "#161b22",
    plot_bgcolor: "#0d1117",
    font: { color: "#e6edf3", size: 10 },
    showlegend: true,
    legend: { font: { size: 9 } },
    title: { text: "Limit order flow diagnostics (Plotly)", font: { size: 13 } },
    shapes: [],
  };

  layout.xaxis = {
    domain: [0.12, 0.99],
    anchor: "y",
    showgrid: true,
    gridcolor: "#30363d",
    zeroline: false,
    title: { text: "Spread from mid (ticks)", font: { size: 10 } },
  };
  layout.yaxis = {
    domain: domains[0],
    anchor: "x",
    showgrid: true,
    gridcolor: "#30363d",
    zeroline: false,
    title: { text: "Cumulative order change", font: { size: 10 } },
  };
  layout.xaxis2 = {
    domain: [0.12, 0.99],
    anchor: "y2",
    showgrid: true,
    gridcolor: "#30363d",
    zeroline: false,
    title: { text: "Spread from mid (ticks)", font: { size: 10 } },
  };
  layout.yaxis2 = {
    domain: domains[1],
    anchor: "x2",
    showgrid: true,
    gridcolor: "#30363d",
    zeroline: false,
    title: { text: "Cumulative (sample)", font: { size: 10 } },
  };

  const data = [];

  if (lofObs.present) {
    const spreadMax = num(lofObs.spread_max) || 100;
    const xCenters = Float64Array.from(lofObs.x || []);
    const cumBid = Float64Array.from(lofObs.cum_bid || []);
    const cumAsk = Float64Array.from(lofObs.cum_ask || []);
    const xBidVals = [];
    const yBidVals = [];
    for (let i = 0; i < xCenters.length; i++) {
      if (xCenters[i] <= 0) {
        xBidVals.push(xCenters[i]);
        yBidVals.push(cumBid[i]);
      }
    }
    xBidVals.reverse();
    yBidVals.reverse();
    const xAskIdx = [];
    const yAskIdx = [];
    for (let i = 0; i < xCenters.length; i++) {
      if (xCenters[i] >= 0) {
        xAskIdx.push(xCenters[i]);
        yAskIdx.push(cumAsk[i]);
      }
    }

    if (xBidVals.length) {
      data.push({
        type: "scatter",
        mode: "lines",
        name: "Bid",
        x: xBidVals,
        y: yBidVals,
        xaxis: "x",
        yaxis: "y",
        line: { color: "green", width: 1.5, shape: "hv" },
      });
    }
    if (xAskIdx.length) {
      data.push({
        type: "scatter",
        mode: "lines",
        name: "Ask",
        x: xAskIdx,
        y: yAskIdx,
        xaxis: "x",
        yaxis: "y",
        line: { color: "red", width: 1.5, shape: "hv" },
      });
    }
    layout.xaxis.range = [-spreadMax, spreadMax];
    const parts = [];
    if (xBidVals.length) parts.push(...yBidVals);
    if (xAskIdx.length) parts.push(...yAskIdx);
    if (parts.length) {
      const lo = Math.min(...parts);
      const hi = Math.max(...parts);
      const py = hi !== lo ? Math.max((hi - lo) * 0.1, 1e-9) : 1;
      layout.yaxis.range = [lo - py, hi + py];
    } else {
      layout.yaxis.range = [-1, 1];
    }
    layout.shapes.push(
      {
        type: "line",
        xref: "x",
        yref: "y",
        x0: 0,
        x1: 0,
        y0: layout.yaxis.range[0],
        y1: layout.yaxis.range[1],
        line: { color: "gray", width: 1 },
      },
      {
        type: "line",
        xref: "x",
        yref: "y",
        x0: layout.xaxis.range[0],
        x1: layout.xaxis.range[1],
        y0: 0,
        y1: 0,
        line: { color: "gray", width: 1, dash: "dash" },
      }
    );
  } else {
    layout.xaxis.range = [-100, 100];
    layout.yaxis.range = [-1, 1];
  }

  if (lofS.present) {
    const spreadMaxOb = parseInt(String(lofS.spread_max || 100), 10) || 100;
    const sampleBid = Float64Array.from(lofS.y_bid || []);
    const sampleAsk = Float64Array.from(lofS.y_ask || []);
    const xBid = Array.from({ length: sampleBid.length }, (_, i) => -i);
    const xAsk = Array.from({ length: sampleAsk.length }, (_, i) => i);

    if (xBid.length && sampleBid.length) {
      data.push({
        type: "scatter",
        mode: "lines",
        name: "Bid (sample)",
        x: xBid,
        y: Array.from(sampleBid),
        xaxis: "x2",
        yaxis: "y2",
        line: { color: "green", width: 1.5, shape: "hv" },
      });
    }
    if (xAsk.length && sampleAsk.length) {
      data.push({
        type: "scatter",
        mode: "lines",
        name: "Ask (sample)",
        x: xAsk,
        y: Array.from(sampleAsk),
        xaxis: "x2",
        yaxis: "y2",
        line: { color: "red", width: 1.5, shape: "hv" },
      });
    }
    layout.xaxis2.range = [-spreadMaxOb, spreadMaxOb];
    const concat = Array.from(sampleBid).concat(Array.from(sampleAsk));
    const cleaned = concat.map((v) => (Number.isFinite(v) ? v : 0));
    if (cleaned.length) {
      let ymin = Math.min(...cleaned);
      let ymax = Math.max(...cleaned);
      if (!Number.isFinite(ymin) || !Number.isFinite(ymax)) {
        ymin = -1;
        ymax = 1;
      }
      const pad =
        ymax !== ymin && Number.isFinite(ymax - ymin)
          ? Math.max((ymax - ymin) * 0.1, 1e-9)
          : 1;
      layout.yaxis2.range = [ymin - pad, ymax + pad];
    } else {
      layout.yaxis2.range = [-1, 1];
    }
    layout.shapes.push(
      {
        type: "line",
        xref: "x2",
        yref: "y2",
        x0: 0,
        x1: 0,
        y0: layout.yaxis2.range[0],
        y1: layout.yaxis2.range[1],
        line: { color: "gray", width: 1 },
      },
      {
        type: "line",
        xref: "x2",
        yref: "y2",
        x0: layout.xaxis2.range[0],
        x1: layout.xaxis2.range[1],
        y0: 0,
        y1: 0,
        line: { color: "gray", width: 1, dash: "dash" },
      }
    );
  } else {
    layout.xaxis2.range = [-100, 100];
    layout.yaxis2.range = [-1, 1];
  }

  return { data, layout };
}

if (typeof window !== "undefined") {
  window.buildActionDepthPlotly = buildActionDepthPlotly;
  window.buildLofPlotly = buildLofPlotly;
}
