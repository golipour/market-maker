from __future__ import annotations

import json
import os
import time
from collections import deque
from typing import Any

import streamlit as st


def _redis_url() -> str:
    return str(os.environ.get("MM_DIAGNOSTICS_REDIS_URL", "redis://redis:6379/0")).strip()


def _redis_channel() -> str:
    return str(os.environ.get("MM_DIAGNOSTICS_CHANNEL", "mm.diagnostics.v1")).strip()


def _max_frames() -> int:
    try:
        return max(50, int(os.environ.get("MM_VIZ_MAX_FRAMES", "2000")))
    except Exception:
        return 2000


def _get_redis():
    import redis  # type: ignore

    return redis.Redis.from_url(_redis_url())


def _frame_title(frame: dict[str, Any]) -> str:
    ex = frame.get("exchange", "")
    sym = frame.get("symbol", "")
    sc = frame.get("point", {}).get("step_counter")
    return f"{ex} {sym} step={sc}"


st.set_page_config(page_title="mm_viz", layout="wide")
st.title("Market Maker Realtime Diagnostics")

with st.sidebar:
    st.markdown("### Stream")
    channel = st.text_input("Redis channel", _redis_channel())
    st.caption(f"Redis: `{_redis_url()}`")
    max_frames = st.number_input("Keep last N frames", min_value=50, max_value=20000, value=_max_frames(), step=50)
    auto_refresh = st.toggle("Auto refresh", value=True)
    refresh_ms = st.number_input("Refresh (ms)", min_value=200, max_value=10000, value=800, step=100)

if "frames" not in st.session_state:
    st.session_state.frames = deque(maxlen=int(max_frames))
if "last_frame" not in st.session_state:
    st.session_state.last_frame = None

frames: deque = st.session_state.frames


def poll_once() -> None:
    r = _get_redis()
    pub = r.pubsub(ignore_subscribe_messages=True)
    pub.subscribe(channel)
    # Non-blocking poll window.
    t_end = time.time() + 0.2
    while time.time() < t_end:
        msg = pub.get_message(timeout=0.05)
        if not msg:
            continue
        try:
            data = msg.get("data")
            if isinstance(data, (bytes, bytearray)):
                data = data.decode("utf-8", errors="replace")
            frame = json.loads(data)
            frames.append(frame)
            st.session_state.last_frame = frame
        except Exception:
            continue


colA, colB = st.columns([2, 1])
with colA:
    if st.button("Poll now"):
        poll_once()
with colB:
    st.write("Latest:", _frame_title(st.session_state.last_frame) if st.session_state.last_frame else "—")

if auto_refresh:
    poll_once()
    st.markdown(f"<meta http-equiv='refresh' content='{float(refresh_ms)/1000.0}'>", unsafe_allow_html=True)

frame = st.session_state.last_frame
if not frame:
    st.info("No frames yet. Start the market maker with `MM_DIAGNOSTICS=1` and Redis configured.")
    st.stop()

tabs = st.tabs(
    [
        "Overview",
        "Action & Depth",
        "LOF",
        "Price diagnostics",
        "QQ volumes",
        "Anchor",
        "Loss/NLL",
        "Raw frame",
    ]
)


def _line_chart(tab, title: str, series: dict[str, list[float]]):
    with tab:
        st.subheader(title)
        st.line_chart(series)


with tabs[0]:
    st.subheader("Latest point")
    st.json(frame.get("point", {}))

    st.subheader("Key series (tail)")
    rs = frame.get("runner_series", {}) or {}
    st.line_chart(
        {
            "wealth_quote": list(rs.get("wealth_quote", []) or [])[-500:],
            "wealth_tick_lot": list(rs.get("wealth_tick_lot", []) or [])[-500:],
        }
    )
    st.line_chart(
        {
            "inventory_lots": list(rs.get("inventory_lots", []) or [])[-500:],
            "cash_tick_lot": list(rs.get("cash_tick_lot", []) or [])[-500:],
        }
    )

with tabs[1]:
    st.subheader("Order book depth (spread space)")
    depth = frame.get("depth", {}) or {}
    st.line_chart(
        {
            "bid_cumvol": (depth.get("bid") or {}).get("y", []),
            "ask_cumvol": (depth.get("ask") or {}).get("y", []),
        }
    )
    st.caption("These are cumulative volumes; x-axis (spread) is available in the raw frame for exact replication.")

    st.subheader("Discretized action ladder")
    ad = frame.get("action_depth", {}) or {}
    if not ad.get("present"):
        st.info("No discretized action yet.")
    else:
        st.line_chart(
            {
                "bid_marginal": ad.get("q_bid", []),
                "ask_marginal": ad.get("q_ask", []),
                "bid_cumulative": ad.get("cum_bid", []),
                "ask_cumulative": ad.get("cum_ask", []),
            }
        )

with tabs[2]:
    st.subheader("Observed cumulative order flow (last transition)")
    lof_obs = frame.get("lof_observed", {}) or {}
    if not lof_obs.get("present"):
        st.info("Need ≥2 steps to compute observed LOF.")
    else:
        st.line_chart({"cum_bid": lof_obs.get("cum_bid", []), "cum_ask": lof_obs.get("cum_ask", [])})

    st.subheader("LOF model sample path")
    lof_s = frame.get("lof_sample", {}) or {}
    if not lof_s.get("present"):
        st.info("LOF sample unavailable.")
    else:
        st.line_chart({"sample_bid": lof_s.get("y_bid", []), "sample_ask": lof_s.get("y_ask", [])})

    st.subheader("LOF params (if available)")
    st.json(frame.get("lof_params", {}))

with tabs[3]:
    st.subheader("Representative prices + residual params")
    pd = frame.get("price_diag", {}) or {}
    st.json({k: v for k, v in pd.items() if k in ("present",)})
    rep = (pd.get("rep_prices") or {})
    st.line_chart(
        {
            "buy_realized": list(rep.get("buy_realized", []) or [])[-500:],
            "buy_computed": list(rep.get("buy_computed", []) or [])[-500:],
            "sell_realized": list(rep.get("sell_realized", []) or [])[-500:],
            "sell_computed": list(rep.get("sell_computed", []) or [])[-500:],
        }
    )
    sig = (pd.get("residual_sigma") or {})
    mu = (pd.get("residual_mu") or {})
    st.line_chart(
        {
            "sigma_buy": list(sig.get("buy", []) or [])[-500:],
            "sigma_sell": list(sig.get("sell", []) or [])[-500:],
            "mu_buy": list(mu.get("buy", []) or [])[-500:],
            "mu_sell": list(mu.get("sell", []) or [])[-500:],
        }
    )
    pe = (pd.get("pred_rep_errors") or {})
    st.line_chart(
        {
            "pred_buy_err": list(pe.get("buy", []) or [])[-500:],
            "pred_sell_err": list(pe.get("sell", []) or [])[-500:],
            "pred_buy_prev": list(pe.get("buy_prev", []) or [])[-500:],
            "pred_sell_prev": list(pe.get("sell_prev", []) or [])[-500:],
        }
    )

with tabs[4]:
    st.subheader("QQ volumes (observed vs model)")
    qq = frame.get("volume_qq", {}) or {}
    st.json({k: v for k, v in qq.items() if k in ("present", "reason", "n_obs", "n_sim_buy", "n_sim_sell")})
    if qq.get("present"):
        st.line_chart({"buy_x_model": (qq.get("buy") or {}).get("x", []), "buy_y_obs": (qq.get("buy") or {}).get("y", [])})
        st.line_chart({"sell_x_model": (qq.get("sell") or {}).get("x", []), "sell_y_obs": (qq.get("sell") or {}).get("y", [])})

with tabs[5]:
    st.subheader("Main vs Anchor representative prices")
    ad = frame.get("anchor_diag", {}) or {}
    st.line_chart(
        {
            "main_buy": (ad.get("main_ref") or {}).get("buy", []),
            "main_sell": (ad.get("main_ref") or {}).get("sell", []),
            "anchor_buy": (ad.get("anchor_ref") or {}).get("buy", []),
            "anchor_sell": (ad.get("anchor_ref") or {}).get("sell", []),
        }
    )
    st.subheader("Anchor learned delta params")
    ap = (ad.get("anchor_params") or {})
    st.line_chart(
        {
            "mu_buy": ap.get("mu_buy", []),
            "mu_sell": ap.get("mu_sell", []),
            "sigma_buy": ap.get("sigma_buy", []),
            "sigma_sell": ap.get("sigma_sell", []),
        }
    )
    st.subheader("Anchor portfolio")
    port = (ad.get("anchor_portfolio") or {})
    st.line_chart({"anchor_inventory": port.get("inventory", []), "anchor_cash": port.get("cash", [])})

with tabs[6]:
    st.subheader("Loss/NLL subplots (raw spec)")
    st.json(frame.get("loss_subplots", []))

with tabs[7]:
    st.subheader("Raw DiagnosticsFrameV1")
    st.json(frame)

