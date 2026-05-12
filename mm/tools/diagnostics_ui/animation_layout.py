"""
Dynamic animation layout: each pipeline component declares its subplots.
The figure is built from these specs so changing estimator/tester does not require animation code changes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mm.layers.runtime.order_flow_runner import OrderFlowRunner


def get_estimator_plot_specs(estimator) -> list[dict]:
    """Number and spec of subplots the estimator needs. Fallback: 1 if no get_plot_specs."""
    if hasattr(estimator, "get_plot_specs"):
        return list(estimator.get_plot_specs())
    return [{"title": "Estimator", "ylabel": ""}]


def get_tester_plot_specs(tester) -> list[dict]:
    """Number and spec of subplots the tester needs. Fallback: 1."""
    if hasattr(tester, "get_plot_specs"):
        return list(tester.get_plot_specs())
    return [{"title": "Model score", "ylabel": ""}]


def get_price_model_plot_specs(runner: "OrderFlowRunner") -> list[dict]:
    """Number and spec of subplots the price model needs (μ_mid, σ_mid). 0 if no price_model."""
    pe = getattr(runner, "price_model", None)
    if pe is not None and hasattr(pe, "get_plot_specs"):
        return list(pe.get_plot_specs())
    return []


def get_limit_order_flow_model_plot_specs(runner: "OrderFlowRunner") -> list[dict]:
    """Number and spec of subplots the limit order flow model needs (e.g. NLL). 0 if absent."""
    lof = getattr(runner, "limit_order_flow_model", None)
    if lof is not None and hasattr(lof, "get_plot_specs"):
        return list(lof.get_plot_specs())
    return []


def build_layout(runner: "OrderFlowRunner") -> tuple[list[tuple[str, int]], list[float]]:
    """
    Build (section_specs, height_ratios) from the runner.
    section_specs: [(section_id, n_axes), ...] in display order.
    height_ratios: one per total axis, for plt.subplots(..., height_ratios=...).
    """
    model = getattr(runner, "mof_model", None)
    tester = runner.tester
    n_model = len(get_estimator_plot_specs(model))
    n_price_model = len(get_price_model_plot_specs(runner))
    n_limit_order_flow = 0
    n_tester = len(get_tester_plot_specs(tester))

    sections = [
        ("market_order_flow", 1),
        ("mid", 1),
        # ("estimator", n_model),  # Moved to second Predictive metrics window
        ("price_model", n_price_model),
        # ("score", n_tester),  # Moved to separate Predictive log loss window
        ("inv", 1),
        ("cash", 1),
        ("wealth", 1),
        ("delta_spread_pnl", 1),
        ("delta_inventory_pnl", 1),
        ("delta_adverse_pnl", 1),
        ("delta_fee_pnl", 1),
        ("realized_dW", 1),
        # ("toxicity", 1),  # Commented out
    ]
    n_total = sum(n for _, n in sections)
    # height_ratios: market_order_flow larger, rest similar
    hr_vol = 1.0
    hr_est = 0.5
    hr_score = 0.5
    hr_state = 0.4
    hr_mid = 0.6
    hr_depth = 0.7
    hr_action_depth = 0.6
    hr_tox = 0.5
    height_ratios = (
        [hr_vol]
        + [hr_mid]
        # + [hr_est] * n_model  # Moved to second Predictive metrics window
        + [hr_est] * n_price_model
        # + [hr_score] * n_tester  # Moved to separate Predictive log loss window
        + [hr_state] * 3
        + [hr_state, hr_state, hr_state, hr_state]  # delta_spread_pnl, delta_inventory_pnl, delta_adverse_pnl, delta_fee_pnl
        + [hr_state]  # realized_dW
        # + [hr_tox]  # Commented out
    )
    return sections, height_ratios


def section_axis_range(sections: list[tuple[str, int]], section_id: str) -> tuple[int, int]:
    """Return (start_idx, end_idx) for section_id (end exclusive)."""
    i = 0
    for sid, n in sections:
        if sid == section_id:
            return (i, i + n)
        i += n
    return (0, 0)


def section_ranges(sections: list[tuple[str, int]]) -> dict[str, tuple[int, int]]:
    """Return dict section_id -> (start_idx, end_idx) for all sections."""
    out = {}
    i = 0
    for sid, n in sections:
        out[sid] = (i, i + n)
        i += n
    return out
