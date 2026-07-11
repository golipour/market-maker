"""Analysis helpers for model-lab walk-forward experiments."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from statistics import mean
from typing import Any

from research.model_lab.core.types import IntervalStep, StepScore
from research.model_lab.core.walk_forward import WalkForwardResult


def score_rows(
    result: WalkForwardResult,
    steps: Sequence[IntervalStep],
    *,
    model_label: str | None = None,
) -> list[dict[str, Any]]:
    """Flatten one walk-forward result and attach same-index market context."""

    rows: list[dict[str, Any]] = []
    for score in result.scores:
        step = steps[score.step_index] if 0 <= score.step_index < len(steps) else None
        rows.append(
            {
                "model": model_label or score.model_name,
                "model_name": score.model_name,
                "model_kind": score.model_kind,
                "family": score.family,
                "step_index": score.step_index,
                "ts": score.ts,
                "nll": score.nll,
                "log_likelihood": score.log_likelihood,
                "skip_reason": score.skip_reason,
                "standardized_residual": score.standardized_residual,
                "total_volume": step.total_volume if step is not None else None,
                "buy_volume": step.buy_volume if step is not None else None,
                "sell_volume": step.sell_volume if step is not None else None,
                "mid": step.mid if step is not None else None,
                "spread": step.spread if step is not None else None,
                "volume_regime": volume_regime(step.total_volume if step is not None else None),
            }
        )
    return rows


def leaderboard(results: Mapping[str, WalkForwardResult]) -> list[dict[str, Any]]:
    """Return summary rows sorted by family then mean NLL."""

    rows: list[dict[str, Any]] = []
    for label, result in results.items():
        summary = result.summary
        if summary is None:
            continue
        rows.append(
            {
                "label": label,
                "model": summary.model_name,
                "kind": summary.model_kind,
                "family": summary.family,
                "n_steps": summary.n_steps,
                "n_scored": summary.n_scored,
                "n_skipped": summary.n_skipped,
                "mean_nll": summary.mean_nll,
                "median_nll": summary.median_nll,
                "total_nll": summary.total_nll,
                "mean_log_likelihood": summary.mean_log_likelihood,
                "skip_reasons": summary.skip_reasons,
            }
        )
    return sorted(rows, key=lambda row: (str(row["family"]), _sort_value(row["mean_nll"])))


def stratified_nll(
    results: Mapping[str, WalkForwardResult],
    steps: Sequence[IntervalStep],
    *,
    field: str = "volume_regime",
) -> list[dict[str, Any]]:
    """Aggregate NLL by model and a simple market regime field."""

    groups: dict[tuple[str, str], list[float]] = {}
    for label, result in results.items():
        for row in score_rows(result, steps, model_label=label):
            nll = row.get("nll")
            if nll is None or not math.isfinite(float(nll)):
                continue
            regime = str(row.get(field) or "unknown")
            groups.setdefault((label, regime), []).append(float(nll))
    out: list[dict[str, Any]] = []
    for (label, regime), values in groups.items():
        ordered = sorted(values)
        out.append(
            {
                "label": label,
                field: regime,
                "n": len(values),
                "mean_nll": mean(values),
                "median_nll": ordered[len(ordered) // 2],
                "p90_nll": percentile(ordered, 0.90),
            }
        )
    return sorted(out, key=lambda row: (row["label"], row[field]))


def paired_nll_deltas(
    results: Mapping[str, WalkForwardResult],
    *,
    baseline_label: str,
) -> list[dict[str, Any]]:
    """Paired per-step NLL deltas versus a baseline, lower means candidate wins."""

    baseline = results[baseline_label]
    baseline_by_step = {
        score.step_index: float(score.nll)
        for score in baseline.scores
        if score.nll is not None and math.isfinite(float(score.nll))
    }
    rows: list[dict[str, Any]] = []
    for label, result in results.items():
        if label == baseline_label:
            continue
        deltas: list[float] = []
        for score in result.scores:
            if score.nll is None or score.step_index not in baseline_by_step:
                continue
            value = float(score.nll)
            if math.isfinite(value):
                deltas.append(value - baseline_by_step[score.step_index])
        if not deltas:
            continue
        rows.append(
            {
                "label": label,
                "baseline": baseline_label,
                "n": len(deltas),
                "mean_delta_nll": mean(deltas),
                "median_delta_nll": sorted(deltas)[len(deltas) // 2],
                "win_rate": sum(1 for delta in deltas if delta < 0.0) / len(deltas),
                "stderr_mean_delta": sample_stderr(deltas),
            }
        )
    return sorted(rows, key=lambda row: row["mean_delta_nll"])


def rolling_mean_nll(result: WalkForwardResult, *, window: int = 100) -> list[tuple[int, float]]:
    """Simple rolling mean NLL series for cold-start and regime-shift plots."""

    values: list[tuple[int, float]] = []
    buf: list[float] = []
    for score in result.scores:
        if score.nll is None or not math.isfinite(float(score.nll)):
            continue
        buf.append(float(score.nll))
        if len(buf) > window:
            buf.pop(0)
        values.append((score.step_index, mean(buf)))
    return values


def stationarity_report(steps: Sequence[IntervalStep], *, blocks: int = 6) -> list[dict[str, Any]]:
    """Block summary for stationary candidate representations."""

    if len(steps) < 2:
        return []
    transitions: list[dict[str, float]] = []
    activity_ewma = 1.0
    gamma = 0.95
    for idx in range(1, len(steps)):
        prev = steps[idx - 1]
        cur = steps[idx]
        total = max(0.0, float(cur.total_volume))
        activity_ewma = gamma * activity_ewma + (1.0 - gamma) * total
        delta_mid = float(cur.mid - prev.mid)
        log_return = math.log(cur.mid / prev.mid) if cur.mid > 0.0 and prev.mid > 0.0 else 0.0
        spread_norm_delta = delta_mid / max(1e-12, float(prev.spread))
        normalized_volume = total / max(1e-12, activity_ewma)
        transitions.append(
            {
                "idx": float(idx),
                "delta_mid": delta_mid,
                "log_return": log_return,
                "spread_norm_delta": spread_norm_delta,
                "normalized_volume": normalized_volume,
            }
        )
    block_count = max(1, int(blocks))
    block_size = max(1, math.ceil(len(transitions) / block_count))
    rows: list[dict[str, Any]] = []
    for block_index in range(block_count):
        chunk = transitions[block_index * block_size : (block_index + 1) * block_size]
        if not chunk:
            continue
        row: dict[str, Any] = {"block": block_index, "n": len(chunk)}
        for key in ("delta_mid", "log_return", "spread_norm_delta", "normalized_volume"):
            values = [item[key] for item in chunk if math.isfinite(item[key])]
            row[f"{key}_mean"] = mean(values) if values else None
            row[f"{key}_std"] = sample_std(values) if len(values) > 1 else None
        rows.append(row)
    return rows


def volume_regime(total_volume: float | None) -> str:
    if total_volume is None:
        return "unknown"
    volume = max(0.0, float(total_volume))
    if volume <= 0.0:
        return "zero"
    if volume < 100.0:
        return "low"
    if volume < 500.0:
        return "medium"
    return "high"


def volume_quintile(total_volume: float | None, *, edges: Sequence[float]) -> str:
    if total_volume is None:
        return "unknown"
    volume = max(0.0, float(total_volume))
    for idx in range(len(edges) - 1):
        if volume <= edges[idx + 1]:
            return f"Q{idx + 1}"
    return f"Q{len(edges) - 1}"


def volume_quintile_edges(steps: Sequence[IntervalStep], *, positive_only: bool = True) -> list[float]:
    """Compute quintile breakpoints from observed total volumes."""

    volumes = [max(0.0, float(step.total_volume)) for step in steps]
    if positive_only:
        volumes = [v for v in volumes if v > 0.0]
    if not volumes:
        return [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    ordered = sorted(volumes)
    edges = [0.0]
    for q in (0.2, 0.4, 0.6, 0.8, 1.0):
        edges.append(percentile(ordered, q) or 0.0)
    return edges


def dataset_profile(steps: Sequence[IntervalStep]) -> dict[str, Any]:
    """Human-readable summary stats for experiment notebooks."""

    if not steps:
        return {"n_steps": 0}
    volumes = [max(0.0, float(step.total_volume)) for step in steps]
    mids = [float(step.mid) for step in steps]
    spreads = [float(step.spread) for step in steps]
    zero_frac = sum(1 for v in volumes if v <= 0.0) / len(volumes)
    positive = [v for v in volumes if v > 0.0]
    deltas = [float(steps[i].mid - steps[i - 1].mid) for i in range(1, len(steps))]
    return {
        "n_steps": len(steps),
        "since": steps[0].ts.isoformat(),
        "until": steps[-1].ts.isoformat(),
        "zero_volume_fraction": zero_frac,
        "volume_p50": percentile(sorted(positive), 0.5) if positive else 0.0,
        "volume_p90": percentile(sorted(positive), 0.9) if positive else 0.0,
        "volume_p99": percentile(sorted(positive), 0.99) if positive else 0.0,
        "mid_min": min(mids),
        "mid_max": max(mids),
        "mid_std": sample_std(mids) if len(mids) > 1 else 0.0,
        "spread_median": sorted(spreads)[len(spreads) // 2],
        "delta_mid_std": sample_std(deltas) if len(deltas) > 1 else 0.0,
        "avg_total_volume": mean(volumes),
    }


def period_split_nll(result: WalkForwardResult, *, fraction: float = 0.5) -> dict[str, Any]:
    """Compare mean NLL in the first vs second half of scored steps."""

    scored = [float(score.nll) for score in result.scores if score.nll is not None and math.isfinite(float(score.nll))]
    if not scored:
        return {"n": 0}
    cut = max(1, int(len(scored) * fraction))
    first, second = scored[:cut], scored[cut:]
    return {
        "n": len(scored),
        "first_half_mean_nll": mean(first) if first else None,
        "second_half_mean_nll": mean(second) if second else None,
        "warmup_improvement": (mean(first) - mean(second)) if first and second else None,
    }


def paired_significance(
    results: Mapping[str, WalkForwardResult],
    *,
    baseline_label: str,
    candidate_label: str,
    n_bootstrap: int = 500,
    seed: int = 0,
) -> dict[str, Any]:
    """Bootstrap CI for paired mean NLL delta (candidate - baseline)."""

    import random

    baseline = results[baseline_label]
    candidate = results[candidate_label]
    baseline_by_step = {
        score.step_index: float(score.nll)
        for score in baseline.scores
        if score.nll is not None and math.isfinite(float(score.nll))
    }
    deltas: list[float] = []
    for score in candidate.scores:
        if score.nll is None or score.step_index not in baseline_by_step:
            continue
        value = float(score.nll)
        if math.isfinite(value):
            deltas.append(value - baseline_by_step[score.step_index])
    if len(deltas) < 2:
        return {"n": len(deltas), "mean_delta_nll": None}
    rng = random.Random(seed)
    boot: list[float] = []
    for _ in range(max(1, int(n_bootstrap))):
        sample = [deltas[rng.randrange(len(deltas))] for _ in range(len(deltas))]
        boot.append(mean(sample))
    boot.sort()
    lo = boot[int(0.025 * (len(boot) - 1))]
    hi = boot[int(0.975 * (len(boot) - 1))]
    observed = mean(deltas)
    return {
        "baseline": baseline_label,
        "candidate": candidate_label,
        "n": len(deltas),
        "mean_delta_nll": observed,
        "ci95_low": lo,
        "ci95_high": hi,
        "significant_at_5pct": (hi < 0.0) or (lo > 0.0),
        "candidate_wins": observed < 0.0,
    }


def stratified_nll_quintiles(
    results: Mapping[str, WalkForwardResult],
    steps: Sequence[IntervalStep],
) -> list[dict[str, Any]]:
    """NLL by data-driven volume quintile."""

    edges = volume_quintile_edges(steps)
    groups: dict[tuple[str, str], list[float]] = {}
    for label, result in results.items():
        for row in score_rows(result, steps, model_label=label):
            nll = row.get("nll")
            if nll is None or not math.isfinite(float(nll)):
                continue
            regime = volume_quintile(row.get("total_volume"), edges=edges)
            groups.setdefault((label, regime), []).append(float(nll))
    out: list[dict[str, Any]] = []
    for (label, regime), values in groups.items():
        ordered = sorted(values)
        out.append(
            {
                "label": label,
                "volume_quintile": regime,
                "n": len(values),
                "mean_nll": mean(values),
                "median_nll": ordered[len(ordered) // 2],
            }
        )
    return sorted(out, key=lambda row: (row["label"], row["volume_quintile"]))


def percentile(sorted_values: Sequence[float], q: float) -> float | None:
    if not sorted_values:
        return None
    q = max(0.0, min(1.0, float(q)))
    pos = q * (len(sorted_values) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(sorted_values[lo])
    frac = pos - lo
    return float(sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac)


def sample_std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((value - m) ** 2 for value in values) / (len(values) - 1))


def sample_stderr(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    return sample_std(values) / math.sqrt(len(values))


def _sort_value(value: object) -> float:
    if value is None:
        return float("inf")
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("inf")
    return out if math.isfinite(out) else float("inf")
