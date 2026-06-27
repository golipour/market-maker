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
