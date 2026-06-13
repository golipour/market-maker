"""Structured optimizer diagnostics for controller events and Grafana."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from mm.diagnostics.models import ActionSnapshot, ObjectiveSnapshot
from mm.mpc.mpc_state import SigmoidMixtureAction
from mm.mpc.objective_terms import OBJECTIVE_CONTRIBUTION_KEYS


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
        return x if math.isfinite(x) else default
    except Exception:
        return default


def _components_to_float(components: dict[str, Any]) -> dict[str, float]:
    return {str(k): _finite(v) for k, v in components.items()}


def compute_contribution_fractions(
    components: dict[str, float],
    *,
    eps: float = 1e-12,
) -> dict[str, float]:
    """
    Share of each weighted objective term in the scalar ``J`` optimized at closure end.

    Uses ``|component| / sum(|component|)`` over objective-level keys only, where each
    ``component`` is the fully weighted term stored in ``J`` (CE, ``w_book·B``, ``-w_inv·Z``).
    """
    abs_components = {
        str(k): abs(float(v))
        for k, v in components.items()
        if str(k) in OBJECTIVE_CONTRIBUTION_KEYS
    }
    total_abs = sum(abs_components.values())
    if total_abs <= float(eps):
        return {k: 0.0 for k in abs_components}
    return {k: v / total_abs for k, v in abs_components.items()}


@dataclass(slots=True)
class EpochObjectiveMetrics:
    """Scalar objective facts from one optimizer closure evaluation."""

    objective_value: float
    loss: float
    components: dict[str, float]
    expected_total_reward: float
    expected_total_return: float
    expected_next_reward: float
    main_reward_mean: float
    global_reward_mean: float

    @classmethod
    def from_tensors(
        cls,
        *,
        objective_value: float,
        components: dict[str, Any],
        expected_total_reward: float,
        expected_total_return: float,
        expected_next_reward: float,
        main_reward_mean: float,
        global_reward_mean: float,
    ) -> EpochObjectiveMetrics:
        comps = _components_to_float(components)
        obj = _finite(objective_value)
        return cls(
            objective_value=obj,
            loss=-obj,
            components=comps,
            expected_total_reward=_finite(expected_total_reward),
            expected_total_return=_finite(expected_total_return),
            expected_next_reward=_finite(expected_next_reward),
            main_reward_mean=_finite(main_reward_mean),
            global_reward_mean=_finite(global_reward_mean),
        )

    def to_snapshot(self) -> ObjectiveSnapshot:
        return ObjectiveSnapshot(
            value=self.objective_value,
            loss=self.loss,
            components=dict(self.components),
            contribution_fractions=compute_contribution_fractions(self.components),
            expected_total_reward=self.expected_total_reward,
            expected_total_return=self.expected_total_return,
            expected_next_reward=self.expected_next_reward,
            main_reward_mean=self.main_reward_mean,
            global_reward_mean=self.global_reward_mean,
        )


def objective_snapshot_payload(snapshot: ObjectiveSnapshot | None) -> dict[str, Any]:
    if snapshot is None:
        return {}
    return {
        "value": snapshot.value,
        "loss": snapshot.loss,
        "components": dict(snapshot.components),
        "contribution_fractions": dict(snapshot.contribution_fractions),
        "expected_total_reward": snapshot.expected_total_reward,
        "expected_total_return": snapshot.expected_total_return,
        "expected_next_reward": snapshot.expected_next_reward,
        "main_reward_mean": snapshot.main_reward_mean,
        "global_reward_mean": snapshot.global_reward_mean,
    }


def build_action_snapshot(
    action: SigmoidMixtureAction | None,
    *,
    mid: float,
    spread_levels: tuple[float, ...] | list[float],
) -> ActionSnapshot:
    if action is None:
        return ActionSnapshot()
    effective = action.as_effective()
    bid: list[dict[str, float]] = []
    ask: list[dict[str, float]] = []
    mid_f = float(mid)
    for spread in spread_levels:
        s = float(spread)
        try:
            db = float(effective.effective_depth_bid(s))
        except Exception:
            db = float("nan")
        try:
            da = float(effective.effective_depth_ask(s))
        except Exception:
            da = float("nan")
        if math.isfinite(db):
            bid.append({"spread_quote": s, "price": mid_f - s, "cumulative_qty": db})
        if math.isfinite(da):
            ask.append({"spread_quote": s, "price": mid_f + s, "cumulative_qty": da})
    return ActionSnapshot(
        effective_depth={
            "kind": "sigmoid_mixture",
            "mid": mid_f,
            "bid": bid,
            "ask": ask,
        },
        raw_params={"flat": list(effective.to_flat())},
    )


def action_snapshot_payload(snapshot: ActionSnapshot | None) -> dict[str, Any]:
    if snapshot is None:
        return {}
    return {
        "effective_depth": dict(snapshot.effective_depth),
        "raw_params": dict(snapshot.raw_params),
    }


@dataclass(slots=True)
class OptimizerRunDiagnostics:
    """Public optimizer diagnostics for one ``optimize`` call."""

    config: dict[str, Any] = field(default_factory=dict)
    timing: dict[str, Any] = field(default_factory=dict)
    initial: EpochObjectiveMetrics | None = None
    final: EpochObjectiveMetrics | None = None
    best: EpochObjectiveMetrics | None = None
    best_epoch: int | None = None
    action_initial: ActionSnapshot | None = None
    action_final: ActionSnapshot | None = None
    action_best: ActionSnapshot | None = None
    gradients: dict[str, Any] = field(default_factory=dict)
    rollout: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        initial_snap = self.initial.to_snapshot() if self.initial else None
        final_snap = self.final.to_snapshot() if self.final else None
        best_snap = self.best.to_snapshot() if self.best else None
        initial_val = initial_snap.value if initial_snap else None
        best_val = best_snap.value if best_snap else None
        final_val = final_snap.value if final_snap else None
        improvement: dict[str, float | None] = {
            "best_minus_initial": (
                (best_val - initial_val)
                if initial_val is not None and best_val is not None
                else None
            ),
            "final_minus_initial": (
                (final_val - initial_val)
                if initial_val is not None and final_val is not None
                else None
            ),
        }
        final_fractions = (
            dict(final_snap.contribution_fractions) if final_snap is not None else {}
        )
        return {
            "config": dict(self.config),
            "timing": dict(self.timing),
            "objective": {
                "initial": objective_snapshot_payload(initial_snap),
                "final": objective_snapshot_payload(final_snap),
                "best": objective_snapshot_payload(best_snap),
                "improvement": improvement,
                "contribution_fractions": final_fractions,
            },
            "action": {
                "initial": action_snapshot_payload(self.action_initial),
                "final": action_snapshot_payload(self.action_final),
                "best": action_snapshot_payload(self.action_best),
            },
            "gradients": dict(self.gradients),
            "rollout": dict(self.rollout),
            "best_epoch": self.best_epoch,
        }
