from __future__ import annotations

import pytest
import torch

from mm.mpc.objective_terms import (
    InventoryZoneParams,
    combine_soft_objective_terms,
    inventory_zone_penalty,
)


def test_inventory_zone_penalty_is_zero_inside_comfort_band() -> None:
    zone = InventoryZoneParams(q_lo=-0.1, q_hi=0.1, q_min=-0.5, q_max=0.5, p=2.0)
    q = torch.tensor([-0.05, 0.0, 0.08], dtype=torch.float64)
    penalty = inventory_zone_penalty(q, zone)
    assert penalty.tolist() == pytest.approx([0.0, 0.0, 0.0])


def test_inventory_zone_penalty_ramps_outside_band() -> None:
    zone = InventoryZoneParams(q_lo=-0.1, q_hi=0.1, q_min=-0.5, q_max=0.5, p=2.0)
    q = torch.tensor([-0.3, 0.25], dtype=torch.float64)
    penalty = inventory_zone_penalty(q, zone)
    assert float(penalty[0]) > 0.0
    assert float(penalty[1]) > 0.0


def test_combine_soft_objective_terms_keeps_ce_component_pure() -> None:
    base = torch.tensor(0.5, dtype=torch.float64)
    book = torch.tensor(0.8, dtype=torch.float64)
    zone = torch.tensor(0.2, dtype=torch.float64)
    objective, components = combine_soft_objective_terms(
        base,
        {"risk_adjusted_return": base},
        w_book=0.1,
        w_inv=0.05,
        book_score=book,
        inventory_zone_mean=zone,
        book_components={"book_quality": book},
        zone_components={"inventory_zone_penalty": zone},
    )
    assert float(objective) == pytest.approx(0.5 + 0.1 * 0.8 - 0.05 * 0.2)
    assert float(components["risk_adjusted_return"]) == pytest.approx(0.5)
