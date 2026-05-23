from __future__ import annotations

from mm.mpc.objective_normalize import inventory_scale, wealth_scale


def test_wealth_and_inventory_scales() -> None:
    assert wealth_scale(1_000_000.0, 10.0, 50_000.0) == 1_000_000.0 + 500_000.0
    assert inventory_scale(10.0, 3.0) == 10.0
    assert inventory_scale(0.0, None) == 1.0
