from __future__ import annotations

from mm.mpc.objective_normalize import wealth_scale


def test_wealth_scale() -> None:
    assert wealth_scale(1_000_000.0, 10.0, 50_000.0) == 1_000_000.0 + 500_000.0
