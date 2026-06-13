from __future__ import annotations

import pytest
import torch

from mm.mpc.joint_mpc import joint_terminal_objective
from mm.mpc.reward_tensor import objective_entropic_return


def test_joint_terminal_objective_uses_consolidated_return_only() -> None:
    returns = torch.tensor([0.0, 0.0], dtype=torch.float64)
    local_returns = torch.tensor([0.1, -0.1], dtype=torch.float64)
    global_returns = torch.tensor([-0.1, 0.1], dtype=torch.float64)
    beta = 1.0

    objective, components = joint_terminal_objective(
        returns,
        local_returns=local_returns,
        global_returns=global_returns,
        return_risk_aversion_beta=beta,
        global_active=1.0,
    )

    expected_total = objective_entropic_return(returns, beta=beta)
    expected_local = objective_entropic_return(local_returns, beta=beta)
    expected_global = objective_entropic_return(global_returns, beta=beta)

    assert float(objective) == pytest.approx(float(expected_total))
    assert float(components["risk_adjusted_return"]) == pytest.approx(float(expected_total))
    assert float(components["local_risk_adjusted_return"]) == pytest.approx(float(expected_local))
    assert float(components["global_risk_adjusted_return"]) == pytest.approx(float(expected_global))
    assert float(expected_local) + float(expected_global) != pytest.approx(float(expected_total))


def test_joint_terminal_objective_omits_global_diagnostic_when_inactive() -> None:
    returns = torch.tensor([0.01], dtype=torch.float64)
    local_returns = torch.tensor([0.01], dtype=torch.float64)
    global_returns = torch.tensor([0.0], dtype=torch.float64)

    _, components = joint_terminal_objective(
        returns,
        local_returns=local_returns,
        global_returns=global_returns,
        return_risk_aversion_beta=1.0,
        global_active=0.0,
    )

    assert "local_risk_adjusted_return" in components
    assert "global_risk_adjusted_return" not in components
