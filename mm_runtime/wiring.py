from __future__ import annotations

from mm_core.ports.market_data import StepperPort


def create_stepper(**kwargs) -> StepperPort:
    """
    Runtime wiring for the current market-data stepper.

    Today returns ``OrderFlowStepper`` from ``mm_runtime.order_flow_stepper``.
    Later: capability-driven WS/REST mux and canonical events (see docs/ARCHITECTURE_RFC.md).
    """
    from mm_runtime.order_flow_stepper import OrderFlowStepper

    return OrderFlowStepper(**kwargs)

