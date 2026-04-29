from __future__ import annotations

from mm_core.ports.market_data import StepperPort


def create_stepper(**kwargs) -> StepperPort:
    """
    Runtime wiring for the current market-data stepper.

    This is a compatibility shim: today it returns the legacy `OrderFlowStepper`.
    Later, it will build a capability-driven WS/REST mux and emit canonical events.
    """
    from order_flow_stepper import OrderFlowStepper

    return OrderFlowStepper(**kwargs)

