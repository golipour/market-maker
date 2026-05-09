from __future__ import annotations

from mm_core.ports.market_data import StepperPort


def create_stepper(**kwargs) -> StepperPort:
    """
    Runtime wiring for the current market-data stepper.

    Returns ``OrderFlowStepper`` from ``mm_runtime.order_flow_stepper``. Provider selection
    (REST vs buffered polling vs websocket for main + hedge) is centralized in
    ``mm_runtime.market_data_mux.MarketDataMux`` (Phase 4).
    """
    from mm_runtime.order_flow_stepper import OrderFlowStepper

    return OrderFlowStepper(**kwargs)

