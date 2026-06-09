"""Replay private event sequences into a venue ledger state machine."""

from __future__ import annotations

from collections.abc import Iterable

from mm.core.models.execution import Fill
from mm.core.models.instrument import InstrumentSpec
from mm.core.models.ledger import VenueLedger
from mm.core.payload_schema import private_state_snapshot_from_payload
from mm.core.models.private import Balance, OpenOrder, OrderUpdate, PrivateStateSnapshot
from mm.ledger.state_machine import VenueLedgerStateMachine

LedgerReplayEvent = PrivateStateSnapshot | Balance | OpenOrder | OrderUpdate | Fill


def apply_ledger_event(
    machine: VenueLedgerStateMachine,
    event: LedgerReplayEvent,
) -> VenueLedger:
    """Apply one private event and return the resulting venue ledger."""

    if isinstance(event, PrivateStateSnapshot):
        return machine.apply_snapshot(event)
    if isinstance(event, dict):
        return machine.apply_snapshot(private_state_snapshot_from_payload(event))
    if isinstance(event, Balance):
        return machine.apply_balance(event)
    if isinstance(event, OpenOrder):
        return machine.apply_open_order(event)
    if isinstance(event, OrderUpdate):
        return machine.apply_order_update(event)
    if isinstance(event, Fill):
        return machine.apply_fill(event)
    raise TypeError(f"unsupported ledger replay event: {type(event)!r}")


def replay_venue_ledger(
    events: Iterable[LedgerReplayEvent],
    *,
    exchange: str,
    spec: InstrumentSpec,
) -> VenueLedger:
    """Rebuild terminal venue ledger state from an ordered event sequence."""

    machine = VenueLedgerStateMachine(exchange, spec)
    ledger = machine.snapshot()
    for event in events:
        ledger = apply_ledger_event(machine, event)
    return ledger
