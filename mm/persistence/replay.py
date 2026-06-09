from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mm.core.events import EventEnvelope, decode_envelope
from mm.core.payload_schema import (
    EXECUTION_LIFECYCLE_SCHEMA,
    EXECUTION_LIFECYCLE_VERSION,
    EXECUTION_SNAPSHOT_SCHEMA,
    EXECUTION_SNAPSHOT_VERSION,
    JOINT_TARGET_BOOK_SCHEMA,
    JOINT_TARGET_BOOK_VERSION,
    assert_payload_schema,
)


def read_event_rows(path: str | Path) -> Iterator[dict[str, Any]]:
    """Read persisted event rows from Parquet or JSONL fallback files."""

    p = Path(path).expanduser()
    if p.is_dir():
        for child in sorted(p.rglob("*")):
            if child.suffix in (".parquet", ".jsonl"):
                yield from read_event_rows(child)
        return
    if p.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("pyarrow is required to read parquet replay files") from exc
        # Avoid Hive partition-column inference from event_date=/exchange=/symbol=/
        # paths colliding with the real columns stored in each event file.
        for row in pq.ParquetFile(p).read().to_pylist():
            yield row
        return
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_envelopes(path: str | Path) -> Iterator[Any]:
    """Yield decoded `mm.event` envelopes from persisted event rows."""

    for row in read_event_rows(path):
        raw = row.get("envelope_json") or row.get("envelope") or row
        yield decode_envelope(raw)


def read_payloads(path: str | Path, *, topic: str | None = None) -> Iterator[Any]:
    """Yield payloads in timestamp order for backtests/replay smoke tests."""

    envelopes = sorted(read_envelopes(path), key=lambda envelope: envelope.timestamp_ms)
    for envelope in envelopes:
        if topic is None or envelope.topic == topic:
            yield envelope.payload


@dataclass(frozen=True, slots=True)
class ReplayLineageChain:
    """Events connected to one strategy frame by explicit lineage ids."""

    frame_id: str
    decisions: tuple[EventEnvelope, ...] = ()
    diagnostics: tuple[EventEnvelope, ...] = ()
    target_books: tuple[EventEnvelope, ...] = ()
    execution_events: tuple[EventEnvelope, ...] = ()
    other_events: tuple[EventEnvelope, ...] = ()

    @property
    def decision_ids(self) -> tuple[str, ...]:
        return _unique_payload_values(self.decisions, "decision_id")

    @property
    def target_book_ids(self) -> tuple[str, ...]:
        return _unique_payload_values(
            (*self.decisions, *self.diagnostics, *self.target_books, *self.execution_events),
            "target_book_id",
        )

    @property
    def execution_plan_ids(self) -> tuple[str, ...]:
        return _unique_payload_values(self.execution_events, "execution_plan_id")

    def execution_events_for_plan(self, execution_plan_id: str) -> tuple[EventEnvelope, ...]:
        wanted = execution_plan_id.strip()
        if not wanted:
            return ()
        return tuple(
            envelope
            for envelope in self.execution_events
            if _payload_value(envelope, "execution_plan_id") == wanted
        )

    def reconstruct_state(self) -> ReplayFrameState:
        """Fold this lineage chain into a replay-friendly frame state summary."""

        return reconstruct_lineage_state(self)


@dataclass(frozen=True, slots=True)
class ReplayExecutionPlanState:
    """Replay summary for one venue execution plan."""

    execution_plan_id: str
    exchange: str = ""
    symbol: str = ""
    target_orders: tuple[dict[str, Any], ...] = ()
    place_requested_orders: tuple[dict[str, Any], ...] = ()
    place_accepted_order_ids: tuple[str, ...] = ()
    place_accepted_orders: tuple[dict[str, Any], ...] = ()
    place_failed_errors: tuple[str, ...] = ()
    cancel_requested_order_ids: tuple[str, ...] = ()
    cancel_accepted_order_ids: tuple[str, ...] = ()
    cancel_failed_errors: tuple[str, ...] = ()
    validation_rejections: tuple[dict[str, Any], ...] = ()
    latest_snapshot: dict[str, Any] | None = None

    @property
    def pending_places(self) -> dict[str, Any]:
        snapshot = self.latest_snapshot or {}
        pending = snapshot.get("pending_places")
        return dict(pending) if isinstance(pending, dict) else {}

    @property
    def pending_cancels(self) -> dict[str, Any]:
        snapshot = self.latest_snapshot or {}
        pending = snapshot.get("pending_cancels")
        return dict(pending) if isinstance(pending, dict) else {}

    @property
    def confirmed_live_orders(self) -> dict[str, Any]:
        snapshot = self.latest_snapshot or {}
        live = snapshot.get("live_orders")
        return dict(live) if isinstance(live, dict) else {}

    @property
    def has_failures(self) -> bool:
        return bool(self.place_failed_errors or self.cancel_failed_errors or self.validation_rejections)


@dataclass(frozen=True, slots=True)
class ReplayFrameState:
    """Replay summary for one strategy frame and its downstream execution."""

    frame_id: str
    latest_decision: dict[str, Any] | None = None
    latest_diagnostics: dict[str, Any] | None = None
    latest_target_book: dict[str, Any] | None = None
    execution_plans: dict[str, ReplayExecutionPlanState] = field(default_factory=dict)

    @property
    def decision_id(self) -> str:
        if not isinstance(self.latest_decision, dict):
            return ""
        value = self.latest_decision.get("decision_id")
        return str(value).strip() if value is not None else ""

    @property
    def target_book_id(self) -> str:
        for payload in (self.latest_target_book, self.latest_decision, self.latest_diagnostics):
            if isinstance(payload, dict) and payload.get("target_book_id") is not None:
                return str(payload["target_book_id"]).strip()
        return ""

    @property
    def confirmed_live_orders(self) -> dict[str, Any]:
        live: dict[str, Any] = {}
        for plan in self.execution_plans.values():
            live.update(plan.confirmed_live_orders)
        return live

    @property
    def pending_places(self) -> dict[str, Any]:
        pending: dict[str, Any] = {}
        for plan in self.execution_plans.values():
            pending.update(plan.pending_places)
        return pending

    @property
    def pending_cancels(self) -> dict[str, Any]:
        pending: dict[str, Any] = {}
        for plan in self.execution_plans.values():
            pending.update(plan.pending_cancels)
        return pending

    @property
    def has_execution_failures(self) -> bool:
        return any(plan.has_failures for plan in self.execution_plans.values())


def reconstruct_lineage_state(chain: ReplayLineageChain) -> ReplayFrameState:
    """Fold a lineage chain into state needed by replay/simulation checks."""

    mutable_plans: dict[str, _MutableExecutionPlanState] = {}
    latest_decision = _latest_payload(chain.decisions)
    latest_diagnostics = _latest_payload(chain.diagnostics)
    latest_target_book = _latest_payload(chain.target_books)
    if latest_target_book is not None:
        assert_payload_schema(latest_target_book, JOINT_TARGET_BOOK_SCHEMA, JOINT_TARGET_BOOK_VERSION)

    for envelope in chain.target_books:
        payload = _dict_payload(envelope)
        assert_payload_schema(payload, JOINT_TARGET_BOOK_SCHEMA, JOINT_TARGET_BOOK_VERSION)
        for plan_id, venue_payload in _venue_target_payloads(payload):
            plan = mutable_plans.setdefault(plan_id, _MutableExecutionPlanState(plan_id))
            plan.exchange = str(venue_payload.get("exchange") or plan.exchange or "")
            plan.symbol = str(venue_payload.get("symbol") or plan.symbol or "")
            plan.target_orders = _tuple_of_dicts(venue_payload.get("orders"))

    for envelope in sorted(chain.execution_events, key=lambda item: item.timestamp_ms):
        payload = _dict_payload(envelope)
        plan_id = _payload_value(envelope, "execution_plan_id")
        if not plan_id:
            plan_id = _fallback_execution_plan_id(payload, envelope)
        plan = mutable_plans.setdefault(plan_id, _MutableExecutionPlanState(plan_id))
        plan.exchange = str(payload.get("exchange") or plan.exchange or "")
        plan.symbol = str(payload.get("symbol") or plan.symbol or "")
        plan.apply(envelope.event_type, payload)

    return ReplayFrameState(
        frame_id=chain.frame_id,
        latest_decision=latest_decision,
        latest_diagnostics=latest_diagnostics,
        latest_target_book=latest_target_book,
        execution_plans={plan_id: plan.freeze() for plan_id, plan in mutable_plans.items()},
    )


def read_lineage_chains(path: str | Path) -> dict[str, ReplayLineageChain]:
    """Read persisted rows and group replay events by ``frame_id``.

    Events without a ``frame_id`` are ignored because they cannot be joined to a
    strategy frame without timestamp/topic inference.
    """

    return lineage_chains_from_envelopes(read_envelopes(path))


def lineage_chains_from_envelopes(envelopes: Iterable[EventEnvelope]) -> dict[str, ReplayLineageChain]:
    groups: dict[str, _MutableLineageChain] = {}
    for envelope in sorted(envelopes, key=lambda item: item.timestamp_ms):
        frame_id = _payload_value(envelope, "frame_id")
        if not frame_id:
            continue
        groups.setdefault(frame_id, _MutableLineageChain(frame_id)).add(envelope)
    return {frame_id: group.freeze() for frame_id, group in groups.items()}


@dataclass(slots=True)
class _MutableLineageChain:
    frame_id: str
    decisions: list[EventEnvelope] = field(default_factory=list)
    diagnostics: list[EventEnvelope] = field(default_factory=list)
    target_books: list[EventEnvelope] = field(default_factory=list)
    execution_events: list[EventEnvelope] = field(default_factory=list)
    other_events: list[EventEnvelope] = field(default_factory=list)

    def add(self, envelope: EventEnvelope) -> None:
        if envelope.event_type in {"StrategyDecision", "ControllerFallback", "ControllerError"}:
            self.decisions.append(envelope)
        elif envelope.event_type == "ControllerDiagnostics":
            self.diagnostics.append(envelope)
        elif envelope.event_type == "JointTargetBook":
            self.target_books.append(envelope)
        elif _is_execution_event(envelope):
            self.execution_events.append(envelope)
        else:
            self.other_events.append(envelope)

    def freeze(self) -> ReplayLineageChain:
        return ReplayLineageChain(
            frame_id=self.frame_id,
            decisions=tuple(self.decisions),
            diagnostics=tuple(self.diagnostics),
            target_books=tuple(self.target_books),
            execution_events=tuple(self.execution_events),
            other_events=tuple(self.other_events),
        )


@dataclass(slots=True)
class _MutableExecutionPlanState:
    execution_plan_id: str
    exchange: str = ""
    symbol: str = ""
    target_orders: tuple[dict[str, Any], ...] = ()
    place_requested_orders: list[dict[str, Any]] = field(default_factory=list)
    place_accepted_order_ids: list[str] = field(default_factory=list)
    place_accepted_orders: list[dict[str, Any]] = field(default_factory=list)
    place_failed_errors: list[str] = field(default_factory=list)
    cancel_requested_order_ids: list[str] = field(default_factory=list)
    cancel_accepted_order_ids: list[str] = field(default_factory=list)
    cancel_failed_errors: list[str] = field(default_factory=list)
    validation_rejections: list[dict[str, Any]] = field(default_factory=list)
    latest_snapshot: dict[str, Any] | None = None

    def apply(self, event_type: str, payload: dict[str, Any]) -> None:
        if event_type == "ExecutionSnapshot":
            assert_payload_schema(payload, EXECUTION_SNAPSHOT_SCHEMA, EXECUTION_SNAPSHOT_VERSION)
        else:
            assert_payload_schema(payload, EXECUTION_LIFECYCLE_SCHEMA, EXECUTION_LIFECYCLE_VERSION)
        if event_type == "PlaceOrdersRequested":
            self.place_requested_orders.extend(_tuple_of_dicts(payload.get("orders")))
        elif event_type == "PlaceOrdersAccepted":
            self.place_accepted_order_ids.extend(_string_list(payload.get("order_ids")))
            self.place_accepted_orders.extend(_tuple_of_dicts(payload.get("orders")))
        elif event_type == "PlaceOrdersFailed":
            self.place_failed_errors.append(_event_error(payload))
        elif event_type in {"CancelOrdersRequested", "CancelActorOrdersRequested"}:
            self.cancel_requested_order_ids.extend(_string_list(payload.get("order_ids")))
        elif event_type == "CancelOrdersAccepted":
            self.cancel_accepted_order_ids.extend(_string_list(payload.get("order_ids")))
        elif event_type == "CancelOrdersFailed":
            self.cancel_failed_errors.append(_event_error(payload))
        elif event_type == "OrderValidationRejected":
            self.validation_rejections.append(dict(payload))
        elif event_type == "ExecutionSnapshot":
            self.latest_snapshot = dict(payload)

    def freeze(self) -> ReplayExecutionPlanState:
        return ReplayExecutionPlanState(
            execution_plan_id=self.execution_plan_id,
            exchange=self.exchange,
            symbol=self.symbol,
            target_orders=self.target_orders,
            place_requested_orders=tuple(self.place_requested_orders),
            place_accepted_order_ids=tuple(_dedupe_strings(self.place_accepted_order_ids)),
            place_accepted_orders=tuple(self.place_accepted_orders),
            place_failed_errors=tuple(error for error in self.place_failed_errors if error),
            cancel_requested_order_ids=tuple(_dedupe_strings(self.cancel_requested_order_ids)),
            cancel_accepted_order_ids=tuple(_dedupe_strings(self.cancel_accepted_order_ids)),
            cancel_failed_errors=tuple(error for error in self.cancel_failed_errors if error),
            validation_rejections=tuple(self.validation_rejections),
            latest_snapshot=self.latest_snapshot,
        )


def _is_execution_event(envelope: EventEnvelope) -> bool:
    return envelope.topic.startswith(("execution:orders:", "execution:state:")) or envelope.event_type in {
        "ExecutionSyncRequested",
        "ExecutionSyncCompleted",
        "ExecutionSyncFailed",
        "PlaceOrdersRequested",
        "PlaceOrdersAccepted",
        "PlaceOrdersFailed",
        "CancelOrdersRequested",
        "CancelOrdersAccepted",
        "CancelOrdersFailed",
        "OrderValidationRejected",
        "ExecutionSnapshot",
    }


def _payload_value(envelope: EventEnvelope, key: str) -> str:
    payload = envelope.payload
    if not isinstance(payload, dict):
        return ""
    value = payload.get(key)
    if value is None:
        return ""
    return str(value).strip()


def _unique_payload_values(envelopes: tuple[EventEnvelope, ...], key: str) -> tuple[str, ...]:
    values = []
    seen = set()
    for envelope in envelopes:
        value = _payload_value(envelope, key)
        if value and value not in seen:
            values.append(value)
            seen.add(value)
    return tuple(values)


def _latest_payload(envelopes: tuple[EventEnvelope, ...]) -> dict[str, Any] | None:
    if not envelopes:
        return None
    return _dict_payload(max(envelopes, key=lambda item: item.timestamp_ms))


def _dict_payload(envelope: EventEnvelope) -> dict[str, Any]:
    return dict(envelope.payload) if isinstance(envelope.payload, dict) else {}


def _venue_target_payloads(payload: dict[str, Any]) -> Iterator[tuple[str, dict[str, Any]]]:
    for key in ("local", "global"):
        value = payload.get(key)
        if isinstance(value, dict):
            plan_id = str(value.get("execution_plan_id") or "").strip()
            if plan_id:
                yield plan_id, value
    venues = payload.get("venues")
    if isinstance(venues, dict):
        for value in venues.values():
            if isinstance(value, dict):
                plan_id = str(value.get("execution_plan_id") or "").strip()
                if plan_id:
                    yield plan_id, value
    plan_id = str(payload.get("execution_plan_id") or "").strip()
    if plan_id:
        yield plan_id, payload


def _tuple_of_dicts(value: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(dict(item) for item in value if isinstance(item, dict))


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _dedupe_strings(values: Iterable[str]) -> list[str]:
    deduped: list[str] = []
    seen = set()
    for value in values:
        clean = str(value).strip()
        if clean and clean not in seen:
            deduped.append(clean)
            seen.add(clean)
    return deduped


def _event_error(payload: dict[str, Any]) -> str:
    for key in ("error", "reason", "message"):
        value = payload.get(key)
        if value:
            return str(value)
    return ""


def _fallback_execution_plan_id(payload: dict[str, Any], envelope: EventEnvelope) -> str:
    exchange = str(payload.get("exchange") or envelope.exchange or "").strip()
    symbol = str(payload.get("symbol") or envelope.symbol or "").strip()
    if exchange or symbol:
        return f"unlinked:{exchange}:{symbol}"
    return "unlinked"


@dataclass(frozen=True, slots=True)
class ReplayFidelityViolation:
    """One replay/backtest invariant failure for a lineage chain."""

    code: str
    message: str
    execution_plan_id: str = ""


@dataclass(frozen=True, slots=True)
class ReplayFidelityReport:
    """Result of checking one frame lineage chain for lifecycle coherence."""

    frame_id: str
    violations: tuple[ReplayFidelityViolation, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.violations


def verify_persisted_lineage_fidelity(path: str | Path) -> dict[str, ReplayFidelityReport]:
    """Load persisted rows and verify each ``frame_id`` lineage chain."""

    return {
        frame_id: verify_lineage_fidelity(chain)
        for frame_id, chain in read_lineage_chains(path).items()
    }


def verify_lineage_fidelity(chain: ReplayLineageChain) -> ReplayFidelityReport:
    """Verify reconstructed target/execution lifecycle invariants for one frame."""

    state = chain.reconstruct_state()
    violations: list[ReplayFidelityViolation] = []
    violations.extend(_lineage_id_violations(chain, state))
    violations.extend(_target_execution_coverage_violations(chain, state))
    violations.extend(_execution_intent_violations(state))
    violations.extend(_execution_lifecycle_order_violations(chain))
    violations.extend(_execution_snapshot_violations(chain))
    return ReplayFidelityReport(frame_id=chain.frame_id, violations=tuple(violations))


def _lineage_id_violations(chain: ReplayLineageChain, state: ReplayFrameState) -> list[ReplayFidelityViolation]:
    violations: list[ReplayFidelityViolation] = []
    expected_decision_id = state.decision_id
    expected_target_book_id = state.target_book_id
    related = (
        *chain.decisions,
        *chain.diagnostics,
        *chain.target_books,
        *chain.execution_events,
    )
    for envelope in related:
        decision_id = _payload_value(envelope, "decision_id")
        if decision_id and expected_decision_id and decision_id != expected_decision_id:
            violations.append(
                ReplayFidelityViolation(
                    code="decision_id_mismatch",
                    message=(
                        f"{envelope.event_type} decision_id={decision_id!r} "
                        f"does not match frame decision_id={expected_decision_id!r}"
                    ),
                )
            )
        target_book_id = _payload_value(envelope, "target_book_id")
        if target_book_id and expected_target_book_id and target_book_id != expected_target_book_id:
            violations.append(
                ReplayFidelityViolation(
                    code="target_book_id_mismatch",
                    message=(
                        f"{envelope.event_type} target_book_id={target_book_id!r} "
                        f"does not match frame target_book_id={expected_target_book_id!r}"
                    ),
                )
            )
    return violations


def _target_execution_coverage_violations(
    chain: ReplayLineageChain,
    state: ReplayFrameState,
) -> list[ReplayFidelityViolation]:
    violations: list[ReplayFidelityViolation] = []
    if chain.execution_events and not chain.target_books:
        violations.append(
            ReplayFidelityViolation(
                code="execution_without_target_book",
                message="execution events exist but no JointTargetBook is linked to this frame",
            )
        )
    target_plan_ids = set(state.execution_plans.keys()) & {
        plan_id
        for plan_id, plan in state.execution_plans.items()
        if plan.target_orders
    }
    for plan_id in sorted(target_plan_ids):
        if not chain.execution_events_for_plan(plan_id):
            violations.append(
                ReplayFidelityViolation(
                    code="target_plan_without_execution",
                    message="target book contains orders but no execution events were recorded",
                    execution_plan_id=plan_id,
                )
            )
    for plan_id in chain.execution_plan_ids:
        if plan_id.startswith("unlinked:"):
            violations.append(
                ReplayFidelityViolation(
                    code="orphan_execution_plan",
                    message=f"execution events use unlinked plan id {plan_id!r}",
                    execution_plan_id=plan_id,
                )
            )
        elif plan_id not in state.execution_plans:
            violations.append(
                ReplayFidelityViolation(
                    code="orphan_execution_plan",
                    message=f"execution events reference unknown plan id {plan_id!r}",
                    execution_plan_id=plan_id,
                )
            )
    return violations


def _execution_intent_violations(state: ReplayFrameState) -> list[ReplayFidelityViolation]:
    violations: list[ReplayFidelityViolation] = []
    for plan_id, plan in state.execution_plans.items():
        target_keys = {_order_key(order) for order in plan.target_orders}
        rejected_keys = {_order_key(order) for order in plan.validation_rejections}
        expected_place_keys = target_keys - rejected_keys
        requested_keys = {_order_key(order) for order in plan.place_requested_orders}
        if requested_keys - expected_place_keys:
            extra = sorted(requested_keys - expected_place_keys)
            violations.append(
                ReplayFidelityViolation(
                    code="place_request_not_from_target",
                    message=f"place requests contain orders not in target minus validation rejections: {extra}",
                    execution_plan_id=plan_id,
                )
            )
        if rejected_keys - target_keys:
            extra = sorted(rejected_keys - target_keys)
            violations.append(
                ReplayFidelityViolation(
                    code="validation_rejection_not_from_target",
                    message=f"validation rejections reference orders not present in target book: {extra}",
                    execution_plan_id=plan_id,
                )
            )
    return violations


def _execution_lifecycle_order_violations(chain: ReplayLineageChain) -> list[ReplayFidelityViolation]:
    violations: list[ReplayFidelityViolation] = []
    for plan_id in chain.execution_plan_ids:
        events = chain.execution_events_for_plan(plan_id)
        place_requested_ms = _first_event_timestamp(events, "PlaceOrdersRequested")
        place_accepted_ms = _first_event_timestamp(events, "PlaceOrdersAccepted")
        place_failed_ms = _first_event_timestamp(events, "PlaceOrdersFailed")
        cancel_requested_ms = _first_event_timestamp(events, "CancelOrdersRequested")
        cancel_accepted_ms = _first_event_timestamp(events, "CancelOrdersAccepted")
        cancel_failed_ms = _first_event_timestamp(events, "CancelOrdersFailed")
        if place_requested_ms is not None:
            for label, later_ms in (
                ("PlaceOrdersAccepted", place_accepted_ms),
                ("PlaceOrdersFailed", place_failed_ms),
            ):
                if later_ms is not None and later_ms < place_requested_ms:
                    violations.append(
                        ReplayFidelityViolation(
                            code="execution_lifecycle_order",
                            message=f"{label} occurred before PlaceOrdersRequested",
                            execution_plan_id=plan_id,
                        )
                    )
        if cancel_requested_ms is not None:
            for label, later_ms in (
                ("CancelOrdersAccepted", cancel_accepted_ms),
                ("CancelOrdersFailed", cancel_failed_ms),
            ):
                if later_ms is not None and later_ms < cancel_requested_ms:
                    violations.append(
                        ReplayFidelityViolation(
                            code="execution_lifecycle_order",
                            message=f"{label} occurred before CancelOrdersRequested",
                            execution_plan_id=plan_id,
                        )
                    )
    return violations


def _execution_snapshot_violations(chain: ReplayLineageChain) -> list[ReplayFidelityViolation]:
    violations: list[ReplayFidelityViolation] = []
    for envelope in chain.execution_events:
        if envelope.event_type != "ExecutionSnapshot":
            continue
        payload = _dict_payload(envelope)
        plan_id = _payload_value(envelope, "execution_plan_id")
        for count_key, dict_key in (
            ("live_order_count", "live_orders"),
            ("pending_place_count", "pending_places"),
            ("pending_cancel_count", "pending_cancels"),
        ):
            if count_key not in payload:
                continue
            collection = payload.get(dict_key)
            if not isinstance(collection, dict):
                continue
            expected = int(payload[count_key])
            actual = len(collection)
            if expected != actual:
                violations.append(
                    ReplayFidelityViolation(
                        code="execution_snapshot_count_mismatch",
                        message=(
                            f"ExecutionSnapshot {count_key}={expected} "
                            f"but len({dict_key})={actual}"
                        ),
                        execution_plan_id=plan_id,
                    )
                )
    return violations


def _first_event_timestamp(events: tuple[EventEnvelope, ...], event_type: str) -> int | None:
    for envelope in events:
        if envelope.event_type == event_type:
            return envelope.timestamp_ms
    return None


def _order_key(order: dict[str, Any]) -> tuple[str, float, float]:
    side = str(order.get("side") or "").strip().lower()
    price = round(float(order.get("price") or 0.0), 8)
    qty = round(float(order.get("qty") or 0.0), 8)
    return side, price, qty
