from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


KNOWN_QUOTES = (
    "USDT",
    "USDC",
    "DAI",
    "BTC",
    "ETH",
    "IRT",
    "IRR",
    "TMN",
    "USD",
    "EUR",
)


@dataclass(frozen=True, slots=True)
class InstrumentSpec:
    """Domain symbol units as seen by strategies and engine components.

    Prices are quote-per-base floats and quantities are base-asset floats. Any
    venue tick/lot precision belongs at the order-command adapter boundary.
    """

    exchange: str
    symbol: str
    base_asset: str
    quote_asset: str
    price_unit: str = "quote_per_base"
    qty_unit: str = "base"
    notional_unit: str = "quote"
    price_step: Optional[float] = None
    qty_step: Optional[float] = None
    min_qty: Optional[float] = None
    min_notional: Optional[float] = None

    @property
    def normalized_symbol(self) -> str:
        return normalize_symbol(self.symbol)

    def is_comparable_to(self, other: "InstrumentSpec") -> bool:
        return self.base_asset == other.base_asset and self.quote_asset == other.quote_asset

    def comparison_policy(self, other: "InstrumentSpec") -> dict[str, object]:
        return {
            "same_base_quote": self.is_comparable_to(other),
            "base_asset": self.base_asset,
            "quote_asset": self.quote_asset,
            "price_scale": "1:1" if self.is_comparable_to(other) else "requires_quote_conversion",
            "qty_scale": "1:1" if self.base_asset == other.base_asset else "requires_base_conversion",
            "tick_size_effect": (
                "ignored_for_market_comparison; tick/lot steps only affect order-boundary rounding"
            ),
        }


@dataclass(frozen=True, slots=True)
class SymbolBinding:
    primary: InstrumentSpec
    hedge: InstrumentSpec | None = None

    @property
    def comparable(self) -> bool:
        return self.hedge is None or self.primary.is_comparable_to(self.hedge)

    @property
    def local(self) -> InstrumentSpec:
        """Neutral alias for ``primary``."""

        return self.primary

    @property
    def global_spec(self) -> InstrumentSpec | None:
        """Neutral alias for ``hedge``."""

        return self.hedge

    def as_contract(self) -> dict[str, object]:
        out: dict[str, object] = {
            "primary": instrument_contract(self.primary),
            "float_domain": {
                "prices": "quote-per-base floats",
                "quantities": "base-asset floats",
                "notional": "quote-asset floats",
                "ticks_lots": "not used in strategy, market events, private state, or actor state",
                "rounding_boundary": "order command adapter only",
            },
        }
        if self.hedge is not None:
            out["hedge"] = instrument_contract(self.hedge)
            out["comparison"] = self.primary.comparison_policy(self.hedge)
        return out


def instrument_contract(spec: InstrumentSpec) -> dict[str, object]:
    return {
        "exchange": spec.exchange,
        "symbol": spec.symbol,
        "base_asset": spec.base_asset,
        "quote_asset": spec.quote_asset,
        "price_unit": spec.price_unit,
        "qty_unit": spec.qty_unit,
        "notional_unit": spec.notional_unit,
        "price_step": spec.price_step,
        "qty_step": spec.qty_step,
        "min_qty": spec.min_qty,
        "min_notional": spec.min_notional,
    }


@dataclass(frozen=True, slots=True)
class InstrumentPrecisionConfig:
    """Optional JSON fallback on ``primary`` / ``hedge`` (same field names as ``InstrumentSpec``).

    At runtime, ``binding`` loads tick/lot/min from the exchange ``exchange_info`` API
    first (see ``instrument_spec_from_trade_stats``); this block fills only fields the
    API left unset.
    """

    price_step: float | None = None
    qty_step: float | None = None
    min_qty: float | None = None
    min_notional: float | None = None


def build_instrument_spec(
    exchange: str,
    symbol: str,
    *,
    price_step: float | None = None,
    qty_step: float | None = None,
    min_qty: float | None = None,
    min_notional: float | None = None,
) -> InstrumentSpec:
    base, quote = split_symbol(symbol)
    return InstrumentSpec(
        exchange=exchange.lower().strip(),
        symbol=symbol,
        base_asset=base,
        quote_asset=quote,
        price_step=price_step,
        qty_step=qty_step,
        min_qty=min_qty,
        min_notional=min_notional,
    )


def merge_instrument_spec(
    exchange: str,
    symbol: str,
    *,
    venue: InstrumentSpec | None = None,
    overrides: InstrumentPrecisionConfig | None = None,
) -> InstrumentSpec:
    """Build domain spec: ``venue`` (e.g. from API) wins; ``overrides`` fill unset fields."""
    base = venue or build_instrument_spec(exchange, symbol)
    if overrides is None:
        return base

    def _pick(venue_val: float | None, override_val: float | None) -> float | None:
        if venue_val is not None:
            return venue_val
        return override_val

    return InstrumentSpec(
        exchange=base.exchange,
        symbol=base.symbol,
        base_asset=base.base_asset,
        quote_asset=base.quote_asset,
        price_unit=base.price_unit,
        qty_unit=base.qty_unit,
        notional_unit=base.notional_unit,
        price_step=_pick(base.price_step, overrides.price_step),
        qty_step=_pick(base.qty_step, overrides.qty_step),
        min_qty=_pick(base.min_qty, overrides.min_qty),
        min_notional=_pick(base.min_notional, overrides.min_notional),
    )


def normalize_symbol(symbol: str) -> str:
    return "".join(ch for ch in symbol.upper().strip() if ch.isalnum())


def split_symbol(symbol: str) -> tuple[str, str]:
    text = symbol.upper().strip()
    for sep in ("_", "-", "/"):
        if sep in text:
            base, quote = text.split(sep, 1)
            return base.strip(), quote.strip()

    normalized = normalize_symbol(text)
    for quote in sorted(KNOWN_QUOTES, key=len, reverse=True):
        if normalized.endswith(quote) and len(normalized) > len(quote):
            return normalized[: -len(quote)], quote
    raise ValueError(f"cannot infer base/quote assets from symbol: {symbol}")


def instrument_precision_from_mapping(payload: Any) -> InstrumentPrecisionConfig | None:
    """Parse optional ``instrument`` block on a market source (InstrumentSpec field names)."""
    if not isinstance(payload, dict):
        return None
    keys = ("price_step", "qty_step", "min_qty", "min_notional")
    if not any(k in payload for k in keys):
        return None
    return InstrumentPrecisionConfig(
        price_step=_optional_float(payload.get("price_step")),
        qty_step=_optional_float(payload.get("qty_step")),
        min_qty=_optional_float(payload.get("min_qty")),
        min_notional=_optional_float(payload.get("min_notional")),
    )


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)
