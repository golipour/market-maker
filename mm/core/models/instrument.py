from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


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
    secondary: InstrumentSpec | None = None

    @property
    def comparable(self) -> bool:
        return self.secondary is None or self.primary.is_comparable_to(self.secondary)

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
        if self.secondary is not None:
            out["secondary"] = instrument_contract(self.secondary)
            out["comparison"] = self.primary.comparison_policy(self.secondary)
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
