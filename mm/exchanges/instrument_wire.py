"""Per-(exchange, domain_symbol) wire profile: API symbols, private legs, price scales, wallet folding.

Domain symbols in config (e.g. ``USDT_IRT``) are the strategy truth. Exchange-specific behavior is
declared only in ``mm/exchanges/<exchange>/instrument_wire.json`` (optional). This module applies
those rules generically; it does not branch on exchange names.

**``instrument_wire.json`` (optional keys)**

- ``api_market_symbol.compact_suffix_replace``: ``{ "from": "IRT", "to": "TMN" }`` — if the
  normalized compact symbol (no underscore) ends with ``from`` (case-insensitive), that suffix is
  replaced by ``to`` for REST/WS market endpoints.

- ``quote_wire``: map domain **quote** asset (e.g. ``"IRT"``) to overrides:
  - ``private_dst_override``: private order API dst currency (e.g. Nobitex ``rls``).
  - ``public_price_scale_from_api``: multiply raw public limit/orderbook/trade **prices** by this
    to get internal floats (e.g. ``0.1`` for Rial → Toman).

- ``wallet_merge.bucket_folds``: list of ``{ "source_tickers": [...], "target_canonical": "IRT",
  "amount_scale": 0.1 }``. Each raw wallet row whose ticker is in ``source_tickers`` contributes
  to ``target_canonical`` with ``free``/``locked``/``total`` multiplied by ``amount_scale``. Rows
  not matched use ``canonical_asset_code`` with scale ``1.0``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from mm.core.asset_aliases import canonical_asset_code
from mm.core.models.instrument import normalize_symbol, split_symbol
from mm.core.models.market import PriceLevel, RawOrderbook, RawTrade

# Back-compat for tests / ``quote_unit``; Nobitex scales are defined in ``nobitex/instrument_wire.json``.
RIAL_PER_TOMAN_NOBITEX = 10.0

_WIRE_CACHE: dict[str, tuple[int, dict[str, Any]]] = {}


def _wire_json_path(exchange: str) -> Path:
    return Path(__file__).resolve().parent / exchange.lower().strip() / "instrument_wire.json"


def _load_wire_overrides(exchange: str) -> dict[str, Any]:
    ex = exchange.lower().strip()
    path = _wire_json_path(ex)
    if not path.is_file():
        return {}
    mtime_ns = path.stat().st_mtime_ns
    hit = _WIRE_CACHE.get(ex)
    if hit is not None and hit[0] == mtime_ns:
        return hit[1]
    raw = json.loads(path.read_text(encoding="utf-8"))
    data = raw if isinstance(raw, dict) else {}
    _WIRE_CACHE[ex] = (mtime_ns, data)
    return data


@dataclass(frozen=True, slots=True)
class InstrumentWireProfile:
    """Wire-level mapping for one venue + domain trading symbol."""

    exchange: str
    domain_symbol: str
    base_asset: str
    quote_asset: str
    api_market_symbol: str
    private_src_currency: str
    private_dst_currency: str
    """Multiply raw **quote** prices from public/market APIs by this to get internal floats."""

    public_price_scale_from_api: float

    def private_order_currencies(self) -> tuple[str, str]:
        return (self.private_src_currency, self.private_dst_currency)

    def scale_public_price_from_api(self, price: float) -> float:
        return float(price) * self.public_price_scale_from_api

    def scale_limit_price_to_api(self, price: float) -> float:
        s = self.public_price_scale_from_api
        if s == 0.0:
            return float(price)
        return float(price) / s

    def scale_orderbook_from_api(self, book: RawOrderbook) -> RawOrderbook:
        if self.public_price_scale_from_api == 1.0:
            return book
        return RawOrderbook(
            exchange=book.exchange,
            symbol=book.symbol,
            timestamp_ms=book.timestamp_ms,
            bids=tuple(
                PriceLevel(self.scale_public_price_from_api(float(level.price)), float(level.qty))
                for level in book.bids
            ),
            asks=tuple(
                PriceLevel(self.scale_public_price_from_api(float(level.price)), float(level.qty))
                for level in book.asks
            ),
            is_snapshot=book.is_snapshot,
        )

    def scale_trade_from_api(self, trade: RawTrade) -> RawTrade:
        if self.public_price_scale_from_api == 1.0:
            return trade
        return RawTrade(
            trade_id=trade.trade_id,
            timestamp_ms=trade.timestamp_ms,
            price=self.scale_public_price_from_api(float(trade.price)),
            qty=float(trade.qty),
            side=trade.side,
            exchange=trade.exchange,
            symbol=trade.symbol,
        )

    def merge_wallet_balances(self, raw: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
        return merge_wallet_balances_raw(self.exchange, raw)


def build_instrument_wire(exchange: str, domain_symbol: str) -> InstrumentWireProfile:
    """Build the wire profile for ``exchange`` and config-style ``domain_symbol`` (e.g. ``USDT_IRT``)."""

    ex = exchange.lower().strip()
    base, quote = split_symbol(domain_symbol)
    compact = normalize_symbol(domain_symbol)
    ovr = _load_wire_overrides(ex)

    api_market_symbol = _apply_compact_suffix_replace(compact, ovr.get("api_market_symbol"))

    private_src = base.lower()
    private_dst = quote.lower()
    public_price_scale = 1.0

    qw = ovr.get("quote_wire")
    if isinstance(qw, dict):
        qrule = qw.get(quote)
        if isinstance(qrule, dict):
            pdo = qrule.get("private_dst_override")
            if isinstance(pdo, str) and pdo.strip():
                private_dst = pdo.strip().lower()
            pps = qrule.get("public_price_scale_from_api")
            if pps is not None:
                public_price_scale = float(pps)

    return InstrumentWireProfile(
        exchange=ex,
        domain_symbol=domain_symbol,
        base_asset=base,
        quote_asset=quote,
        api_market_symbol=api_market_symbol,
        private_src_currency=private_src,
        private_dst_currency=private_dst,
        public_price_scale_from_api=public_price_scale,
    )


def _apply_compact_suffix_replace(compact: str, api_section: Any) -> str:
    if not isinstance(api_section, dict):
        return compact
    rep = api_section.get("compact_suffix_replace")
    if not isinstance(rep, dict):
        return compact
    fr = str(rep.get("from", "")).strip()
    to = str(rep.get("to", "")).strip()
    if not fr or not to:
        return compact
    c = compact.upper()
    fu = fr.upper()
    if c.endswith(fu):
        return c[: -len(fu)] + to.upper()
    return compact


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _wallet_row_contributions(exchange: str, raw_asset: str) -> list[tuple[str, float]]:
    a = str(raw_asset or "").upper().strip()
    if not a:
        return []
    ovr = _load_wire_overrides(exchange)
    wm = ovr.get("wallet_merge")
    if isinstance(wm, dict):
        for fold in wm.get("bucket_folds") or ():
            if not isinstance(fold, dict):
                continue
            tickers = {str(t).upper() for t in fold.get("source_tickers") or ()}
            if a in tickers:
                tgt = str(fold.get("target_canonical", "IRT")).upper().strip()
                if not tgt:
                    continue
                sc = float(fold.get("amount_scale", 1.0))
                return [(tgt, sc)]
    c = canonical_asset_code(a)
    return [(c, 1.0)] if c else []


def _merge_by_contributions(
    result: dict[str, dict[str, float]],
    contributions_fn: Callable[[str], list[tuple[str, float]]],
) -> dict[str, dict[str, float]]:
    if not result:
        return {}
    merged: dict[str, dict[str, float]] = {}
    for asset, bucket in result.items():
        if not isinstance(bucket, dict):
            continue
        free = _safe_float(bucket.get("free", 0.0))
        locked = _safe_float(bucket.get("locked", 0.0))
        total = _safe_float(bucket.get("total", free + locked))
        for canon, scale in contributions_fn(str(asset).upper()):
            if not canon:
                continue
            af = free * scale
            al = locked * scale
            at = total * scale
            prev = merged.get(canon)
            if prev is None:
                merged[canon] = {"free": af, "locked": al, "total": at}
            else:
                merged[canon] = {
                    "free": prev["free"] + af,
                    "locked": prev["locked"] + al,
                    "total": prev["total"] + at,
                }
    return merged


def merge_wallet_balances_raw(exchange: str, raw: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    """Normalize raw wallet dict keys/amounts using ``<exchange>/instrument_wire.json`` ``wallet_merge``."""

    ex = exchange.lower().strip()
    return _merge_by_contributions(raw, lambda a: _wallet_row_contributions(ex, a))
