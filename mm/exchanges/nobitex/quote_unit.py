"""Nobitex IRT pairs: public API uses Rial; internal model matches Wallex-style Toman (÷10)."""

from __future__ import annotations

from mm.core.models.market import PriceLevel, RawOrderbook, RawTrade

# Nobitex documents IRT/RLS notional in Rial while strategies use Toman for *_IRT symbols.
NOBITEX_RIAL_PER_TOMAN = 10.0


def nobitex_irt_quoted_pair(symbol: str) -> bool:
    """True when ``symbol`` is a ``BASE_IRT`` pair (config style ``BTC_IRT`` or ``BTCIRT``)."""

    s = str(symbol or "").strip().upper().replace("_", "")
    return len(s) > 3 and s.endswith("IRT")


def scale_public_quote_price_from_api(symbol: str, price: float) -> float:
    """Order book / trade prices from Nobitex REST/WS → internal quote-per-base (Toman)."""

    if not nobitex_irt_quoted_pair(symbol) or not price:
        return float(price)
    return float(price) / NOBITEX_RIAL_PER_TOMAN


def scale_limit_price_to_nobitex_api(symbol: str, price: float) -> float:
    """Internal limit price (Toman) → Rial string/value expected by Nobitex ``market/orders/add``."""

    if not nobitex_irt_quoted_pair(symbol) or not price:
        return float(price)
    return float(price) * NOBITEX_RIAL_PER_TOMAN


def scale_nobitex_orderbook_quotes(symbol: str, book: RawOrderbook) -> RawOrderbook:
    if not nobitex_irt_quoted_pair(symbol):
        return book
    return RawOrderbook(
        exchange=book.exchange,
        symbol=book.symbol,
        timestamp_ms=book.timestamp_ms,
        bids=tuple(
            PriceLevel(scale_public_quote_price_from_api(symbol, float(level.price)), float(level.qty))
            for level in book.bids
        ),
        asks=tuple(
            PriceLevel(scale_public_quote_price_from_api(symbol, float(level.price)), float(level.qty))
            for level in book.asks
        ),
        is_snapshot=book.is_snapshot,
    )


def scale_nobitex_trade_quote(symbol: str, trade: RawTrade) -> RawTrade:
    if not nobitex_irt_quoted_pair(symbol):
        return trade
    return RawTrade(
        trade_id=trade.trade_id,
        timestamp_ms=trade.timestamp_ms,
        price=scale_public_quote_price_from_api(symbol, float(trade.price)),
        qty=float(trade.qty),
        side=trade.side,
        exchange=trade.exchange,
        symbol=trade.symbol,
    )
