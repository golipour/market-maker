from __future__ import annotations

"""
Normalize venue-native order books (tick/lot) to canonical float DTOs at the adapter edge.
"""

from mm.integrations.trade_stats import Orderbook

from mm_core.domain.orderbook import CanonicalLevel, CanonicalOrderbook
from mm_core.units.converter import UnitConverter
from mm_core.units.spec import InstrumentSpec


def to_canonical_orderbook(ob: Orderbook, spec: InstrumentSpec) -> CanonicalOrderbook:
    """
    Map ``trade_stats.Orderbook`` (``Level`` in ticks/lots) to ``CanonicalOrderbook``.

    Call this when ingesting a snapshot before passing data to models or ``mm_core`` pricing.
    """
    uc = UnitConverter(spec)
    mid_quote = float(uc.price_from_ticks(float(ob.mid)))
    bids = tuple(
        CanonicalLevel(
            price_quote=float(uc.price_from_ticks(lev.price)),
            qty_base=float(uc.qty_from_lots(lev.qty)),
        )
        for lev in ob.bids
    )
    asks = tuple(
        CanonicalLevel(
            price_quote=float(uc.price_from_ticks(lev.price)),
            qty_base=float(uc.qty_from_lots(lev.qty)),
        )
        for lev in ob.asks
    )
    return CanonicalOrderbook(
        t_ms=int(ob.t_ms),
        mid_quote=mid_quote,
        bids=bids,
        asks=asks,
    )
