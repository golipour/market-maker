"""Compatibility shim: canonical ``MarketEvent`` is ``mm.core.models.market.MarketEvent``."""

from mm.core.models.market import MarketEvent

__all__ = ["MarketEvent"]
