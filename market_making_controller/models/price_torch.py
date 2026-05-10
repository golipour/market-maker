"""Compatibility re-export; implementation lives in ``mm.mpc.price_torch``."""

from mm.mpc.price_torch import GaussianPriceModel, PriceModel, TorchResidualPriceModel

__all__ = ["PriceModel", "GaussianPriceModel", "TorchResidualPriceModel"]
