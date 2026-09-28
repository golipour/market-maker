"""Import the REAL production price model without dragging in HTTP deps.

`mm.integrations.trade_stats` imports `requests` at module scope, but only uses
it inside live exchange adapters -- the `Orderbook`/`Level` dataclasses we need
touch nothing network-related.  The analysis venv has no `requests` and the
sandbox forbids installing into it, so we inject a minimal inert stub before
importing.  Nothing in the code path we exercise can reach it.
"""
import sys
import types


def install_requests_stub() -> None:
    if "requests" in sys.modules:
        return
    try:
        import requests  # noqa: F401
        return
    except ImportError:
        pass

    mod = types.ModuleType("requests")

    def _blocked(*args, **kwargs):
        raise RuntimeError(
            "network access is stubbed out in the model_lab analysis venv; "
            "this code path must not need HTTP"
        )

    class Response:
        status_code = 0

        def raise_for_status(self):
            _blocked()

        def json(self):
            _blocked()

    mod.get = _blocked
    mod.post = _blocked
    mod.Session = object
    mod.Response = Response
    mod.exceptions = types.SimpleNamespace(RequestException=RuntimeError)
    sys.modules["requests"] = mod

    urllib_mod = types.ModuleType("urllib3")
    sys.modules.setdefault("urllib3", urllib_mod)


def load_production_price_model():
    """Return the real TorchConditionalMechanicalMixturePriceModel class."""
    install_requests_stub()
    sys.path.insert(0, "/Users/teanab/Documents/projects/market-maker")
    from mm.mpc.price_torch import TorchConditionalMechanicalMixturePriceModel

    return TorchConditionalMechanicalMixturePriceModel


def load_orderbook_types():
    install_requests_stub()
    sys.path.insert(0, "/Users/teanab/Documents/projects/market-maker")
    from mm.integrations.trade_stats import Level, Orderbook

    return Orderbook, Level
