from __future__ import annotations

import pytest

from mm.exchanges.wallex import private_rest


def test_wallex_get_balances_value_is_total_not_free(monkeypatch) -> None:
    class FakeResponse:
        def json(self):
            return {
                "success": True,
                "result": {
                    "balances": {
                        "USDT": {
                            "asset": "USDT",
                            "value": "626.51336125",
                            "locked": "313.8200000000001",
                        }
                    }
                },
            }

    monkeypatch.setattr(private_rest, "_get_auth", lambda path, token: FakeResponse().json())

    balances = private_rest.get_balances("token")

    assert balances["USDT"]["total"] == 626.51336125
    assert balances["USDT"]["locked"] == 313.8200000000001
    assert balances["USDT"]["free"] == pytest.approx(312.69336125)
    assert balances["USDT"]["free"] + balances["USDT"]["locked"] == pytest.approx(balances["USDT"]["total"])


def test_wallex_get_balances_explicit_free_available_path(monkeypatch) -> None:
    class FakeResponse:
        def json(self):
            return {
                "success": True,
                "result": {
                    "balances": {
                        "USDT": {
                            "asset": "USDT",
                            "free": "10",
                            "locked": "2",
                            "total": "12",
                        }
                    }
                },
            }

    monkeypatch.setattr(private_rest, "_get_auth", lambda path, token: FakeResponse().json())

    balances = private_rest.get_balances("token")

    assert balances["USDT"] == {"free": 10.0, "locked": 2.0, "total": 12.0}
