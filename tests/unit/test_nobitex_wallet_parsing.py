from __future__ import annotations

from mm.exchanges.nobitex import private_rest


def test_nobitex_v1_wallet_list_maps_active_balance_to_free(monkeypatch) -> None:
    def fake_get_auth(path, token):
        if path != "/users/wallets/list":
            raise AssertionError(path)
        return {
            "status": "ok",
            "wallets": [
                {
                    "currency": "usdt",
                    "balance": "23",
                    "blockedBalance": "3",
                    "activeBalance": "20",
                }
            ],
        }

    monkeypatch.setattr(private_rest, "_get_auth", fake_get_auth)

    balances = private_rest.get_balances("token")

    assert balances["USDT"] == {"free": 20.0, "locked": 3.0, "total": 23.0}


def test_nobitex_v2_wallets_maps_balance_to_wallet_total(monkeypatch) -> None:
    calls: list[str] = []

    def fake_get_auth(path, token):
        calls.append(path)
        if path == "/users/wallets/list":
            return {"status": "error"}
        return {
            "status": "ok",
            "wallets": {
                "USDT": {"balance": "23", "blocked": "3"},
            },
        }

    monkeypatch.setattr(private_rest, "_get_auth", fake_get_auth)

    balances = private_rest.get_balances("token")

    assert "/v2/wallets" in calls
    assert balances["USDT"] == {"free": 20.0, "locked": 3.0, "total": 23.0}


def test_nobitex_v2_wallets_does_not_double_count_order_collateral(monkeypatch) -> None:
    def fake_get_auth(path, token):
        if path == "/users/wallets/list":
            return {"status": "error"}
        return {
            "status": "ok",
            "wallets": {
                "IRT": {"balance": "17939224.39", "blocked": "7160381.44"},
            },
        }

    monkeypatch.setattr(private_rest, "_get_auth", fake_get_auth)

    balances = private_rest.get_balances("token")

    assert balances["IRT"] == {
        "free": 10_778_842.95,
        "locked": 7_160_381.44,
        "total": 17_939_224.39,
    }
