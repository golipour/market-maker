"""Exchange auth profiles and credential resolution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

AuthMode = Literal["api_key", "signed_key", "signed_key_passphrase"]

AUTH_MODE_LABELS: dict[AuthMode, str] = {
    "api_key": "API key",
    "signed_key": "API key + secret",
    "signed_key_passphrase": "API key + secret + passphrase",
}


@dataclass(frozen=True, slots=True)
class ExchangeAuthSpec:
    exchange: str
    default_mode: AuthMode
    supported_modes: tuple[AuthMode, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "exchange": self.exchange,
            "default_mode": self.default_mode,
            "supported_modes": list(self.supported_modes),
            "mode_labels": {mode: AUTH_MODE_LABELS[mode] for mode in self.supported_modes},
        }


EXCHANGE_AUTH_SPECS: dict[str, ExchangeAuthSpec] = {
    "wallex": ExchangeAuthSpec("wallex", "api_key", ("api_key",)),
    "nobitex": ExchangeAuthSpec("nobitex", "signed_key", ("signed_key", "api_key")),
    "tabdeal": ExchangeAuthSpec("tabdeal", "signed_key", ("signed_key",)),
    "bitpin": ExchangeAuthSpec("bitpin", "signed_key", ("signed_key",)),
    "kucoin": ExchangeAuthSpec("kucoin", "signed_key_passphrase", ("signed_key_passphrase",)),
}


def exchange_auth_spec(exchange: str) -> ExchangeAuthSpec:
    key = exchange.lower().strip()
    try:
        return EXCHANGE_AUTH_SPECS[key]
    except KeyError as exc:
        raise ValueError(f"unsupported exchange for auth: {exchange!r}") from exc


def validate_credentials_for_exchange(
    exchange: str,
    *,
    auth_mode: str,
    api_key: str | None,
    api_secret: str | None,
    api_passphrase: str | None,
) -> None:
    spec = exchange_auth_spec(exchange)
    mode = str(auth_mode or spec.default_mode).strip()
    if mode not in spec.supported_modes:
        supported = ", ".join(spec.supported_modes)
        raise ValueError(f"{exchange} auth_mode must be one of: {supported}")
    key = str(api_key or "").strip()
    secret = str(api_secret or "").strip()
    passphrase = str(api_passphrase or "").strip()
    if not key:
        raise ValueError("api_key is required")
    if mode in ("signed_key", "signed_key_passphrase") and not secret:
        raise ValueError("api_secret is required for signed auth")
    if mode == "signed_key_passphrase" and not passphrase:
        raise ValueError("api_passphrase is required for this exchange")


def resolve_exchange_auth(
    exchange: str,
    *,
    auth_mode: str | None = None,
    api_key: str | None = None,
    api_secret: str | None = None,
    api_passphrase: str | None = None,
) -> Any:
    """Build the runtime auth object (or plain token string) for one exchange."""

    exchange_key = exchange.lower().strip()
    spec = exchange_auth_spec(exchange_key)
    mode = str(auth_mode or spec.default_mode).strip()
    key = str(api_key or "").strip()
    secret = str(api_secret or "").strip()
    passphrase = str(api_passphrase or "").strip()

    if mode == "api_key":
        return key

    if exchange_key == "nobitex":
        from mm.exchanges.nobitex.private_rest import NobitexApiKeyAuth

        return NobitexApiKeyAuth(key, secret)

    if exchange_key == "tabdeal":
        from mm.exchanges.tabdeal.private_rest import TabdealApiKeyAuth

        return TabdealApiKeyAuth(key, secret)

    if exchange_key == "bitpin":
        from mm.exchanges.bitpin.private_rest import BitpinApiKeyAuth

        return BitpinApiKeyAuth(key, secret)

    if exchange_key == "kucoin":
        from mm.exchanges.kucoin.private_rest import KucoinApiKeyAuth

        return KucoinApiKeyAuth(key, secret, passphrase)

    raise ValueError(f"unsupported signed auth for exchange: {exchange_key}")
