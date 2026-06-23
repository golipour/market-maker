def parse_model_spec(spec: str) -> tuple[str, str, dict[str, Any]]:
    """Parse ``family:kind`` or ``family:kind{json}``."""

    raw = spec.strip()
    if ":" not in raw:
        raise ValueError(f"invalid model spec {spec!r}; expected family:kind")
    family, rest = raw.split(":", 1)
    family = family.strip().lower()
    if family not in {"flow", "price", "baseline"}:
        raise ValueError(f"unknown family {family!r} in {spec!r}")

    config: dict[str, Any] = {}
    if "{" in rest:
        kind, _, tail = rest.partition("{")
        kind = kind.strip()
        import json

        config = json.loads("{" + tail)
    else:
        kind = rest.strip()

    return family, kind, config