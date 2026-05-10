"""Load local secrets from ``mm/.env`` (optional).

Copy ``mm/.env.example`` to ``mm/.env`` and set values. Variables are not
overwritten if already set in the process environment (``override=False``).
"""

from __future__ import annotations

from pathlib import Path


def mm_package_root() -> Path:
    return Path(__file__).resolve().parent


def load_mm_env(*, override: bool = False) -> bool:
    """Load ``mm/.env`` if present. Returns True if python-dotenv ran, else False."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return False
    env_path = mm_package_root() / ".env"
    if not env_path.is_file():
        return False
    load_dotenv(env_path, override=override)
    return True
