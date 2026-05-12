"""Load local secrets from the repository root ``.env`` (optional, via python-dotenv)."""

from __future__ import annotations

from pathlib import Path


def mm_package_root() -> Path:
    return Path(__file__).resolve().parent


def repo_root() -> Path:
    return mm_package_root().parent


def load_mm_env(*, override: bool = False) -> bool:
    """Load ``<repo>/.env`` if present. Returns True if a file was loaded."""

    try:
        from dotenv import load_dotenv
    except ImportError:
        return False
    env_path = repo_root() / ".env"
    if not env_path.is_file():
        return False
    load_dotenv(env_path, override=override)
    return True
