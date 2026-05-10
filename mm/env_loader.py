"""Load local secrets from repo and package ``.env`` files (optional).

Copy ``.env.example`` to ``.env`` or ``mm/.env.example`` to ``mm/.env`` and set
values. Variables are not overwritten if already set in the process environment
(``override=False``). The package file is loaded after the repo file so it can
provide mm-specific defaults when ``override=True`` is used.
"""

from __future__ import annotations

from pathlib import Path


def mm_package_root() -> Path:
    return Path(__file__).resolve().parent


def repo_root() -> Path:
    return mm_package_root().parent


def load_mm_env(*, override: bool = False) -> bool:
    """Load repo ``.env`` and ``mm/.env`` if present.

    Returns True if python-dotenv loaded at least one file, else False.
    """

    try:
        from dotenv import load_dotenv
    except ImportError:
        return False
    loaded = False
    for env_path in (repo_root() / ".env", mm_package_root() / ".env"):
        if env_path.is_file():
            load_dotenv(env_path, override=override)
            loaded = True
    return loaded
