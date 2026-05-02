"""
Load key=value pairs from a repo-root `.env` file (optional `python-dotenv`).

Does not override variables already set in the process environment.
"""

from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_local_env() -> None:
    """
    Best-effort load of `<repo>/.env`. No-op if python-dotenv is missing or file absent.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    env_file = repo_root() / ".env"
    if not env_file.is_file():
        return

    # Existing OS env wins (override=False) so `VAR=x python -m mm_runtime` still works.
    load_dotenv(env_file, override=False)
