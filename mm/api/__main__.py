"""Control plane / admin API entrypoint.

Run: ``python3 -m mm.api.admin``
"""

from __future__ import annotations

import os

import uvicorn

from mm.env_loader import load_mm_env


def main() -> None:
    load_mm_env()
    host = os.environ.get("MM_ADMIN_HOST", "0.0.0.0")
    port = int(os.environ.get("MM_ADMIN_PORT", "8080"))
    uvicorn.run("mm.api.admin:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
