#!/usr/bin/env python3
"""Set Cursor Override OpenAI Base URL. Quit Cursor completely (Cmd+Q) before running."""

from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

DB_PATH = Path.home() / "Library/Application Support/Cursor/User/globalStorage/state.vscdb"
KEY = (
    "src.vs.platform.reactivestorage.browser.reactiveStorageServiceImpl"
    ".persistentStorage.applicationUser"
)


def cursor_running() -> bool:
    try:
        out = subprocess.check_output(["pgrep", "-x", "Cursor"], text=True).strip()
        return bool(out)
    except subprocess.CalledProcessError:
        return False


def normalize_base_url(url: str) -> str:
    url = url.strip()
    if not url:
        raise ValueError("URL must not be empty")

    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("URL must be absolute, e.g. https://example.com/v1")

    # Cursor appends /chat/completions; base should usually end with /v1
    return url.rstrip("/")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Set Cursor's Override OpenAI Base URL in local state."
    )
    parser.add_argument(
        "url",
        help="OpenAI-compatible base URL, e.g. https://gateway.example.com/v1",
    )
    parser.add_argument(
        "--enable-openai-key",
        action="store_true",
        help="Also set useOpenAIKey=true (needed for some BYOK setups).",
    )
    args = parser.parse_args()

    try:
        base_url = normalize_base_url(args.url)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    if not DB_PATH.exists():
        print(f"ERROR: Cursor DB not found at {DB_PATH}", file=sys.stderr)
        return 1

    if cursor_running():
        print("ERROR: Cursor is still running.")
        print("Quit Cursor fully with Cmd+Q, wait a second, then re-run this script.")
        return 2

    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    try:
        conn.execute("PRAGMA busy_timeout = 30000")
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        row = conn.execute("SELECT value FROM ItemTable WHERE key = ?", (KEY,)).fetchone()
        if not row:
            print("ERROR: reactive storage key missing", file=sys.stderr)
            return 1

        data = json.loads(row[0])
        old = data.get("openAIBaseUrl")
        data["openAIBaseUrl"] = base_url
        if args.enable_openai_key:
            data["useOpenAIKey"] = True

        conn.execute(
            "INSERT OR REPLACE INTO ItemTable (key, value) VALUES (?, ?)",
            (KEY, json.dumps(data, separators=(",", ":"))),
        )
        conn.commit()
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        conn.close()

    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    row = conn.execute("SELECT value FROM ItemTable WHERE key = ?", (KEY,)).fetchone()
    data = json.loads(row[0])
    current = data.get("openAIBaseUrl")
    conn.close()

    if current != base_url:
        print(f"ERROR: verify failed, got {current!r}", file=sys.stderr)
        return 1

    print(f"Previous: {old}")
    print(f"Updated:  {current}")
    print(f"useOpenAIKey: {data.get('useOpenAIKey')}")
    print("\nDone. Reopen Cursor and check Settings → Models.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
