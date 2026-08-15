#!/usr/bin/env python3
"""Add custom models to Cursor. Quit Cursor completely (Cmd+Q) before running."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

DB_PATH = Path.home() / "Library/Application Support/Cursor/User/globalStorage/state.vscdb"
KEY = (
    "src.vs.platform.reactivestorage.browser.reactiveStorageServiceImpl"
    ".persistentStorage.applicationUser"
)

NEW_MODELS = [
    "pol/gpt-5.6-sol",
    "pol/gpt-5.6-terra",
    "pol/gpt-5.4",
    "pol/deepseek-pro",
    "pol/Catniti/deepseek-v4-flash",
    "pol/MarcosFRG/deepseek-v4-pro",
    "pol/vendouple/deepseek-v4",
    "pol/vendouple/deepseek-v4-pro",
    "pol/grok-large",
    "pol/CloudCompile/grok-4.5",
    "pol/claude",
    "pol/claude-large",
    "pol/vendouple/claude-opus-4-8",
    "pol/MarcosFRG/gemini-3.1-pro-preview",
    "pol/gemini-large",
    "pol/gemini",
    "pol/gemini-fast",
    "pol/gemini-fast"
]


def cursor_running() -> bool:
    try:
        out = subprocess.check_output(["pgrep", "-x", "Cursor"], text=True).strip()
        return bool(out)
    except subprocess.CalledProcessError:
        return False


def main() -> int:
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
        ai = data.setdefault("aiSettings", {})
        user_added = list(ai.get("userAddedModels") or [])
        enabled = list(ai.get("modelOverrideEnabled") or [])
        disabled = list(ai.get("modelOverrideDisabled") or [])

        existing = set(user_added)
        added: list[str] = []
        for m in NEW_MODELS:
            if m not in existing:
                user_added.append(m)
                added.append(m)
                existing.add(m)
            if m not in enabled:
                enabled.append(m)
            if m in disabled:
                disabled = [x for x in disabled if x != m]

        ai["userAddedModels"] = user_added
        ai["modelOverrideEnabled"] = enabled
        ai["modelOverrideDisabled"] = disabled
        data["aiSettings"] = ai

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
    ua = data["aiSettings"]["userAddedModels"]
    missing = [m for m in NEW_MODELS if m not in ua]
    conn.close()

    print(f"Added {len(added)} new models (total custom: {len(ua)})")
    for m in added:
        print(f"  + {m}")
    if missing:
        print("ERROR: verify failed, missing:", missing, file=sys.stderr)
        return 1

    print("\nDone. Reopen Cursor and check Settings → Models.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
