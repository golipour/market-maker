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
    "cx/gpt-5.6-terra-max",
    "cx/gpt-5.6-terra-xhigh",
    "cx/gpt-5.6-terra-high",
    "cx/gpt-5.6-terra-medium",
    "cx/gpt-5.6-terra-low",
    "cx/gpt-5.5",
    "cx/gpt-5.5-xhigh",
    "cx/gpt-5.5-high",
    "cx/gpt-5.5-medium",
    "cx/gpt-5.5-low",
    "xao/grok-4.5",
    "xao/grok-4.3",
    "xao/grok-4.20-0309-reasoning",
    "nvidia/z-ai/glm-5.2",
    "nvidia/openai/gpt-oss-120b",
    "nvidia/nvidia/nemotron-3-ultra-550b-a55b",
    "agentrouter/claude-opus-4-8",
    "agentrouter/claude-opus-5",
    "agentrouter/gpt-5.6-sol",
    "ollamacloud/minimax-m3",
    "ollamacloud/nemotron-3-ultra",
    "trk/moonshotai/kimi-k3-free",
    "oc/big-pickle",
    "oc/deepseek-v4-flash-free",
    "oc/mimo-v2.5-free",
    "oc/nemotron-3-ultra-free",
    "oc/laguna-s-2.1-free",
    "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free",
    "openrouter/openai/gpt-oss-20b:free",
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
