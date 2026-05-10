"""
Append-only **step journal** (JSON lines). Engine triggers when ``MM_STEP_JOURNAL_PATH`` is set.

Stores compact scalars from ``ParsedStep`` — not full order books (deep replay needs venue feeds).
"""

from __future__ import annotations

import json
import os
from typing import Any

from mm.layers.core.app.parsed_step import ParsedStep


def append_step_journal(path: str, parsed: ParsedStep) -> None:
    """Best-effort append; never raises."""
    if not path or not str(path).strip():
        return
    p = os.path.expanduser(str(path).strip())
    try:
        d = os.path.dirname(p)
        if d:
            os.makedirs(d, exist_ok=True)
    except Exception:
        pass
    try:
        row = compact_step_dict(parsed)
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:
        return


def compact_step_dict(parsed: ParsedStep) -> dict[str, Any]:
    ob = parsed.ob
    return {
        "t_end_ms": int(parsed.t_end_ms),
        "ob_mid": float(getattr(ob, "mid", 0.0) or 0.0),
        "buy_lots": int(parsed.buy_lots),
        "sell_lots": int(parsed.sell_lots),
        "inventory": int(parsed.inventory),
        "cash": float(parsed.cash),
        "main_buy_ref": float(parsed.main_buy_ref),
        "main_sell_ref": float(parsed.main_sell_ref),
        "hedge_buy_ref": parsed.hedge_buy_ref,
        "hedge_sell_ref": parsed.hedge_sell_ref,
        "hedge_inventory": int(parsed.hedge_inventory),
        "hedge_cash": float(parsed.hedge_cash),
    }


def journal_enabled_path() -> str | None:
    v = os.environ.get("MM_STEP_JOURNAL_PATH", "").strip()
    return v or None
