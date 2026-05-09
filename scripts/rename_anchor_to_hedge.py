#!/usr/bin/env python3
"""Historical note: bulk anchor→hedge rename was applied to this repo (2026).

Do not re-run blindly: adjust SKIP_PARTS/EXTS and transform() if you need a similar migration elsewhere.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_PARTS = frozenset({"nobitex-docs-api", ".git", "__pycache__", "node_modules"})
EXTS = {".py", ".js", ".md", ".html", ".yaml", ".yml"}


def skip(path: Path) -> bool:
    return any(part in SKIP_PARTS for part in path.parts)


def transform(text: str) -> str:
    text = text.replace("anchor_hedge_", "hedge_")
    text = text.replace("AnchorGaussianRepPriceModel", "HedgeGaussianRepPriceModel")
    text = text.replace("anchor_", "hedge_")
    text = text.replace("_anchor_", "_hedge_")
    text = text.replace('"anchor"', '"hedge"')
    text = text.replace("'anchor'", "'hedge'")
    text = re.sub(r"([a-zA-Z0-9])_anchor\b", r"\1_hedge", text)
    return text


def main() -> None:
    for path in ROOT.rglob("*"):
        if not path.is_file() or skip(path) or path.suffix not in EXTS:
            continue
        if path.resolve() == Path(__file__).resolve():
            continue
        raw = path.read_text(encoding="utf-8")
        new = transform(raw)
        if new != raw:
            path.write_text(new, encoding="utf-8")
            print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
