"""Regression: duplicate state_key in one batch must not break mm_latest_state upsert."""

from __future__ import annotations

from mm.persistence.postgres import _dedupe_latest_rows_by_state_key


def test_dedupe_latest_rows_keeps_last_per_state_key() -> None:
    rows = [
        ("k1", "t1", "e1", "", None, {}, {}),
        ("k1", "t1", "e2", "", None, {}, {}),
        ("k2", "t2", "e3", "", None, {}, {}),
    ]
    out = _dedupe_latest_rows_by_state_key(rows)
    assert len(out) == 2
    assert out[0][2] == "e2"
    assert out[1][0] == "k2"
