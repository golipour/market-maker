"""Inventory the available Bitwana + Nobitex history (fast, index-friendly)."""
import sys

import psycopg2

DSN = "host=127.0.0.1 port=15432 user=mm password=mm dbname=postgres connect_timeout=10"


def main() -> int:
    with psycopg2.connect(DSN) as conn:
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute("SET statement_timeout = '25s'")
        cur.execute("SELECT now()")
        print("now:", cur.fetchone()[0], flush=True)

        print("\n== indexes on mm_events ==", flush=True)
        try:
            cur.execute(
                "SELECT indexname, indexdef FROM pg_indexes "
                "WHERE tablename IN ('mm_events','mm_market_facts')"
            )
            for nm, df in cur.fetchall():
                print(" ", nm, "|", df[:160], flush=True)
        except Exception as exc:  # noqa: BLE001
            print("  index lookup failed:", exc, flush=True)

        for ex in ("bitwana", "nobitex"):
            print(f"\n== {ex} ==", flush=True)
            for label, sql, args in (
                (
                    "events count",
                    "SELECT count(*) FROM mm_events "
                    "WHERE event_type='MarketEvent' AND exchange=%s",
                    (ex,),
                ),
                (
                    "events symbols",
                    "SELECT symbol, count(*) FROM mm_events "
                    "WHERE event_type='MarketEvent' AND exchange=%s GROUP BY 1",
                    (ex,),
                ),
                (
                    "market_facts",
                    "SELECT symbol, interval_ms, count(*), min(ts), max(ts) "
                    "FROM mm_market_facts WHERE exchange=%s GROUP BY 1,2 ORDER BY 3 DESC",
                    (ex,),
                ),
            ):
                try:
                    cur.execute(sql, args)
                    for r in cur.fetchall():
                        print(f"  {label}: {r}", flush=True)
                except Exception as exc:  # noqa: BLE001
                    print(f"  {label}: ERROR {exc}", flush=True)
                    conn.rollback()
    return 0


if __name__ == "__main__":
    sys.exit(main())
