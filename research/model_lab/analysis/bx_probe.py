"""Per-symbol (indexed) probes for Bitwana + Nobitex MarketEvent history."""
import sys

import psycopg2

DSN = "host=127.0.0.1 port=15432 user=mm password=mm dbname=postgres connect_timeout=10"


def run(cur, sql, args=()):
    try:
        cur.execute(sql, args)
        return cur.fetchall()
    except Exception as exc:  # noqa: BLE001
        try:
            cur.execute("ROLLBACK")
        except Exception:
            pass
        return [("ERR", str(exc)[:80])]


def main() -> int:
    with psycopg2.connect(DSN) as conn:
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute("SET statement_timeout = '20s'")

        print("== symbols seen in last 6h ==", flush=True)
        for r in run(cur, "SELECT exchange, symbol, event_type, count(*) FROM mm_events "
                          "WHERE ts > now() - interval '6 hours' AND exchange IN "
                          "('bitwana','nobitex') GROUP BY 1,2,3 HAVING count(*) > 50 "
                          "ORDER BY 4 DESC"):
            print("  ", r, flush=True)

        for ex in ("bitwana", "nobitex"):
            print(f"\n== {ex}: MarketEvent span by symbol (indexed) ==", flush=True)
            for r in run(cur, "SELECT symbol, count(*), min(ts), max(ts) FROM mm_events "
                              "WHERE exchange=%s AND ts > now() - interval '7 days' "
                              "AND event_type='MarketEvent' GROUP BY 1 ORDER BY 2 DESC",
                          (ex,)):
                print("  ", r, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
