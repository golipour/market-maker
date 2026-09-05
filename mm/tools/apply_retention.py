"""Apply persistence retention and tiering policies.

Run:
    python -m mm.tools.apply_retention --database-url "$MM_DATABASE_URL"
    python -m mm.tools.apply_retention --parquet-root ./data/parquet --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from mm.persistence.retention import RetentionPolicy, RetentionResult, apply_retention, retention_policy_from_mapping
from mm.runtime.config import load_runtime_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Optional runtime JSON config with persistence.retention")
    parser.add_argument("--database-url", default=os.environ.get("MM_DATABASE_URL", ""))
    parser.add_argument("--parquet-root", default=os.environ.get("MM_PARQUET_ROOT", ""))
    parser.add_argument("--spool-dir", default="persistence_spool/postgres")
    parser.add_argument("--events-days", type=int, default=None)
    parser.add_argument("--facts-days", type=int, default=None)
    parser.add_argument("--latest-state-days", type=int, default=None)
    parser.add_argument("--admin-audit-days", type=int, default=None)
    parser.add_argument("--parquet-days", type=int, default=None)
    parser.add_argument("--spool-archive-days", type=int, default=None)
    parser.add_argument("--spool-failed-days", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-canary", action="store_true", help="Do not refresh the Postgres retention canary row.")
    args = parser.parse_args(argv)

    policy = _resolve_policy(args)
    database_url = str(args.database_url or "").strip() or None
    parquet_root = str(args.parquet_root or "").strip() or None
    if args.config and not database_url:
        runtime = load_runtime_config(args.config)
        database_url = runtime.persistence.resolved_database_url()
        if parquet_root is None:
            parquet_root = runtime.persistence.resolved_parquet_root()
    elif not args.config:
        # Auto-load the in-cluster configmap if the file is mounted and no
        # explicit overrides were given. Lets the cron job be config-driven.
        candidate = Path("/config/runtime.json")
        if candidate.is_file():
            try:
                runtime = load_runtime_config(str(candidate))
                if database_url is None:
                    database_url = runtime.persistence.resolved_database_url()
                if parquet_root is None:
                    parquet_root = runtime.persistence.resolved_parquet_root()
                if not any(
                    getattr(args, name) is not None
                    for name in (
                        "events_days",
                        "facts_days",
                        "latest_state_days",
                        "admin_audit_days",
                        "parquet_days",
                        "spool_archive_days",
                        "spool_failed_days",
                    )
                ):
                    policy = runtime.persistence.retention
            except Exception:
                pass

    if database_url is None and parquet_root is None and not Path(args.spool_dir).expanduser().exists():
        parser.error("provide --database-url, --parquet-root, and/or an existing --spool-dir")

    result = apply_retention(
        policy,
        database_url=database_url,
        parquet_root=parquet_root,
        spool_dir=args.spool_dir,
        dry_run=bool(args.dry_run),
        write_canary=not bool(args.skip_canary),
    )
    print(json.dumps(_result_to_dict(result), sort_keys=True))
    return 0


def _resolve_policy(args: argparse.Namespace) -> RetentionPolicy:
    base = RetentionPolicy()
    if args.config:
        runtime = load_runtime_config(args.config)
        base = runtime.persistence.retention
    overrides = {
        "events_days": args.events_days,
        "facts_days": args.facts_days,
        "latest_state_days": args.latest_state_days,
        "admin_audit_days": args.admin_audit_days,
        "parquet_days": args.parquet_days,
        "spool_archive_days": args.spool_archive_days,
        "spool_failed_days": args.spool_failed_days,
    }
    fields = {
        name: overrides[name] if overrides[name] is not None else getattr(base, name)
        for name in overrides
    }
    return RetentionPolicy(**fields)


def _result_to_dict(result: RetentionResult) -> dict[str, Any]:
    canary_age_ms = getattr(result, "postgres_canary_age_ms", 0)
    return {
        "dry_run": result.dry_run,
        "postgres_canary_age_ms": int(canary_age_ms),
        "postgres": {
            "rows_affected": result.postgres.rows_affected,
            "tables": [
                {
                    "table": item.table,
                    "tier": item.tier,
                    "retention_days": item.retention_days,
                    "rows_affected": item.rows_affected,
                }
                for item in result.postgres.tables
            ],
        },
        "parquet": {
            "partitions_removed": result.parquet.partitions_removed,
            "files_removed": result.parquet.files_removed,
            "bytes_removed": result.parquet.bytes_removed,
        },
        "spool": {
            "files_removed": result.spool.files_removed,
            "bytes_removed": result.spool.bytes_removed,
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
