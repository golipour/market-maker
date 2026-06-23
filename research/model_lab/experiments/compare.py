"""Compare models with walk-forward NLL on persisted intervals."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from mm.env_loader import load_mm_env

from research.model_lab.core.walk_forward import run_walk_forward
from research.model_lab.data.loader import load_interval_steps, parse_since, resolve_database_url
from research.model_lab.models.registry import build_model, list_builtin_models


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_output_dir() -> Path:
    return _repo_root() / "research" / "model_lab" / "outputs"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Walk-forward model comparison on persisted market intervals.",
    )
    parser.add_argument("--exchange", default="nobitex")
    parser.add_argument("--symbol", default="USDT_IRT")
    parser.add_argument("--interval-ms", type=int, default=30000)
    parser.add_argument("--since", default="48h", help="e.g. 48h, 7d, 2026-06-01T00:00:00Z")
    parser.add_argument("--until", default=None, help="ISO timestamp; default now (UTC)")
    parser.add_argument(
        "--source",
        choices=["auto", "events", "market_facts"],
        default="auto",
        help="auto prefers mm_events (full book), else mm_market_facts",
    )
    parser.add_argument(
        "--models",
        default="flow:hurdle_lognormal,price:residual,price:conditional_mechanical_mixture,baseline:zero_flow,baseline:mid_random_walk",
        help="Comma-separated family:kind specs (see GUIDE.md)",
    )
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--min-updates-before-score", type=int, default=1)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--write-scores", action="store_true", help="Write per-step JSONL scores")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    load_mm_env(_repo_root() / ".env")
    args = parse_args(argv)

    if args.list_models:
        for spec in list_builtin_models():
            print(spec)
        return 0

    until = (
        datetime.fromisoformat(args.until.replace("Z", "+00:00"))
        if args.until
        else datetime.now(timezone.utc)
    )
    since = parse_since(args.since, until=until)

    steps = load_interval_steps(
        database_url=args.database_url,
        exchange=args.exchange,
        symbol=args.symbol,
        interval_ms=args.interval_ms,
        since=since,
        until=until,
        source=args.source,
    )
    if len(steps) < 2:
        print(
            f"Need at least 2 interval steps; got {len(steps)} for "
            f"{args.exchange}/{args.symbol} interval_ms={args.interval_ms}",
            file=sys.stderr,
        )
        return 1

    specs = [part.strip() for part in args.models.split(",") if part.strip()]
    output_dir = Path(args.output_dir) if args.output_dir else _default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    summary_rows: list[dict] = []

    print(
        f"Loaded {len(steps)} steps | {args.exchange}/{args.symbol} | "
        f"interval_ms={args.interval_ms} | since={since} until={until}"
    )
    print(f"Database: {resolve_database_url(args.database_url)}")
    print()

    for spec in specs:
        model = build_model(spec)
        result = run_walk_forward(
            model,
            steps,
            min_updates_before_score=args.min_updates_before_score,
        )
        summary = result.summary
        assert summary is not None
        row = {
            "model": summary.model_name,
            "kind": summary.model_kind,
            "family": summary.family,
            "n_steps": summary.n_steps,
            "n_scored": summary.n_scored,
            "n_skipped": summary.n_skipped,
            "mean_nll": summary.mean_nll,
            "median_nll": summary.median_nll,
            "total_nll": summary.total_nll,
            "mean_log_likelihood": summary.mean_log_likelihood,
            "skip_reasons": summary.skip_reasons,
        }
        summary_rows.append(row)
        print(
            f"{summary.model_name:40s}  mean_nll={summary.mean_nll!s:>12}  "
            f"median_nll={summary.median_nll!s:>12}  scored={summary.n_scored}/{summary.n_steps}"
        )

        if args.write_scores:
            scores_path = output_dir / f"{run_id}_{summary.model_name}_scores.jsonl"
            with scores_path.open("w", encoding="utf-8") as handle:
                for score in result.scores:
                    handle.write(
                        json.dumps(
                            {
                                "ts": score.ts.isoformat(),
                                "step_index": score.step_index,
                                "nll": score.nll,
                                "log_likelihood": score.log_likelihood,
                                "skip_reason": score.skip_reason,
                                "observation": score.observation,
                            },
                            default=str,
                        )
                        + "\n"
                    )

    summary_path = output_dir / f"{run_id}_summary.json"
    payload = {
        "run_id": run_id,
        "exchange": args.exchange,
        "symbol": args.symbol,
        "interval_ms": args.interval_ms,
        "since": since.isoformat() if since else None,
        "until": until.isoformat(),
        "n_steps": len(steps),
        "models": summary_rows,
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print()
    print(f"Wrote summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
