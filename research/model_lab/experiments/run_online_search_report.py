"""Run online model-search experiments and write JSON + Markdown report."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def _repo_root() -> Path:
    root = Path(__file__).resolve().parents[3]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def spec_from(kind: str, **config: object) -> str:
    if not config:
        return kind
    return f"{kind}{json.dumps(config, separators=(',', ':'))}"


def evaluate_specs(specs: list[str], steps, *, min_updates: int = 1):
    from research.model_lab.core.walk_forward import run_walk_forward
    from research.model_lab.models.registry import build_model

    results = {}
    total = len(specs)
    for index, spec in enumerate(specs, start=1):
        print(f"[{index}/{total}] {spec}", flush=True)
        model = build_model(spec)
        results[spec] = run_walk_forward(model, steps, min_updates_before_score=min_updates)
    return results


def build_spec_grids() -> tuple[list[str], list[str], list[str], list[str]]:
    flow_specs: list[str] = []
    flow_kinds = ["hurdle_lognormal", "hurdle_gamma", "hurdle_weibull", "gamma", "weibull"]
    decays = [0.80, 0.90, 0.95, 0.975, 0.99, 0.995]
    for kind in flow_kinds:
        for decay in decays:
            if kind == "hurdle_lognormal":
                flow_specs.append(
                    spec_from(f"flow:{kind}", decay=decay, init_p0=0.3, init_mu=0.0, init_sigma=1.0)
                )
            else:
                flow_specs.append(spec_from(f"flow:{kind}", decay=decay))
    for init_p0 in [0.1, 0.3, 0.5, 0.6, 0.7]:
        flow_specs.append(
            spec_from("flow:hurdle_lognormal", decay=0.95, init_p0=init_p0, init_mu=0.0, init_sigma=1.0)
        )

    price_specs: list[str] = []
    for decay in [0.95, 0.975, 0.99, 0.995]:
        for reference_vwap_vol in [300.0, 500.0, 1000.0, 2000.0]:
            for shape_alpha in [0.3, 0.5, 0.7, 1.0]:
                price_specs.append(
                    spec_from(
                        "price:conditional_mechanical_mixture",
                        decay=decay,
                        reference_vwap_vol=reference_vwap_vol,
                        shape_alpha=shape_alpha,
                        student_df=3.0,
                        initial_sigma=50.0,
                        initial_log_sigma=0.25,
                    )
                )
    for residual_decay in [0.80, 0.90, 0.95, 0.975, 0.99, 0.995]:
        for residual_sigma in [25.0, 50.0, 100.0, 200.0]:
            price_specs.append(
                spec_from(
                    "price:residual",
                    residual_decay=residual_decay,
                    residual_sigma=residual_sigma,
                    reference_vwap_vol=1000.0,
                )
            )
    for decay in [0.90, 0.95, 0.99, 0.995]:
        for initial_sigma in [25.0, 50.0, 100.0, 200.0]:
            price_specs.append(spec_from("baseline:mid_random_walk", decay=decay, initial_sigma=initial_sigma))

    experimental_flow: list[str] = []
    for decay in [0.90, 0.95, 0.975, 0.99]:
        for df in [3.0, 5.0, 8.0]:
            experimental_flow.append(
                spec_from(
                    "flow:hurdle_log_student_t",
                    decay=decay,
                    init_p0=0.3,
                    init_mu=0.0,
                    init_sigma=1.0,
                    init_df=df,
                )
            )
        experimental_flow.append(
            spec_from("flow:bivariate_hurdle_lognormal", decay=decay, init_mu=0.0, init_sigma=1.0)
        )

    experimental_price: list[str] = []
    for decay in [0.90, 0.95, 0.99, 0.995]:
        for df in [3.0, 5.0, 8.0]:
            experimental_price.append(
                spec_from("price:student_t_mid_delta", decay=decay, initial_sigma=50.0, df=df)
            )
        experimental_price.append(
            spec_from("price:flow_conditioned_mid_delta", decay=decay, variance_decay=decay, initial_sigma=50.0)
        )

    return flow_specs, price_specs, experimental_flow, experimental_price


def _fmt(value) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def render_markdown_report(payload: dict) -> str:
    ds = payload["dataset"]
    lines = [
        "# Online Model Search Report",
        "",
        f"**Run ID:** `{payload['run_id']}`  ",
        f"**Generated:** {payload['generated_at']}  ",
        f"**Venue:** {ds['exchange']} / {ds['symbol']} / {ds['interval_ms']}ms  ",
        f"**Window:** {ds['since']} → {ds['until']}  ",
        f"**Steps:** {ds['n_steps']} ({ds['n_scored']} scored transitions)  ",
        f"**Zero-volume fraction:** {ds['zero_volume_fraction']:.1%}  ",
        f"**Mid range:** {ds['min_mid']:.0f} – {ds['max_mid']:.0f} IRT  ",
        "",
        "## Evaluation contract",
        "",
        "Prequential walk-forward: at step *t*, score with θ_{t-1}, then update. Lower mean NLL is better within the same family.",
        "",
        "## Production baselines",
        "",
        "| Model | Mean NLL | Median NLL | Scored |",
        "|-------|----------|------------|--------|",
    ]
    for row in payload["baselines"]:
        lines.append(
            f"| `{row['label']}` | {_fmt(row['mean_nll'])} | {_fmt(row['median_nll'])} | {row['n_scored']} |"
        )

    lines.extend(["", "## Flow family — top 10 by mean NLL", ""])
    lines.extend(["| Rank | Model | Mean NLL | Median NLL |", "|------|-------|----------|------------|"])
    for idx, row in enumerate(payload["flow_top"], start=1):
        lines.append(
            f"| {idx} | `{row['label']}` | {_fmt(row['mean_nll'])} | {_fmt(row['median_nll'])} |"
        )

    lines.extend(["", "### Flow vs production baseline (paired ΔNLL, negative = candidate wins)", ""])
    lines.extend(
        ["| Model | Mean ΔNLL | Win rate | N |", "|-------|-----------|----------|---|"]
    )
    for row in payload["flow_pairwise"][:10]:
        lines.append(
            f"| `{row['label']}` | {_fmt(row['mean_delta_nll'])} | {row['win_rate']:.1%} | {row['n']} |"
        )

    lines.extend(["", "### Flow NLL by volume regime (best model per regime)", ""])
    lines.extend(["| Regime | Best model | Mean NLL | N |", "|--------|------------|----------|---|"])
    for row in payload["flow_regime_winners"]:
        lines.append(f"| {row['regime']} | `{row['label']}` | {_fmt(row['mean_nll'])} | {row['n']} |")

    lines.extend(["", "## Price family — top 10 by mean NLL", ""])
    lines.extend(["| Rank | Model | Mean NLL | Median NLL |", "|------|-------|----------|------------|"])
    for idx, row in enumerate(payload["price_top"], start=1):
        lines.append(
            f"| {idx} | `{row['label']}` | {_fmt(row['mean_nll'])} | {_fmt(row['median_nll'])} |"
        )

    lines.extend(["", "### Price vs production baseline (paired ΔNLL)", ""])
    lines.extend(["| Model | Mean ΔNLL | Win rate | N |", "|-------|-----------|----------|---|"])
    for row in payload["price_pairwise"][:10]:
        lines.append(
            f"| `{row['label']}` | {_fmt(row['mean_delta_nll'])} | {row['win_rate']:.1%} | {row['n']} |"
        )

    lines.extend(["", "### Price NLL by volume regime (best model per regime)", ""])
    lines.extend(["| Regime | Best model | Mean NLL | N |", "|--------|------------|----------|---|"])
    for row in payload["price_regime_winners"]:
        lines.append(f"| {row['regime']} | `{row['label']}` | {_fmt(row['mean_nll'])} | {row['n']} |")

    lines.extend(["", "## Experimental candidates (research-only)", ""])
    lines.extend(["| Family | Model | Mean NLL | vs baseline ΔNLL |", "|--------|-------|----------|------------------|"])
    for row in payload["experimental_highlights"]:
        lines.append(
            f"| {row['family']} | `{row['label']}` | {_fmt(row['mean_nll'])} | {_fmt(row.get('mean_delta_nll'))} |"
        )

    lines.extend(["", "## Stationarity diagnostics", ""])
    lines.append(
        "Block std of transformed series (lower drift across blocks suggests fixed-weight / neural models may be viable):"
    )
    lines.append("")
    lines.extend(["| Block | log_return_std | spread_norm_delta_std | normalized_volume_std |", "|-------|----------------|----------------------|------------------------|"])
    for row in payload["stationarity"]:
        lines.append(
            f"| {row['block']} | {_fmt(row.get('log_return_std'))} | {_fmt(row.get('spread_norm_delta_std'))} | {_fmt(row.get('normalized_volume_std'))} |"
        )

    lines.extend(["", "## Findings", ""])
    for bullet in payload["findings"]:
        lines.append(f"- {bullet}")

    lines.extend(["", "## Recommendations", ""])
    for bullet in payload["recommendations"]:
        lines.append(f"- {bullet}")

    return "\n".join(lines) + "\n"


def regime_winners(strata: list[dict], *, family_prefix: str | None = None) -> list[dict]:
    by_regime: dict[str, list[dict]] = {}
    for row in strata:
        if family_prefix and not str(row["label"]).startswith(family_prefix):
            continue
        by_regime.setdefault(str(row["volume_regime"]), []).append(row)
    winners = []
    for regime, rows in sorted(by_regime.items()):
        best = min(rows, key=lambda r: float(r["mean_nll"]))
        winners.append(
            {
                "regime": regime,
                "label": best["label"],
                "mean_nll": best["mean_nll"],
                "n": best["n"],
            }
        )
    return winners


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exchange", default="nobitex")
    parser.add_argument("--symbol", default="USDT_IRT")
    parser.add_argument("--interval-ms", type=int, default=30000)
    parser.add_argument("--since", default="48h")
    parser.add_argument("--source", default="auto")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    root = _repo_root()
    from mm.env_loader import load_mm_env
    from research.model_lab.core.analysis import (
        leaderboard,
        paired_nll_deltas,
        stationarity_report,
        stratified_nll,
    )
    from research.model_lab.data.loader import load_interval_steps, parse_since

    load_mm_env()
    until = datetime.now(timezone.utc)
    since = parse_since(args.since, until=until)

    steps = load_interval_steps(
        exchange=args.exchange,
        symbol=args.symbol,
        interval_ms=args.interval_ms,
        since=since,
        until=until,
        source=args.source,
    )
    if len(steps) < 2:
        print(f"Need >= 2 steps, got {len(steps)}", file=sys.stderr)
        return 1

    flow_specs, price_specs, experimental_flow, experimental_price = build_spec_grids()
    price_baseline = spec_from(
        "price:conditional_mechanical_mixture",
        decay=0.99,
        reference_vwap_vol=1000.0,
        shape_alpha=0.5,
        student_df=3.0,
        initial_sigma=50.0,
        initial_log_sigma=0.25,
    )
    flow_baseline = spec_from("flow:hurdle_lognormal", decay=0.95, init_p0=0.3, init_mu=0.0, init_sigma=1.0)
    baseline_specs = [
        "baseline:zero_flow",
        "baseline:mid_random_walk",
        flow_baseline,
        price_baseline,
        spec_from("price:residual", residual_decay=0.95, residual_sigma=50.0, reference_vwap_vol=1000.0),
    ]

    print(f"Loaded {len(steps)} steps. Running experiments...")
    baseline_results = evaluate_specs(baseline_specs, steps)
    flow_results = evaluate_specs(flow_specs, steps)
    price_results = evaluate_specs(price_specs, steps)
    experimental_results = evaluate_specs(experimental_flow + experimental_price, steps)

    all_flow = {**flow_results, **{k: v for k, v in experimental_results.items() if k.startswith("flow:")}}
    all_price = {**price_results, **{k: v for k, v in experimental_results.items() if k.startswith("price:")}}
    for spec, result in baseline_results.items():
        if spec.startswith("flow:"):
            all_flow.setdefault(spec, result)
        if spec.startswith("price:") or spec.startswith("baseline:mid"):
            all_price.setdefault(spec, result)

    flow_board = [r for r in leaderboard(all_flow) if r["family"] == "flow"]
    price_board = [r for r in leaderboard(all_price) if r["family"] == "price"]
    flow_pairwise = paired_nll_deltas(all_flow, baseline_label=flow_baseline)
    price_pairwise = paired_nll_deltas(all_price, baseline_label=price_baseline)
    flow_strata = stratified_nll(all_flow, steps)
    price_strata = stratified_nll(all_price, steps)
    stationarity = stationarity_report(steps, blocks=8)

    volumes = [s.total_volume for s in steps]
    mids = [s.mid for s in steps]
    zero_frac = sum(1 for v in volumes if v <= 0.0) / len(volumes)

    best_flow = flow_board[0] if flow_board else {}
    best_price = price_board[0] if price_board else {}
    prod_flow_row = leaderboard({flow_baseline: all_flow[flow_baseline]})[0]
    prod_price_row = leaderboard({price_baseline: all_price[price_baseline]})[0]

    experimental_highlights = []
    for spec, result in experimental_results.items():
        summary = result.summary
        if summary is None:
            continue
        family = summary.family
        baseline = flow_baseline if family == "flow" else price_baseline
        baseline_result = all_flow[baseline] if family == "flow" else all_price[baseline]
        pairwise = paired_nll_deltas({spec: result, baseline: baseline_result}, baseline_label=baseline)
        delta = pairwise[0]["mean_delta_nll"] if pairwise else None
        experimental_highlights.append(
            {
                "family": family,
                "label": spec,
                "mean_nll": summary.mean_nll,
                "mean_delta_nll": delta,
            }
        )
    experimental_highlights.sort(key=lambda r: float(r["mean_nll"] or 1e18))

    findings = [
        f"Dataset is sparse: {zero_frac:.1%} of 30s buckets have zero trade volume.",
        f"Best flow model: `{best_flow.get('label', 'n/a')}` with mean NLL {_fmt(best_flow.get('mean_nll'))} (production baseline: {_fmt(prod_flow_row.get('mean_nll'))}).",
        f"Best price model: `{best_price.get('label', 'n/a')}` with mean NLL {_fmt(best_price.get('mean_nll'))} (production baseline: {_fmt(prod_price_row.get('mean_nll'))}).",
    ]
    if flow_pairwise:
        top = flow_pairwise[0]
        findings.append(
            f"Largest flow improvement vs production: `{top['label']}` with mean ΔNLL {_fmt(top['mean_delta_nll'])} (win rate {top['win_rate']:.1%})."
        )
    if price_pairwise:
        top = price_pairwise[0]
        findings.append(
            f"Largest price improvement vs production: `{top['label']}` with mean ΔNLL {_fmt(top['mean_delta_nll'])} (win rate {top['win_rate']:.1%})."
        )

    recommendations = []
    if best_flow.get("label") != flow_baseline:
        recommendations.append(f"Consider adopting `{best_flow.get('label')}` for local flow in MPC presets (verify burst-regime NLL).")
    else:
        recommendations.append("Production hurdle lognormal (decay=0.95) remains best overall flow model on this window.")
    if best_price.get("label") != price_baseline:
        recommendations.append(f"Evaluate `{best_price.get('label')}` in shadow mode before replacing conditional mixture.")
    else:
        recommendations.append("Production conditional mechanical mixture remains best full-book price model on this window.")
    recommendations.append("Compare zero-volume vs high-volume regime tables before choosing a winner for live quoting.")
    recommendations.append("Neural sequence models are deferred unless longer history or stronger stationarity is confirmed.")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(args.output_dir) if args.output_dir else root / "research" / "model_lab" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "exchange": args.exchange,
            "symbol": args.symbol,
            "interval_ms": args.interval_ms,
            "since": since.isoformat() if since else None,
            "until": until.isoformat(),
            "n_steps": len(steps),
            "n_scored": len(steps) - 1,
            "zero_volume_fraction": zero_frac,
            "min_mid": min(mids),
            "max_mid": max(mids),
            "avg_total_volume": sum(volumes) / len(volumes),
        },
        "baselines": [
            {
                "label": spec,
                "mean_nll": baseline_results[spec].summary.mean_nll if baseline_results[spec].summary else None,
                "median_nll": baseline_results[spec].summary.median_nll if baseline_results[spec].summary else None,
                "n_scored": baseline_results[spec].summary.n_scored if baseline_results[spec].summary else 0,
            }
            for spec in baseline_specs
        ],
        "flow_top": flow_board[:15],
        "price_top": price_board[:15],
        "flow_pairwise": flow_pairwise[:15],
        "price_pairwise": price_pairwise[:15],
        "flow_regime_winners": regime_winners(flow_strata),
        "price_regime_winners": regime_winners(price_strata),
        "experimental_highlights": experimental_highlights[:12],
        "stationarity": stationarity,
        "findings": findings,
        "recommendations": recommendations,
    }

    json_path = output_dir / f"{run_id}_online_model_search_summary.json"
    md_path = output_dir / f"{run_id}_online_model_search_REPORT.md"
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    md_path.write_text(render_markdown_report(payload), encoding="utf-8")

    latest_md = output_dir / "LATEST_online_model_search_REPORT.md"
    latest_json = output_dir / "LATEST_online_model_search_summary.json"
    latest_md.write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")
    latest_json.write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {latest_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
