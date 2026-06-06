"""Fit simple order-flow regressions against persisted market events.

Example:
    python -m mm.tools.orderflow_price_regression \
        --local-exchange wallex --global-exchange nobitex --symbol USDT_IRT \
        --interval-ms 10000 --lookbacks 1h,6h,24h,7d,all

    python -m mm.tools.orderflow_price_regression --rolling-window 1h --rolling-count 8

    python -m mm.tools.orderflow_price_regression --side-targets --rolling-window 30m
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, Sequence

import numpy as np

from mm.env_loader import load_mm_env


@dataclass(frozen=True, slots=True)
class MarketRow:
    ts: datetime
    buy_volume: float
    sell_volume: float
    best_bid: float
    best_ask: float
    best_bid_qty: float
    best_ask_qty: float
    bid_depth_5: float
    ask_depth_5: float
    microprice: float
    bids: list[tuple[float, float]]
    asks: list[tuple[float, float]]
    inventory: float = float("nan")
    cash: float = float("nan")

    @property
    def mid(self) -> float:
        return 0.5 * (self.best_bid + self.best_ask)

    @property
    def signed_flow(self) -> float:
        return self.buy_volume - self.sell_volume

    @property
    def imbalance(self) -> float:
        denom = self.buy_volume + self.sell_volume
        if denom <= 0:
            return 0.0
        return self.signed_flow / denom

    @property
    def spread(self) -> float:
        return self.best_ask - self.best_bid

    @property
    def queue_imbalance_5(self) -> float:
        denom = self.bid_depth_5 + self.ask_depth_5
        if denom <= 0:
            return 0.0
        return (self.bid_depth_5 - self.ask_depth_5) / denom

    @property
    def microprice_minus_mid(self) -> float:
        return self.microprice - self.mid


@dataclass(frozen=True, slots=True)
class FitResult:
    n: int
    y_name: str
    feature_names: list[str]
    beta: np.ndarray
    stderr: np.ndarray
    t_stats: np.ndarray
    sigma2_mle: float
    rmse: float
    r2: float
    corr: list[float]


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    n: int
    rmse: float
    r2: float
    baseline_rmse: float
    zero_baseline_rmse: float
    direction_accuracy: float
    baseline_direction_accuracy: float
    zero_baseline_direction_accuracy: float


@dataclass(frozen=True, slots=True)
class TrainTestResult:
    train: EvaluationResult
    test: EvaluationResult
    split_ratio: float


@dataclass(frozen=True, slots=True)
class VwapFullEvaluation:
    n: int
    full_rmse: float
    full_r2: float
    mechanical_rmse: float
    mechanical_r2: float
    mean_baseline_rmse: float


@dataclass(frozen=True, slots=True)
class VwapDesign:
    residual: np.ndarray
    x_without_intercept: np.ndarray
    feature_names: list[str]
    y_name: str
    observed_delta: np.ndarray
    mechanical_delta: np.ndarray


@dataclass(frozen=True, slots=True)
class OnlineEvaluation:
    n_scored: int
    rmse: float
    nll: float
    mean_rmse: float
    mean_nll: float
    zero_rmse: float
    zero_nll: float
    mechanical_rmse: float
    final_beta: np.ndarray
    final_mean: float
    final_sigma: float
    final_mean_sigma: float
    final_zero_sigma: float


@dataclass(frozen=True, slots=True)
class DecomposedHurdleEvaluation:
    n_scored: int
    active_fraction: float
    buy_rmse: float
    buy_nll: float
    buy_zero_rmse: float
    buy_zero_nll: float
    buy_mechanical_rmse: float
    sell_rmse: float
    sell_nll: float
    sell_zero_rmse: float
    sell_zero_nll: float
    sell_mechanical_rmse: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a 3-step order-flow model from mm_events MarketEvent rows. "
            "The default target is log mid-price return."
        )
    )
    parser.add_argument("--database-url-env", default="MM_DATABASE_URL")
    parser.add_argument("--local-exchange", default="wallex")
    parser.add_argument("--global-exchange", default="nobitex")
    parser.add_argument("--symbol", default="USDT_IRT")
    parser.add_argument("--interval-ms", type=int, default=10000)
    parser.add_argument(
        "--lookbacks",
        default="1h,6h,24h,7d,all",
        help="Comma-separated windows ending at the latest event per venue. Use s/m/h/d or all.",
    )
    parser.add_argument(
        "--rolling-window",
        default=None,
        help="Fixed window size for separate past windows, e.g. 1h. Overrides --lookbacks when set.",
    )
    parser.add_argument(
        "--rolling-count",
        type=int,
        default=8,
        help="Number of fixed past windows to evaluate when --rolling-window is set.",
    )
    parser.add_argument(
        "--target",
        choices=("log_return", "delta"),
        default="log_return",
        help="Dependent variable: log(mid_t / mid_t-1) or mid_t - mid_t-1.",
    )
    parser.add_argument(
        "--features",
        choices=("signed", "imbalance", "buy_sell"),
        default="signed",
        help="Regressors to lag from t back through t-3.",
    )
    parser.add_argument("--lags", type=int, default=3, help="Number of lagged steps behind current flow.")
    parser.add_argument("--min-rows", type=int, default=30)
    parser.add_argument(
        "--side-targets",
        action="store_true",
        help="Fit best_bid from sell volume and best_ask from buy volume instead of mid from selected features.",
    )
    parser.add_argument(
        "--book-features",
        action="store_true",
        help="With --side-targets, add orderbook-aware features and predict the next step's bid/ask delta.",
    )
    parser.add_argument(
        "--current-step-book-features",
        action="store_true",
        help="With --side-targets --book-features, predict the current step's bid/ask delta instead of the next step.",
    )
    parser.add_argument(
        "--mechanical-residual",
        action="store_true",
        help="With --side-targets, fit realized-minus-mechanical bid/ask residuals for the current step.",
    )
    parser.add_argument(
        "--side-mechanical-delta",
        action="store_true",
        help="Fit current best bid/ask delta from only the mechanical best-price delta.",
    )
    parser.add_argument(
        "--online-side-mechanical-delta",
        action="store_true",
        help="Online prequential version of --side-mechanical-delta with RLS decay and EWMA NLL.",
    )
    parser.add_argument(
        "--online-vwap-mechanical-delta",
        action="store_true",
        help="Online fixed-depth buy/sell VWAP delta from only the mechanical fixed-depth VWAP delta.",
    )
    parser.add_argument(
        "--online-decomposed-hurdle-vwap",
        action="store_true",
        help="Online decomposed mid/spread/shape VWAP model with quiet/active hurdle residual regimes.",
    )
    parser.add_argument(
        "--online-decomposed-two-regime-vwap",
        action="store_true",
        help="Online decomposed two-regime VWAP model: market impact is linear in mechanical_delta; refresh is adaptive Gaussian.",
    )
    parser.add_argument(
        "--online-decomposed-soft-mixture-vwap",
        action="store_true",
        help="Online decomposed soft-mixture VWAP model with learned sigmoid refresh probability.",
    )
    parser.add_argument(
        "--online-decomposed-em-mixture-vwap",
        action="store_true",
        help="Online decomposed EM mixture VWAP model with learned constant market probability and no spread component.",
    )
    parser.add_argument(
        "--online-decomposed-flow-gated-mixture-vwap",
        action="store_true",
        help="Online decomposed mixture VWAP model with pi_market sigmoid in total market flow and optional heavy-tailed refresh.",
    )
    parser.add_argument(
        "--continuous-sections",
        action="store_true",
        help="Evaluate selected online model separately on continuous timestamp sections.",
    )
    parser.add_argument(
        "--max-gap-factor",
        type=float,
        default=1.5,
        help="For --continuous-sections, split when timestamp gap exceeds this multiple of interval_ms.",
    )
    parser.add_argument(
        "--gate-learning-rate",
        type=float,
        default=0.02,
        help="Learning rate for the sigmoid gate in --online-decomposed-soft-mixture-vwap.",
    )
    parser.add_argument(
        "--soft-mixture-diagnostics",
        type=int,
        default=0,
        help="Print this many worst point errors for --online-decomposed-soft-mixture-vwap.",
    )
    parser.add_argument(
        "--soft-mixture-market-deviation-cap-mult",
        type=float,
        default=0.0,
        help=(
            "For soft-mixture point RMSE only, cap market side-delta deviation "
            "from mechanical side-delta to this multiple of previous spread. "
            "Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--soft-mixture-market-guard-mode",
        choices=("cap", "drop"),
        default="cap",
        help=(
            "When --soft-mixture-market-deviation-cap-mult is positive, either cap "
            "invalid market point deltas around mechanical or drop the market "
            "component from the point mean."
        ),
    )
    parser.add_argument(
        "--em-mixture-point-mode",
        choices=("mean", "map"),
        default="mean",
        help="Point prediction for --online-decomposed-em-mixture-vwap: mixture mean or dominant/MAP component.",
    )
    parser.add_argument(
        "--flow-gate-learning-rate",
        type=float,
        default=0.02,
        help="Learning rate for the flow-gated mixture sigmoid pi_market update.",
    )
    parser.add_argument(
        "--flow-gate-kind",
        choices=("sigmoid", "binned"),
        default="sigmoid",
        help=(
            "Gate form for --online-decomposed-flow-gated-mixture-vwap. "
            "sigmoid learns pi_market by SGD; binned learns EWMA pi_market per log1p(total_flow) bin."
        ),
    )
    parser.add_argument(
        "--flow-gate-bins",
        default="0,2,4,6,8",
        help="Comma-separated log1p(total_flow) bin edges for --flow-gate-kind binned.",
    )
    parser.add_argument(
        "--flow-mixture-point-mode",
        choices=("mean", "map"),
        default="map",
        help="Point prediction for --online-decomposed-flow-gated-mixture-vwap.",
    )
    parser.add_argument(
        "--refresh-distribution",
        choices=("gaussian", "student_t", "two_gaussian"),
        default="student_t",
        help="Refresh component likelihood for --online-decomposed-flow-gated-mixture-vwap.",
    )
    parser.add_argument(
        "--refresh-student-df",
        type=float,
        default=3.0,
        help="Degrees of freedom for Student-t refresh likelihood.",
    )
    parser.add_argument(
        "--refresh-jump-sigma-mult",
        type=float,
        default=8.0,
        help="Initial sigma_jump/sigma_quiet multiplier for two-Gaussian refresh.",
    )
    parser.add_argument(
        "--flow-gate-spread-features",
        choices=("none", "prev", "both"),
        default="none",
        help=(
            "Additional spread features for --online-decomposed-flow-gated-mixture-vwap: "
            "prev adds log1p(spread_t-1), both also adds abs(mechanical log-spread delta)."
        ),
    )
    parser.add_argument(
        "--online-cross-venue-wallex-from-nobitex",
        action="store_true",
        help="Predict Wallex current deltas online from aligned Nobitex mechanical/orderflow features.",
    )
    parser.add_argument(
        "--cross-venue-delta-lags",
        action="store_true",
        help="Regress local exchange price deltas on global exchange price deltas at t and past lags.",
    )
    parser.add_argument(
        "--cross-venue-gated-delta",
        action="store_true",
        help="Hard-gated same-step cross-venue model: quiet predicts zero, active regresses local delta on global delta_t.",
    )
    parser.add_argument(
        "--quote-refresh-diagnostics",
        action="store_true",
        help="Analyze intervals with mechanical_delta=0 and large price_delta, especially zero-flow quote refresh events.",
    )
    parser.add_argument(
        "--cross-venue-exclude-current-delta",
        action="store_true",
        help="With --cross-venue-delta-lags, use only source delta lags t-1...t-k and exclude contemporaneous delta_t.",
    )
    parser.add_argument(
        "--cross-venue-gate-percentiles",
        default="50,70,80,90",
        help="Comma-separated train percentiles of abs(global delta_t) used as active gates.",
    )
    parser.add_argument(
        "--quote-refresh-sigma-multiplier",
        type=float,
        default=2.0,
        help="Large-jump threshold multiplier for quiet sigma in --quote-refresh-diagnostics.",
    )
    parser.add_argument(
        "--hard-gated-quiet",
        action="store_true",
        help="With --online-decomposed-hurdle-vwap, quiet regime predicts zero VWAP delta and bypasses mechanical movement.",
    )
    parser.add_argument(
        "--inventory-cash-vwap-depth",
        action="store_true",
        help="With --online-vwap-mechanical-delta, use cash/mid depth for buy VWAP and inventory depth for sell VWAP.",
    )
    parser.add_argument(
        "--online-flow-spread-features",
        action="store_true",
        help="With online mechanical-delta models, add buy/sell volume and spread at t and t-1.",
    )
    parser.add_argument(
        "--online-spread-features",
        action="store_true",
        help="With online mechanical-delta models, add spread at t and t-1.",
    )
    parser.add_argument(
        "--online-current-spread-feature",
        action="store_true",
        help="With online mechanical-delta models, add only spread_t.",
    )
    parser.add_argument(
        "--online-current-flow-features",
        action="store_true",
        help="With online VWAP mechanical-delta models, add buy/sell volume at t and t-1.",
    )
    parser.add_argument(
        "--online-current-flow-spread-delta-lag",
        action="store_true",
        help="With online VWAP mechanical-delta models, add buy_volume_t, sell_volume_t, spread_t, and delta_price_t-1.",
    )
    parser.add_argument(
        "--online-current-flow-prev-book-features",
        action="store_true",
        help="With online VWAP mechanical-delta models, add buy/sell volume at t plus spread, queue imbalance, and microprice offset at t-1.",
    )
    parser.add_argument(
        "--online-signed-total-spread-features",
        action="store_true",
        help="With online VWAP mechanical-delta models, add signed_flow_t, total_flow_t, and spread_t-1.",
    )
    parser.add_argument(
        "--online-flow-features",
        action="store_true",
        help="Alias for --online-current-flow-features.",
    )
    parser.add_argument(
        "--online-mechanical-delta-lags",
        type=int,
        default=0,
        help="With online VWAP mechanical-delta models, include mechanical_delta lags through this many past steps.",
    )
    parser.add_argument(
        "--online-delta-price-lags",
        type=int,
        default=0,
        help="With online VWAP mechanical-delta models, include observed delta_price lags through this many past steps.",
    )
    parser.add_argument(
        "--online-decay",
        type=float,
        default=0.95,
        help="Forgetting factor for online RLS beta and EWMA variance updates.",
    )
    parser.add_argument(
        "--online-initial-sigma",
        type=float,
        default=50.0,
        help="Initial Gaussian sigma for online NLL scoring.",
    )
    parser.add_argument(
        "--vwap-residual",
        action="store_true",
        help="Fit current-step buy/sell VWAP residuals after mechanically consuming current flow from the previous book.",
    )
    parser.add_argument(
        "--vwap-current-step",
        action="store_true",
        help="Fit current-step buy/sell VWAP deltas using mechanical VWAP delta plus current/past flow features.",
    )
    parser.add_argument(
        "--vwap-book-current-step",
        action="store_true",
        help="Fit current-step VWAP deltas using previous-step book-aware flow features.",
    )
    parser.add_argument(
        "--vwap-complete-current-step",
        action="store_true",
        help="Fit current-step VWAP deltas from mechanical VWAP delta plus current/past flow and book-pressure features.",
    )
    parser.add_argument(
        "--no-depth-ratio-features",
        action="store_true",
        help="Drop flow/depth ratio features from the complete VWAP current-step model.",
    )
    parser.add_argument(
        "--vwap-depth",
        type=float,
        default=1000.0,
        help="Base quantity depth used for buy/sell VWAP experiments.",
    )
    parser.add_argument(
        "--hurdle-flow-threshold",
        type=float,
        default=0.0,
        help="Active regime if total current flow exceeds this threshold.",
    )
    parser.add_argument(
        "--hurdle-mechanical-threshold",
        type=float,
        default=1e-9,
        help="Active regime if abs(mechanical mid delta) exceeds this threshold.",
    )
    parser.add_argument(
        "--hurdle-thin-depth-threshold",
        type=float,
        default=-1.0,
        help="Active regime if previous top-5 bid/ask depth is below this value; negative disables.",
    )
    parser.add_argument(
        "--hurdle-gate-mode",
        choices=("threshold", "depth_percentile", "flow_to_depth"),
        default="threshold",
        help="Gate mode for decomposed hurdle VWAP experiments.",
    )
    parser.add_argument(
        "--hurdle-depth-percentile",
        type=float,
        default=20.0,
        help="For --hurdle-gate-mode depth_percentile, active if side depth is below this percentile.",
    )
    parser.add_argument(
        "--hurdle-flow-depth-ratio",
        type=float,
        default=0.25,
        help="For --hurdle-gate-mode flow_to_depth, active if side flow / previous side depth exceeds this threshold.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Chronological train fraction used for out-of-sample evaluation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_mm_env(override=False)
    database_url = os.environ.get(args.database_url_env)
    if not database_url:
        raise SystemExit(f"{args.database_url_env} is not set; add it to .env or export it.")

    import psycopg2

    windows = parse_windows(args)
    specs = (
        ("local", args.local_exchange),
        ("global", args.global_exchange),
    )
    topic_by_exchange = {
        exchange: f"market:events:{exchange}:{args.symbol}:{args.interval_ms}"
        for _, exchange in specs
    }

    with psycopg2.connect(database_url) as conn:
        if args.quote_refresh_diagnostics:
            for role, exchange in specs:
                topic = topic_by_exchange[exchange]
                latest_ts = fetch_latest_ts(conn, topic, exchange, args.symbol)
                if latest_ts is None:
                    print(f"\n[{role}] {exchange} {args.symbol}: no MarketEvent rows found for topic {topic}")
                    continue
                print(
                    f"\n[{role}] {exchange} {args.symbol} interval_ms={args.interval_ms} "
                    f"latest={latest_ts.isoformat()}"
                )
                for window in windows:
                    until = latest_ts - window.offset
                    since = None if window.lookback is None else until - window.lookback
                    rows = fetch_market_rows(
                        conn,
                        topic,
                        exchange,
                        args.symbol,
                        role=role,
                        since=since,
                        until=until,
                    )
                    print_quote_refresh_diagnostics(
                        window.label,
                        rows,
                        exchange=exchange,
                        min_rows=args.min_rows,
                        vwap_depth=args.vwap_depth,
                        sigma_multiplier=args.quote_refresh_sigma_multiplier,
                    )
            return

        if args.online_cross_venue_wallex_from_nobitex:
            local_topic = topic_by_exchange[args.local_exchange]
            global_topic = topic_by_exchange[args.global_exchange]
            latest_local = fetch_latest_ts(conn, local_topic, args.local_exchange, args.symbol)
            latest_global = fetch_latest_ts(conn, global_topic, args.global_exchange, args.symbol)
            if latest_local is None or latest_global is None:
                raise SystemExit("Need both local and global MarketEvent rows for cross-venue analysis.")
            latest_ts = min(latest_local, latest_global)
            print(
                f"\n[cross] target={args.local_exchange} source={args.global_exchange} "
                f"{args.symbol} interval_ms={args.interval_ms} latest_aligned_until={latest_ts.isoformat()}"
            )
            for window in windows:
                until = latest_ts - window.offset
                since = None if window.lookback is None else until - window.lookback
                target_rows = fetch_market_rows(
                    conn,
                    local_topic,
                    args.local_exchange,
                    args.symbol,
                    role="local",
                    since=since,
                    until=until,
                )
                source_rows = fetch_market_rows(
                    conn,
                    global_topic,
                    args.global_exchange,
                    args.symbol,
                    role="global",
                    since=since,
                    until=until,
                )
                print_online_cross_venue_window_results(
                    window.label,
                    target_rows,
                    source_rows,
                    target_exchange=args.local_exchange,
                    source_exchange=args.global_exchange,
                    min_rows=args.min_rows,
                    decay=args.online_decay,
                    initial_sigma=args.online_initial_sigma,
                    vwap_depth=args.vwap_depth,
                    mechanical_delta_lags=args.online_mechanical_delta_lags,
                )
            return

        if args.cross_venue_delta_lags or args.cross_venue_gated_delta:
            local_topic = topic_by_exchange[args.local_exchange]
            global_topic = topic_by_exchange[args.global_exchange]
            latest_local = fetch_latest_ts(conn, local_topic, args.local_exchange, args.symbol)
            latest_global = fetch_latest_ts(conn, global_topic, args.global_exchange, args.symbol)
            if latest_local is None or latest_global is None:
                raise SystemExit("Need both local and global MarketEvent rows for cross-venue delta-lag analysis.")
            latest_ts = min(latest_local, latest_global)
            mode = "cross-gated-delta" if args.cross_venue_gated_delta else "cross-delta"
            print(
                f"\n[{mode}] target={args.local_exchange} source={args.global_exchange} "
                f"{args.symbol} interval_ms={args.interval_ms} latest_aligned_until={latest_ts.isoformat()}"
            )
            gate_percentiles = parse_float_list(args.cross_venue_gate_percentiles)
            for window in windows:
                until = latest_ts - window.offset
                since = None if window.lookback is None else until - window.lookback
                target_rows = fetch_market_rows(
                    conn,
                    local_topic,
                    args.local_exchange,
                    args.symbol,
                    role="local",
                    since=since,
                    until=until,
                )
                source_rows = fetch_market_rows(
                    conn,
                    global_topic,
                    args.global_exchange,
                    args.symbol,
                    role="global",
                    since=since,
                    until=until,
                )
                if args.cross_venue_gated_delta:
                    print_cross_venue_gated_delta_results(
                        window.label,
                        target_rows,
                        source_rows,
                        target_exchange=args.local_exchange,
                        source_exchange=args.global_exchange,
                        gate_percentiles=gate_percentiles,
                        min_rows=args.min_rows,
                        train_ratio=args.train_ratio,
                        vwap_depth=args.vwap_depth,
                    )
                else:
                    print_cross_venue_delta_lag_results(
                        window.label,
                        target_rows,
                        source_rows,
                        target_exchange=args.local_exchange,
                        source_exchange=args.global_exchange,
                        lags=args.lags,
                        include_current=not args.cross_venue_exclude_current_delta,
                        min_rows=args.min_rows,
                        train_ratio=args.train_ratio,
                        vwap_depth=args.vwap_depth,
                    )
            return

        for role, exchange in specs:
            topic = topic_by_exchange[exchange]
            latest_ts = fetch_latest_ts(conn, topic, exchange, args.symbol)
            if latest_ts is None:
                print(f"\n[{role}] {exchange} {args.symbol}: no MarketEvent rows found for topic {topic}")
                continue

            print_header(role, exchange, args.symbol, args.interval_ms, topic, latest_ts)
            for window in windows:
                until = latest_ts - window.offset
                since = None if window.lookback is None else until - window.lookback
                rows = fetch_market_rows(
                    conn,
                    topic,
                    exchange,
                    args.symbol,
                    role=role,
                    since=since,
                    until=until,
                )
                if args.online_decomposed_flow_gated_mixture_vwap:
                    print_online_decomposed_flow_gated_mixture_vwap_results(
                        window.label,
                        rows,
                        min_rows=args.min_rows,
                        decay=args.online_decay,
                        initial_sigma=args.online_initial_sigma,
                        vwap_depth=args.vwap_depth,
                        interval_ms=args.interval_ms,
                        continuous_sections=args.continuous_sections,
                        max_gap_factor=args.max_gap_factor,
                        gate_learning_rate=args.flow_gate_learning_rate,
                        gate_kind=args.flow_gate_kind,
                        gate_bins=parse_float_list(args.flow_gate_bins),
                        point_mode=args.flow_mixture_point_mode,
                        refresh_distribution=args.refresh_distribution,
                        refresh_student_df=args.refresh_student_df,
                        refresh_jump_sigma_mult=args.refresh_jump_sigma_mult,
                        spread_features=args.flow_gate_spread_features,
                    )
                elif args.online_decomposed_em_mixture_vwap:
                    print_online_decomposed_em_mixture_vwap_results(
                        window.label,
                        rows,
                        min_rows=args.min_rows,
                        decay=args.online_decay,
                        initial_sigma=args.online_initial_sigma,
                        vwap_depth=args.vwap_depth,
                        interval_ms=args.interval_ms,
                        continuous_sections=args.continuous_sections,
                        max_gap_factor=args.max_gap_factor,
                        point_mode=args.em_mixture_point_mode,
                    )
                elif args.online_decomposed_soft_mixture_vwap:
                    print_online_decomposed_soft_mixture_vwap_results(
                        window.label,
                        rows,
                        min_rows=args.min_rows,
                        decay=args.online_decay,
                        initial_sigma=args.online_initial_sigma,
                        vwap_depth=args.vwap_depth,
                        interval_ms=args.interval_ms,
                        continuous_sections=args.continuous_sections,
                        max_gap_factor=args.max_gap_factor,
                        gate_learning_rate=args.gate_learning_rate,
                        diagnostic_count=args.soft_mixture_diagnostics,
                        market_deviation_cap_mult=args.soft_mixture_market_deviation_cap_mult,
                        market_guard_mode=args.soft_mixture_market_guard_mode,
                    )
                elif args.online_decomposed_two_regime_vwap:
                    print_online_decomposed_two_regime_vwap_results(
                        window.label,
                        rows,
                        min_rows=args.min_rows,
                        decay=args.online_decay,
                        initial_sigma=args.online_initial_sigma,
                        vwap_depth=args.vwap_depth,
                    )
                elif args.online_decomposed_hurdle_vwap:
                    print_online_decomposed_hurdle_vwap_results(
                        window.label,
                        rows,
                        min_rows=args.min_rows,
                        decay=args.online_decay,
                        initial_sigma=args.online_initial_sigma,
                        vwap_depth=args.vwap_depth,
                        flow_threshold=args.hurdle_flow_threshold,
                        mechanical_threshold=args.hurdle_mechanical_threshold,
                        thin_depth_threshold=args.hurdle_thin_depth_threshold,
                        hard_gated_quiet=args.hard_gated_quiet,
                        gate_mode=args.hurdle_gate_mode,
                        depth_percentile=args.hurdle_depth_percentile,
                        flow_depth_ratio=args.hurdle_flow_depth_ratio,
                    )
                elif args.vwap_complete_current_step:
                    print_vwap_complete_current_step_window_results(
                        window.label,
                        rows,
                        lags=args.lags,
                        min_rows=args.min_rows,
                        train_ratio=args.train_ratio,
                        vwap_depth=args.vwap_depth,
                        include_depth_ratios=not args.no_depth_ratio_features,
                    )
                elif args.vwap_book_current_step:
                    print_vwap_book_current_step_window_results(
                        window.label,
                        rows,
                        min_rows=args.min_rows,
                        train_ratio=args.train_ratio,
                        vwap_depth=args.vwap_depth,
                    )
                elif args.vwap_current_step:
                    print_vwap_current_step_window_results(
                        window.label,
                        rows,
                        lags=args.lags,
                        min_rows=args.min_rows,
                        train_ratio=args.train_ratio,
                        vwap_depth=args.vwap_depth,
                    )
                elif args.vwap_residual:
                    print_vwap_residual_window_results(
                        window.label,
                        rows,
                        lags=args.lags,
                        min_rows=args.min_rows,
                        train_ratio=args.train_ratio,
                        vwap_depth=args.vwap_depth,
                    )
                elif args.online_vwap_mechanical_delta:
                    print_online_vwap_mechanical_delta_window_results(
                        window.label,
                        rows,
                        min_rows=args.min_rows,
                        decay=args.online_decay,
                        initial_sigma=args.online_initial_sigma,
                        vwap_depth=args.vwap_depth,
                        inventory_cash_depth=args.inventory_cash_vwap_depth,
                        flow_spread_features=args.online_flow_spread_features,
                        spread_features=args.online_spread_features,
                        current_spread_feature=args.online_current_spread_feature,
                        current_flow_spread_delta_lag=args.online_current_flow_spread_delta_lag,
                        current_flow_prev_book_features=args.online_current_flow_prev_book_features,
                        signed_total_spread_features=args.online_signed_total_spread_features,
                        current_flow_features=args.online_current_flow_features or args.online_flow_features,
                        mechanical_delta_lags=args.online_mechanical_delta_lags,
                        delta_price_lags=args.online_delta_price_lags,
                    )
                elif args.online_side_mechanical_delta:
                    print_online_side_mechanical_delta_window_results(
                        window.label,
                        rows,
                        min_rows=args.min_rows,
                        decay=args.online_decay,
                        initial_sigma=args.online_initial_sigma,
                    )
                elif args.side_mechanical_delta:
                    print_side_mechanical_delta_window_results(
                        window.label,
                        rows,
                        min_rows=args.min_rows,
                        train_ratio=args.train_ratio,
                    )
                elif args.side_targets:
                    print_side_window_results(
                        window.label,
                        rows,
                        lags=args.lags,
                        min_rows=args.min_rows,
                        train_ratio=args.train_ratio,
                        target=args.target,
                        book_features=args.book_features,
                        current_step_book_features=args.current_step_book_features,
                        mechanical_residual=args.mechanical_residual,
                    )
                else:
                    result = fit_orderflow_model(
                        rows,
                        target=args.target,
                        feature_kind=args.features,
                        lags=args.lags,
                        min_rows=args.min_rows,
                    )
                    split_result = train_test_orderflow_model(
                        rows,
                        target=args.target,
                        feature_kind=args.features,
                        lags=args.lags,
                        min_rows=args.min_rows,
                        train_ratio=args.train_ratio,
                    )
                    print_window_result(window.label, rows, result, split_result)


@dataclass(frozen=True, slots=True)
class WindowSpec:
    label: str
    lookback: timedelta | None
    offset: timedelta = timedelta(0)


def parse_windows(args: argparse.Namespace) -> list[WindowSpec]:
    if args.rolling_window:
        lookback = parse_duration(args.rolling_window)
        if lookback is None:
            raise SystemExit("--rolling-window must be a finite duration like 1h, not all.")
        count = max(1, int(args.rolling_count))
        return [
            WindowSpec(
                label=f"{format_duration(lookback)} ending {format_duration(lookback * offset)} ago",
                lookback=lookback,
                offset=lookback * offset,
            )
            for offset in range(count)
        ]
    return parse_lookbacks(args.lookbacks)


def parse_lookbacks(raw: str) -> list[WindowSpec]:
    out: list[WindowSpec] = []
    for item in (part.strip().lower() for part in raw.split(",")):
        if not item:
            continue
        out.append(WindowSpec(label=item, lookback=parse_duration(item)))
    if not out:
        raise SystemExit("No valid lookback windows supplied.")
    return out


def parse_float_list(raw: str) -> list[float]:
    values: list[float] = []
    for item in (part.strip() for part in raw.split(",")):
        if not item:
            continue
        try:
            values.append(float(item))
        except ValueError as exc:
            raise SystemExit(f"Invalid float value {item!r} in comma-separated list {raw!r}.") from exc
    if not values:
        raise SystemExit("Expected at least one float value.")
    return values


def flow_bin_index(value: float, edges: Sequence[float]) -> int:
    return int(np.searchsorted(np.asarray(edges, dtype=np.float64), value, side="right"))


def parse_duration(item: str) -> timedelta | None:
    item = item.strip().lower()
    if item == "all":
        return None
    unit = item[-1]
    try:
        value = float(item[:-1])
    except ValueError as exc:
        raise SystemExit(f"Invalid duration {item!r}; expected examples like 1h, 24h, 7d, all.") from exc
    if unit == "s":
        return timedelta(seconds=value)
    if unit == "m":
        return timedelta(minutes=value)
    if unit == "h":
        return timedelta(hours=value)
    if unit == "d":
        return timedelta(days=value)
    raise SystemExit(f"Invalid duration unit in {item!r}; use s, m, h, d, or all.")


def format_duration(delta: timedelta) -> str:
    seconds = int(delta.total_seconds())
    if seconds == 0:
        return "0s"
    for suffix, scale in (("d", 86400), ("h", 3600), ("m", 60)):
        if seconds % scale == 0:
            return f"{seconds // scale}{suffix}"
    return f"{seconds}s"


def split_continuous_sections(
    rows: Sequence[MarketRow],
    *,
    interval_ms: int,
    max_gap_factor: float,
    min_rows: int,
) -> list[list[MarketRow]]:
    if not rows:
        return []
    max_gap = timedelta(milliseconds=max(1, int(interval_ms)) * max(1.0, float(max_gap_factor)))
    sections: list[list[MarketRow]] = []
    current: list[MarketRow] = [rows[0]]
    for prev, row in zip(rows, rows[1:], strict=False):
        if row.ts - prev.ts > max_gap:
            if len(current) >= min_rows:
                sections.append(current)
            current = [row]
        else:
            current.append(row)
    if len(current) >= min_rows:
        sections.append(current)
    return sections


def fetch_latest_ts(conn, topic: str, exchange: str, symbol: str) -> datetime | None:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT MAX(ts)
            FROM mm_events
            WHERE event_type = 'MarketEvent'
              AND topic = %s
              AND exchange = %s
              AND symbol = %s
            """,
            (topic, exchange, symbol),
        )
        row = cur.fetchone()
    return row[0] if row and row[0] is not None else None


def fetch_market_rows(
    conn,
    topic: str,
    exchange: str,
    symbol: str,
    *,
    role: str,
    since: datetime | None,
    until: datetime,
) -> list[MarketRow]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                me.ts,
                COALESCE((me.payload->>'buy_volume')::double precision, 0.0) AS buy_volume,
                COALESCE((me.payload->>'sell_volume')::double precision, 0.0) AS sell_volume,
                COALESCE(
                    (me.payload->'orderbook'->'bids'->0->>'price')::double precision,
                    (me.payload->>'microprice')::double precision
                ) AS best_bid,
                COALESCE(
                    (me.payload->'orderbook'->'asks'->0->>'price')::double precision,
                    (me.payload->>'microprice')::double precision
                ) AS best_ask,
                COALESCE((me.payload->'orderbook'->'bids'->0->>'qty')::double precision, 0.0) AS best_bid_qty,
                COALESCE((me.payload->'orderbook'->'asks'->0->>'qty')::double precision, 0.0) AS best_ask_qty,
                COALESCE((
                    SELECT SUM((level->>'qty')::double precision)
                    FROM jsonb_array_elements(me.payload->'orderbook'->'bids') WITH ORDINALITY AS x(level, ord)
                    WHERE ord <= 5
                ), 0.0) AS bid_depth_5,
                COALESCE((
                    SELECT SUM((level->>'qty')::double precision)
                    FROM jsonb_array_elements(me.payload->'orderbook'->'asks') WITH ORDINALITY AS x(level, ord)
                    WHERE ord <= 5
                ), 0.0) AS ask_depth_5,
                COALESCE((me.payload->>'microprice')::double precision, 0.0) AS microprice,
                me.payload->'orderbook' AS orderbook,
                (diag.payload #>> '{state,local_inventory}')::double precision AS local_inventory,
                (diag.payload #>> '{state,local_cash}')::double precision AS local_cash,
                (diag.payload #>> '{state,global_inventory}')::double precision AS global_inventory,
                (diag.payload #>> '{state,global_cash}')::double precision AS global_cash
            FROM mm_events me
            LEFT JOIN mm_events diag
              ON diag.event_type = 'ControllerDiagnostics'
             AND diag.symbol = me.symbol
             AND diag.ts = me.ts
            WHERE me.event_type = 'MarketEvent'
              AND me.topic = %s
              AND me.exchange = %s
              AND me.symbol = %s
              AND me.ts <= %s
              AND (%s::timestamptz IS NULL OR me.ts >= %s)
            ORDER BY me.ts ASC
            """,
            (topic, exchange, symbol, until, since, since),
        )
        rows = cur.fetchall()
    use_global = role == "global"
    return [
        MarketRow(
            ts=row[0],
            buy_volume=safe_float(row[1]),
            sell_volume=safe_float(row[2]),
            best_bid=safe_float(row[3]),
            best_ask=safe_float(row[4]),
            best_bid_qty=safe_float(row[5]),
            best_ask_qty=safe_float(row[6]),
            bid_depth_5=safe_float(row[7]),
            ask_depth_5=safe_float(row[8]),
            microprice=safe_float(row[9]),
            bids=parse_levels(row[10], "bids"),
            asks=parse_levels(row[10], "asks"),
            inventory=safe_float(row[13] if use_global else row[11]),
            cash=safe_float(row[14] if use_global else row[12]),
        )
        for row in rows
        if row[3] is not None and row[4] is not None and safe_float(row[3]) > 0 and safe_float(row[4]) > 0
    ]


def fit_orderflow_model(
    rows: Sequence[MarketRow],
    *,
    target: str,
    feature_kind: str,
    lags: int,
    min_rows: int,
) -> FitResult | None:
    design = build_design_matrix(rows, target=target, feature_kind=feature_kind, lags=lags, min_rows=min_rows)
    if design is None:
        return None
    y, x_without_intercept, names, y_name = design
    x = np.column_stack([np.ones(y.size), x_without_intercept])
    beta = np.linalg.pinv(x) @ y
    residuals = y - x @ beta
    sse = float(residuals @ residuals)
    sigma2_mle = sse / y.size
    rmse = math.sqrt(sigma2_mle)
    centered = y - float(np.mean(y))
    sst = float(centered @ centered)
    r2 = 1.0 - sse / sst if sst > 0 else float("nan")

    dof = max(1, y.size - x.shape[1])
    sigma2_unbiased = sse / dof
    xtx_inv = np.linalg.pinv(x.T @ x)
    stderr = np.sqrt(np.maximum(0.0, np.diag(xtx_inv) * sigma2_unbiased))
    t_stats = np.divide(beta, stderr, out=np.full_like(beta, np.nan), where=stderr > 0)
    corr = [pearson(x_without_intercept[:, i], y) for i in range(x_without_intercept.shape[1])]

    return FitResult(
        n=int(y.size),
        y_name=y_name,
        feature_names=names,
        beta=beta,
        stderr=stderr,
        t_stats=t_stats,
        sigma2_mle=sigma2_mle,
        rmse=rmse,
        r2=r2,
        corr=corr,
    )


def fit_side_model(
    rows: Sequence[MarketRow],
    *,
    side: str,
    target: str,
    lags: int,
    min_rows: int,
    book_features: bool = False,
    current_step_book_features: bool = False,
    mechanical_residual: bool = False,
) -> FitResult | None:
    if mechanical_residual:
        design = build_mechanical_residual_design_matrix(
            rows,
            side=side,
            lags=lags,
            min_rows=min_rows,
            book_features=book_features,
        )
    else:
        design = build_side_design_matrix(
            rows,
            side=side,
            target=target,
            lags=lags,
            min_rows=min_rows,
            book_features=book_features,
            current_step_book_features=current_step_book_features,
        )
    if design is None:
        return None
    y, x_without_intercept, names, y_name = design
    x = np.column_stack([np.ones(y.size), x_without_intercept])
    beta = np.linalg.pinv(x) @ y
    residuals = y - x @ beta
    sse = float(residuals @ residuals)
    sigma2_mle = sse / y.size
    rmse = math.sqrt(sigma2_mle)
    centered = y - float(np.mean(y))
    sst = float(centered @ centered)
    r2 = 1.0 - sse / sst if sst > 0 else float("nan")
    dof = max(1, y.size - x.shape[1])
    sigma2_unbiased = sse / dof
    xtx_inv = np.linalg.pinv(x.T @ x)
    stderr = np.sqrt(np.maximum(0.0, np.diag(xtx_inv) * sigma2_unbiased))
    t_stats = np.divide(beta, stderr, out=np.full_like(beta, np.nan), where=stderr > 0)
    corr = [pearson(x_without_intercept[:, i], y) for i in range(x_without_intercept.shape[1])]
    return FitResult(
        n=int(y.size),
        y_name=y_name,
        feature_names=names,
        beta=beta,
        stderr=stderr,
        t_stats=t_stats,
        sigma2_mle=sigma2_mle,
        rmse=rmse,
        r2=r2,
        corr=corr,
    )


def train_test_orderflow_model(
    rows: Sequence[MarketRow],
    *,
    target: str,
    feature_kind: str,
    lags: int,
    min_rows: int,
    train_ratio: float,
) -> TrainTestResult | None:
    design = build_design_matrix(rows, target=target, feature_kind=feature_kind, lags=lags, min_rows=min_rows)
    if design is None:
        return None
    y, x_without_intercept, _names, _y_name = design
    x = np.column_stack([np.ones(y.size), x_without_intercept])
    ratio = min(0.95, max(0.05, float(train_ratio)))
    split_idx = int(math.floor(y.size * ratio))
    if split_idx < min_rows or y.size - split_idx < max(5, min_rows // 4):
        return None

    x_train = x[:split_idx, :]
    y_train = y[:split_idx]
    x_test = x[split_idx:, :]
    y_test = y[split_idx:]
    beta = np.linalg.pinv(x_train) @ y_train
    baseline = float(np.mean(y_train))

    return TrainTestResult(
        train=evaluate_predictions(y_train, x_train @ beta, baseline),
        test=evaluate_predictions(y_test, x_test @ beta, baseline),
        split_ratio=ratio,
    )


def train_test_side_model(
    rows: Sequence[MarketRow],
    *,
    side: str,
    target: str,
    lags: int,
    min_rows: int,
    train_ratio: float,
    book_features: bool = False,
    current_step_book_features: bool = False,
    mechanical_residual: bool = False,
) -> TrainTestResult | None:
    if mechanical_residual:
        design = build_mechanical_residual_design_matrix(
            rows,
            side=side,
            lags=lags,
            min_rows=min_rows,
            book_features=book_features,
        )
    else:
        design = build_side_design_matrix(
            rows,
            side=side,
            target=target,
            lags=lags,
            min_rows=min_rows,
            book_features=book_features,
            current_step_book_features=current_step_book_features,
        )
    if design is None:
        return None
    y, x_without_intercept, _names, _y_name = design
    x = np.column_stack([np.ones(y.size), x_without_intercept])
    ratio = min(0.95, max(0.05, float(train_ratio)))
    split_idx = int(math.floor(y.size * ratio))
    if split_idx < min_rows or y.size - split_idx < max(5, min_rows // 4):
        return None

    x_train = x[:split_idx, :]
    y_train = y[:split_idx]
    x_test = x[split_idx:, :]
    y_test = y[split_idx:]
    beta = np.linalg.pinv(x_train) @ y_train
    baseline = float(np.mean(y_train))
    return TrainTestResult(
        train=evaluate_predictions(y_train, x_train @ beta, baseline),
        test=evaluate_predictions(y_test, x_test @ beta, baseline),
        split_ratio=ratio,
    )


def fit_side_mechanical_delta_model(
    rows: Sequence[MarketRow],
    *,
    side: str,
    min_rows: int,
) -> FitResult | None:
    design = build_side_mechanical_delta_design(rows, side=side, min_rows=min_rows)
    if design is None:
        return None
    y, x_without_intercept, names, y_name, _mechanical = design
    x = np.column_stack([np.ones(y.size), x_without_intercept])
    beta = np.linalg.pinv(x) @ y
    residuals = y - x @ beta
    sse = float(residuals @ residuals)
    sigma2_mle = sse / y.size
    rmse = math.sqrt(sigma2_mle)
    centered = y - float(np.mean(y))
    sst = float(centered @ centered)
    r2 = 1.0 - sse / sst if sst > 0 else float("nan")
    dof = max(1, y.size - x.shape[1])
    sigma2_unbiased = sse / dof
    xtx_inv = np.linalg.pinv(x.T @ x)
    stderr = np.sqrt(np.maximum(0.0, np.diag(xtx_inv) * sigma2_unbiased))
    t_stats = np.divide(beta, stderr, out=np.full_like(beta, np.nan), where=stderr > 0)
    corr = [pearson(x_without_intercept[:, i], y) for i in range(x_without_intercept.shape[1])]
    return FitResult(
        n=int(y.size),
        y_name=y_name,
        feature_names=names,
        beta=beta,
        stderr=stderr,
        t_stats=t_stats,
        sigma2_mle=sigma2_mle,
        rmse=rmse,
        r2=r2,
        corr=corr,
    )


def train_test_side_mechanical_delta_model(
    rows: Sequence[MarketRow],
    *,
    side: str,
    min_rows: int,
    train_ratio: float,
) -> tuple[TrainTestResult, EvaluationResult] | None:
    design = build_side_mechanical_delta_design(rows, side=side, min_rows=min_rows)
    if design is None:
        return None
    y, x_without_intercept, _names, _y_name, mechanical_delta = design
    x = np.column_stack([np.ones(y.size), x_without_intercept])
    ratio = min(0.95, max(0.05, float(train_ratio)))
    split_idx = int(math.floor(y.size * ratio))
    if split_idx < min_rows or y.size - split_idx < max(5, min_rows // 4):
        return None

    x_train = x[:split_idx, :]
    y_train = y[:split_idx]
    x_test = x[split_idx:, :]
    y_test = y[split_idx:]
    beta = np.linalg.pinv(x_train) @ y_train
    baseline = float(np.mean(y_train))
    model_eval = TrainTestResult(
        train=evaluate_predictions(y_train, x_train @ beta, baseline),
        test=evaluate_predictions(y_test, x_test @ beta, baseline),
        split_ratio=ratio,
    )
    mechanical_eval = evaluate_predictions(y_test, mechanical_delta[split_idx:], baseline)
    return model_eval, mechanical_eval


def online_side_mechanical_delta_model(
    rows: Sequence[MarketRow],
    *,
    side: str,
    min_rows: int,
    decay: float,
    initial_sigma: float,
) -> OnlineEvaluation | None:
    design = build_side_mechanical_delta_design(rows, side=side, min_rows=min_rows)
    if design is None:
        return None
    return online_mechanical_delta_design(design, min_rows=min_rows, decay=decay, initial_sigma=initial_sigma)


def online_vwap_mechanical_delta_model(
    rows: Sequence[MarketRow],
    *,
    side: str,
    min_rows: int,
    decay: float,
    initial_sigma: float,
    vwap_depth: float,
    inventory_cash_depth: bool,
    flow_spread_features: bool,
    spread_features: bool,
    current_spread_feature: bool,
    current_flow_spread_delta_lag: bool,
    current_flow_prev_book_features: bool,
    signed_total_spread_features: bool,
    current_flow_features: bool,
    mechanical_delta_lags: int,
    delta_price_lags: int,
) -> OnlineEvaluation | None:
    design = build_fixed_depth_vwap_mechanical_delta_design(
        rows,
        side=side,
        min_rows=min_rows,
        vwap_depth=vwap_depth,
        inventory_cash_depth=inventory_cash_depth,
        flow_spread_features=flow_spread_features,
        spread_features=spread_features,
        current_spread_feature=current_spread_feature,
        current_flow_spread_delta_lag=current_flow_spread_delta_lag,
        current_flow_prev_book_features=current_flow_prev_book_features,
        signed_total_spread_features=signed_total_spread_features,
        current_flow_features=current_flow_features,
        mechanical_delta_lags=mechanical_delta_lags,
        delta_price_lags=delta_price_lags,
    )
    if design is None:
        return None
    return online_mechanical_delta_design(design, min_rows=min_rows, decay=decay, initial_sigma=initial_sigma)


def online_cross_venue_model(
    target_rows: Sequence[MarketRow],
    source_rows: Sequence[MarketRow],
    *,
    target_kind: str,
    min_rows: int,
    decay: float,
    initial_sigma: float,
    vwap_depth: float,
    mechanical_delta_lags: int,
) -> tuple[OnlineEvaluation, list[str]] | None:
    design = build_cross_venue_design(
        target_rows,
        source_rows,
        target_kind=target_kind,
        min_rows=min_rows,
        vwap_depth=vwap_depth,
        mechanical_delta_lags=mechanical_delta_lags,
    )
    if design is None:
        return None
    result = online_mechanical_delta_design(
        design,
        min_rows=min_rows,
        decay=decay,
        initial_sigma=initial_sigma,
    )
    if result is None:
        return None
    _y, _x, names, _y_name, _mechanical = design
    return result, names


def online_mechanical_delta_design(
    design: tuple[np.ndarray, np.ndarray, list[str], str, np.ndarray],
    *,
    min_rows: int,
    decay: float,
    initial_sigma: float,
) -> OnlineEvaluation | None:
    y, x_without_intercept, _names, _y_name, mechanical_delta = design
    x_all = np.column_stack([np.ones(y.size), x_without_intercept])
    warmup = min(max(1, int(min_rows)), max(1, y.size - 1))
    lam = min(0.9999, max(0.50, float(decay)))
    one_minus_lam = 1.0 - lam
    beta = np.zeros(x_all.shape[1], dtype=np.float64)
    p = np.eye(x_all.shape[1], dtype=np.float64) * 1_000_000.0
    sigma2 = max(float(initial_sigma) ** 2, 1e-6)
    mean_mu = 0.0
    mean_sigma2 = sigma2
    zero_sigma2 = sigma2

    sq_errors: list[float] = []
    mean_sq_errors: list[float] = []
    zero_sq_errors: list[float] = []
    mechanical_sq_errors: list[float] = []
    nll_values: list[float] = []
    mean_nll_values: list[float] = []
    zero_nll_values: list[float] = []

    for idx in range(y.size):
        x = x_all[idx, :]
        target = float(y[idx])
        mu = float(x @ beta)
        model_error = target - mu
        mean_error = target - mean_mu
        zero_error = target
        mechanical_error = target - float(mechanical_delta[idx])

        if idx >= warmup:
            model_var = max(sigma2, 1e-9)
            mean_var = max(mean_sigma2, 1e-9)
            zero_var = max(zero_sigma2, 1e-9)
            sq_errors.append(model_error * model_error)
            mean_sq_errors.append(mean_error * mean_error)
            zero_sq_errors.append(zero_error * zero_error)
            mechanical_sq_errors.append(mechanical_error * mechanical_error)
            nll_values.append(gaussian_nll(model_error, model_var))
            mean_nll_values.append(gaussian_nll(mean_error, mean_var))
            zero_nll_values.append(gaussian_nll(zero_error, zero_var))

        px = p @ x
        denom = lam + float(x @ px)
        gain = px / denom
        beta = beta + gain * model_error
        p = (p - np.outer(gain, x) @ p) / lam
        p = 0.5 * (p + p.T)

        sigma2 = lam * sigma2 + one_minus_lam * (model_error * model_error)
        mean_mu = lam * mean_mu + one_minus_lam * target
        mean_sigma2 = lam * mean_sigma2 + one_minus_lam * (mean_error * mean_error)
        zero_sigma2 = lam * zero_sigma2 + one_minus_lam * (zero_error * zero_error)

    if not sq_errors:
        return None
    return OnlineEvaluation(
        n_scored=len(sq_errors),
        rmse=math.sqrt(float(np.mean(sq_errors))),
        nll=float(np.mean(nll_values)),
        mean_rmse=math.sqrt(float(np.mean(mean_sq_errors))),
        mean_nll=float(np.mean(mean_nll_values)),
        zero_rmse=math.sqrt(float(np.mean(zero_sq_errors))),
        zero_nll=float(np.mean(zero_nll_values)),
        mechanical_rmse=math.sqrt(float(np.mean(mechanical_sq_errors))),
        final_beta=beta,
        final_mean=mean_mu,
        final_sigma=math.sqrt(max(sigma2, 0.0)),
        final_mean_sigma=math.sqrt(max(mean_sigma2, 0.0)),
        final_zero_sigma=math.sqrt(max(zero_sigma2, 0.0)),
    )


def gaussian_nll(error: float, variance: float) -> float:
    return 0.5 * (math.log(2.0 * math.pi * variance) + (error * error) / variance)


def online_decomposed_hurdle_vwap_model(
    rows: Sequence[MarketRow],
    *,
    min_rows: int,
    decay: float,
    initial_sigma: float,
    vwap_depth: float,
    flow_threshold: float,
    mechanical_threshold: float,
    thin_depth_threshold: float,
    hard_gated_quiet: bool,
    gate_mode: str,
    depth_percentile: float,
    flow_depth_ratio: float,
) -> DecomposedHurdleEvaluation | None:
    if len(rows) < max(min_rows, 3):
        return None
    depth = max(0.0, float(vwap_depth))
    if depth <= 1e-12:
        return None

    lam = min(0.9999, max(0.50, float(decay)))
    alpha = 1.0 - lam
    eps = 1e-9
    init_var = max(float(initial_sigma) ** 2, eps)
    bid_depth_threshold = float("nan")
    ask_depth_threshold = float("nan")
    if gate_mode == "depth_percentile":
        pct = min(100.0, max(0.0, float(depth_percentile)))
        bid_depths = np.array([row.bid_depth_5 for row in rows if math.isfinite(row.bid_depth_5)], dtype=np.float64)
        ask_depths = np.array([row.ask_depth_5 for row in rows if math.isfinite(row.ask_depth_5)], dtype=np.float64)
        if bid_depths.size == 0 or ask_depths.size == 0:
            return None
        bid_depth_threshold = float(np.percentile(bid_depths, pct))
        ask_depth_threshold = float(np.percentile(ask_depths, pct))
    component_names = ("mid", "log_spread", "log_buy_shape", "log_sell_shape")
    means = {
        False: {name: 0.0 for name in component_names},
        True: {name: 0.0 for name in component_names},
    }
    buy_var = {False: init_var, True: init_var}
    sell_var = {False: init_var, True: init_var}
    zero_buy_var = init_var
    zero_sell_var = init_var

    seen = 0
    active_seen = 0
    buy_sq: list[float] = []
    sell_sq: list[float] = []
    buy_zero_sq: list[float] = []
    sell_zero_sq: list[float] = []
    buy_mech_sq: list[float] = []
    sell_mech_sq: list[float] = []
    buy_nll: list[float] = []
    sell_nll: list[float] = []
    buy_zero_nll: list[float] = []
    sell_zero_nll: list[float] = []

    for idx in range(1, len(rows)):
        prev = rows[idx - 1]
        curr = rows[idx]

        prev_buy = vwap_to_depth(prev.asks, depth, ask_side=True)
        prev_sell = vwap_to_depth(prev.bids, depth, ask_side=False)
        curr_buy = vwap_to_depth(curr.asks, depth, ask_side=True)
        curr_sell = vwap_to_depth(curr.bids, depth, ask_side=False)
        ask_after = consume_levels(prev.asks, curr.buy_volume, ask_side=True)
        bid_after = consume_levels(prev.bids, curr.sell_volume, ask_side=False)
        mech_buy = vwap_to_depth(ask_after, depth, ask_side=True)
        mech_sell = vwap_to_depth(bid_after, depth, ask_side=False)
        mech_bid = consume_best_price(prev.bids, curr.sell_volume, side="bid")
        mech_ask = consume_best_price(prev.asks, curr.buy_volume, side="ask")

        values = [prev_buy, prev_sell, curr_buy, curr_sell, mech_buy, mech_sell, mech_bid, mech_ask]
        if not all(math.isfinite(value) for value in values):
            continue

        obs_mid = curr.mid
        mech_mid = 0.5 * (mech_bid + mech_ask)
        obs_spread = curr.spread
        mech_spread = mech_ask - mech_bid
        obs_buy_shape = curr_buy - obs_mid
        obs_sell_shape = obs_mid - curr_sell
        mech_buy_shape = mech_buy - mech_mid
        mech_sell_shape = mech_mid - mech_sell
        if min(obs_spread, mech_spread, obs_buy_shape, obs_sell_shape, mech_buy_shape, mech_sell_shape) <= eps:
            continue

        total_flow = curr.buy_volume + curr.sell_volume
        thin_book = thin_depth_threshold >= 0.0 and min(prev.bid_depth_5, prev.ask_depth_5) <= thin_depth_threshold
        if gate_mode == "threshold":
            active_buy = total_flow > flow_threshold or abs(mech_mid - prev.mid) > mechanical_threshold or thin_book
            active_sell = active_buy
        elif gate_mode == "depth_percentile":
            active_buy = prev.ask_depth_5 <= ask_depth_threshold
            active_sell = prev.bid_depth_5 <= bid_depth_threshold
        elif gate_mode == "flow_to_depth":
            active_buy = curr.buy_volume / max(prev.ask_depth_5, eps) > flow_depth_ratio
            active_sell = curr.sell_volume / max(prev.bid_depth_5, eps) > flow_depth_ratio
        else:
            raise ValueError(f"unknown hurdle gate mode {gate_mode}")
        active = active_buy or active_sell
        active_seen += int(active)

        prev_mid = prev.mid
        prev_buy_shape = prev_buy - prev_mid
        prev_sell_shape = prev_mid - prev_sell
        if min(prev_buy_shape, prev_sell_shape) <= eps:
            continue

        if hard_gated_quiet and not active_buy and not active_sell:
            pred_mid = prev_mid
            pred_buy_shape = max(prev_buy_shape, eps)
            pred_sell_shape = max(prev_sell_shape, eps)
        else:
            pred_mid = mech_mid + means[active]["mid"]
            pred_buy_shape = math.exp(math.log(max(mech_buy_shape, eps)) + means[active]["log_buy_shape"])
            pred_sell_shape = math.exp(math.log(max(mech_sell_shape, eps)) + means[active]["log_sell_shape"])
        pred_buy = pred_mid + pred_buy_shape
        pred_sell = pred_mid - pred_sell_shape
        if hard_gated_quiet and not active_buy:
            pred_buy = prev_buy
        if hard_gated_quiet and not active_sell:
            pred_sell = prev_sell

        obs_buy_delta = curr_buy - prev_buy
        obs_sell_delta = curr_sell - prev_sell
        pred_buy_delta = pred_buy - prev_buy
        pred_sell_delta = pred_sell - prev_sell
        mech_buy_delta = mech_buy - prev_buy
        mech_sell_delta = mech_sell - prev_sell

        buy_err = obs_buy_delta - pred_buy_delta
        sell_err = obs_sell_delta - pred_sell_delta
        if seen >= min_rows:
            buy_sq.append(buy_err * buy_err)
            sell_sq.append(sell_err * sell_err)
            buy_zero_sq.append(obs_buy_delta * obs_buy_delta)
            sell_zero_sq.append(obs_sell_delta * obs_sell_delta)
            buy_mech_sq.append((obs_buy_delta - mech_buy_delta) ** 2)
            sell_mech_sq.append((obs_sell_delta - mech_sell_delta) ** 2)
            buy_nll.append(gaussian_nll(buy_err, max(buy_var[active_buy], eps)))
            sell_nll.append(gaussian_nll(sell_err, max(sell_var[active_sell], eps)))
            buy_zero_nll.append(gaussian_nll(obs_buy_delta, max(zero_buy_var, eps)))
            sell_zero_nll.append(gaussian_nll(obs_sell_delta, max(zero_sell_var, eps)))

        if not (hard_gated_quiet and not active):
            residuals = {
                "mid": obs_mid - mech_mid,
                "log_spread": math.log(max(obs_spread, eps)) - math.log(max(mech_spread, eps)),
                "log_buy_shape": math.log(max(obs_buy_shape, eps)) - math.log(max(mech_buy_shape, eps)),
                "log_sell_shape": math.log(max(obs_sell_shape, eps)) - math.log(max(mech_sell_shape, eps)),
            }
            for name, residual in residuals.items():
                means[active][name] = lam * means[active][name] + alpha * residual
        buy_var[active_buy] = lam * buy_var[active_buy] + alpha * (buy_err * buy_err)
        sell_var[active_sell] = lam * sell_var[active_sell] + alpha * (sell_err * sell_err)
        zero_buy_var = lam * zero_buy_var + alpha * (obs_buy_delta * obs_buy_delta)
        zero_sell_var = lam * zero_sell_var + alpha * (obs_sell_delta * obs_sell_delta)
        seen += 1

    if not buy_sq:
        return None
    return DecomposedHurdleEvaluation(
        n_scored=len(buy_sq),
        active_fraction=active_seen / max(1, seen),
        buy_rmse=math.sqrt(float(np.mean(buy_sq))),
        buy_nll=float(np.mean(buy_nll)),
        buy_zero_rmse=math.sqrt(float(np.mean(buy_zero_sq))),
        buy_zero_nll=float(np.mean(buy_zero_nll)),
        buy_mechanical_rmse=math.sqrt(float(np.mean(buy_mech_sq))),
        sell_rmse=math.sqrt(float(np.mean(sell_sq))),
        sell_nll=float(np.mean(sell_nll)),
        sell_zero_rmse=math.sqrt(float(np.mean(sell_zero_sq))),
        sell_zero_nll=float(np.mean(sell_zero_nll)),
        sell_mechanical_rmse=math.sqrt(float(np.mean(sell_mech_sq))),
    )


def online_decomposed_two_regime_vwap_model(
    rows: Sequence[MarketRow],
    *,
    min_rows: int,
    decay: float,
    initial_sigma: float,
    vwap_depth: float,
) -> DecomposedHurdleEvaluation | None:
    if len(rows) < max(min_rows, 3):
        return None
    depth = max(0.0, float(vwap_depth))
    if depth <= 1e-12:
        return None

    lam = min(0.9999, max(0.50, float(decay)))
    alpha = 1.0 - lam
    eps = 1e-9
    init_var = max(float(initial_sigma) ** 2, eps)
    component_names = ("mid", "log_spread", "log_buy_shape", "log_sell_shape")

    market_beta = {name: np.zeros(2, dtype=np.float64) for name in component_names}
    market_p = {name: np.eye(2, dtype=np.float64) * 1_000_000.0 for name in component_names}
    refresh_mean = {name: 0.0 for name in component_names}
    buy_var = {False: init_var, True: init_var}
    sell_var = {False: init_var, True: init_var}
    zero_buy_var = init_var
    zero_sell_var = init_var

    seen = 0
    refresh_seen = 0
    buy_sq: list[float] = []
    sell_sq: list[float] = []
    buy_zero_sq: list[float] = []
    sell_zero_sq: list[float] = []
    buy_mech_sq: list[float] = []
    sell_mech_sq: list[float] = []
    buy_nll: list[float] = []
    sell_nll: list[float] = []
    buy_zero_nll: list[float] = []
    sell_zero_nll: list[float] = []

    for idx in range(1, len(rows)):
        prev = rows[idx - 1]
        curr = rows[idx]

        prev_buy = vwap_to_depth(prev.asks, depth, ask_side=True)
        prev_sell = vwap_to_depth(prev.bids, depth, ask_side=False)
        curr_buy = vwap_to_depth(curr.asks, depth, ask_side=True)
        curr_sell = vwap_to_depth(curr.bids, depth, ask_side=False)
        ask_after = consume_levels(prev.asks, curr.buy_volume, ask_side=True)
        bid_after = consume_levels(prev.bids, curr.sell_volume, ask_side=False)
        mech_buy = vwap_to_depth(ask_after, depth, ask_side=True)
        mech_sell = vwap_to_depth(bid_after, depth, ask_side=False)
        mech_bid = consume_best_price(prev.bids, curr.sell_volume, side="bid")
        mech_ask = consume_best_price(prev.asks, curr.buy_volume, side="ask")

        values = [prev_buy, prev_sell, curr_buy, curr_sell, mech_buy, mech_sell, mech_bid, mech_ask]
        if not all(math.isfinite(value) for value in values):
            continue

        prev_mid = prev.mid
        obs_mid = curr.mid
        mech_mid = 0.5 * (mech_bid + mech_ask)
        prev_spread = prev.spread
        obs_spread = curr.spread
        mech_spread = mech_ask - mech_bid
        prev_buy_shape = prev_buy - prev_mid
        prev_sell_shape = prev_mid - prev_sell
        obs_buy_shape = curr_buy - obs_mid
        obs_sell_shape = obs_mid - curr_sell
        mech_buy_shape = mech_buy - mech_mid
        mech_sell_shape = mech_mid - mech_sell
        if min(
            prev_spread,
            obs_spread,
            mech_spread,
            prev_buy_shape,
            prev_sell_shape,
            obs_buy_shape,
            obs_sell_shape,
            mech_buy_shape,
            mech_sell_shape,
        ) <= eps:
            continue

        obs_components = {
            "mid": obs_mid - prev_mid,
            "log_spread": math.log(obs_spread) - math.log(prev_spread),
            "log_buy_shape": math.log(obs_buy_shape) - math.log(prev_buy_shape),
            "log_sell_shape": math.log(obs_sell_shape) - math.log(prev_sell_shape),
        }
        mech_components = {
            "mid": mech_mid - prev_mid,
            "log_spread": math.log(mech_spread) - math.log(prev_spread),
            "log_buy_shape": math.log(mech_buy_shape) - math.log(prev_buy_shape),
            "log_sell_shape": math.log(mech_sell_shape) - math.log(prev_sell_shape),
        }
        if not all(math.isfinite(value) for value in (*obs_components.values(), *mech_components.values())):
            continue

        total_flow = curr.buy_volume + curr.sell_volume
        refresh = total_flow <= eps and abs(mech_components["mid"]) <= eps
        refresh_seen += int(refresh)

        pred_components: dict[str, float] = {}
        for name in component_names:
            if refresh:
                pred_components[name] = refresh_mean[name]
            else:
                x = np.array([1.0, mech_components[name]], dtype=np.float64)
                pred_components[name] = float(x @ market_beta[name])

        pred_mid = prev_mid + pred_components["mid"]
        pred_buy_shape = prev_buy_shape * math.exp(pred_components["log_buy_shape"])
        pred_sell_shape = prev_sell_shape * math.exp(pred_components["log_sell_shape"])
        pred_buy = pred_mid + pred_buy_shape
        pred_sell = pred_mid - pred_sell_shape

        obs_buy_delta = curr_buy - prev_buy
        obs_sell_delta = curr_sell - prev_sell
        pred_buy_delta = pred_buy - prev_buy
        pred_sell_delta = pred_sell - prev_sell
        mech_buy_delta = mech_buy - prev_buy
        mech_sell_delta = mech_sell - prev_sell
        buy_err = obs_buy_delta - pred_buy_delta
        sell_err = obs_sell_delta - pred_sell_delta

        if seen >= min_rows:
            buy_sq.append(buy_err * buy_err)
            sell_sq.append(sell_err * sell_err)
            buy_zero_sq.append(obs_buy_delta * obs_buy_delta)
            sell_zero_sq.append(obs_sell_delta * obs_sell_delta)
            buy_mech_sq.append((obs_buy_delta - mech_buy_delta) ** 2)
            sell_mech_sq.append((obs_sell_delta - mech_sell_delta) ** 2)
            buy_nll.append(gaussian_nll(buy_err, max(buy_var[refresh], eps)))
            sell_nll.append(gaussian_nll(sell_err, max(sell_var[refresh], eps)))
            buy_zero_nll.append(gaussian_nll(obs_buy_delta, max(zero_buy_var, eps)))
            sell_zero_nll.append(gaussian_nll(obs_sell_delta, max(zero_sell_var, eps)))

        for name in component_names:
            target = obs_components[name]
            if refresh:
                refresh_mean[name] = lam * refresh_mean[name] + alpha * target
            else:
                x = np.array([1.0, mech_components[name]], dtype=np.float64)
                comp_err = target - float(x @ market_beta[name])
                px = market_p[name] @ x
                denom = lam + float(x @ px)
                gain = px / denom
                market_beta[name] = market_beta[name] + gain * comp_err
                market_p[name] = (market_p[name] - np.outer(gain, x) @ market_p[name]) / lam
                market_p[name] = 0.5 * (market_p[name] + market_p[name].T)

        buy_var[refresh] = lam * buy_var[refresh] + alpha * (buy_err * buy_err)
        sell_var[refresh] = lam * sell_var[refresh] + alpha * (sell_err * sell_err)
        zero_buy_var = lam * zero_buy_var + alpha * (obs_buy_delta * obs_buy_delta)
        zero_sell_var = lam * zero_sell_var + alpha * (obs_sell_delta * obs_sell_delta)
        seen += 1

    if not buy_sq:
        return None
    return DecomposedHurdleEvaluation(
        n_scored=len(buy_sq),
        active_fraction=1.0 - (refresh_seen / max(1, seen)),
        buy_rmse=math.sqrt(float(np.mean(buy_sq))),
        buy_nll=float(np.mean(buy_nll)),
        buy_zero_rmse=math.sqrt(float(np.mean(buy_zero_sq))),
        buy_zero_nll=float(np.mean(buy_zero_nll)),
        buy_mechanical_rmse=math.sqrt(float(np.mean(buy_mech_sq))),
        sell_rmse=math.sqrt(float(np.mean(sell_sq))),
        sell_nll=float(np.mean(sell_nll)),
        sell_zero_rmse=math.sqrt(float(np.mean(sell_zero_sq))),
        sell_zero_nll=float(np.mean(sell_zero_nll)),
        sell_mechanical_rmse=math.sqrt(float(np.mean(sell_mech_sq))),
    )


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-min(value, 60.0))
        return 1.0 / (1.0 + z)
    z = math.exp(max(value, -60.0))
    return z / (1.0 + z)


def normal_logpdf(error: float, variance: float) -> float:
    var = max(variance, 1e-9)
    return -0.5 * (math.log(2.0 * math.pi * var) + (error * error) / var)


def student_t_logpdf(error: float, variance: float, df: float) -> float:
    nu = max(2.1, float(df))
    var = max(variance, 1e-9)
    scale2 = var * (nu - 2.0) / nu
    scale2 = max(scale2, 1e-9)
    return (
        math.lgamma((nu + 1.0) / 2.0)
        - math.lgamma(nu / 2.0)
        - 0.5 * math.log(nu * math.pi * scale2)
        - ((nu + 1.0) / 2.0) * math.log1p((error * error) / (nu * scale2))
    )


def refresh_logpdf(error: float, variance: float, distribution: str, student_df: float) -> float:
    if distribution == "student_t":
        return student_t_logpdf(error, variance, student_df)
    return normal_logpdf(error, variance)


def two_gaussian_logpdf(error: float, quiet_var: float, jump_var: float, jump_prob: float) -> float:
    p_jump = min(0.999, max(0.001, float(jump_prob)))
    return logaddexp(
        math.log1p(-p_jump) + normal_logpdf(error, quiet_var),
        math.log(p_jump) + normal_logpdf(error, jump_var),
    )


def two_gaussian_jump_resp(error: float, quiet_var: float, jump_var: float, jump_prob: float) -> float:
    p_jump = min(0.999, max(0.001, float(jump_prob)))
    log_jump = math.log(p_jump) + normal_logpdf(error, jump_var)
    log_mix = two_gaussian_logpdf(error, quiet_var, jump_var, p_jump)
    return math.exp(log_jump - log_mix)


def logaddexp(a: float, b: float) -> float:
    high = max(a, b)
    low = min(a, b)
    return high + math.log1p(math.exp(low - high))


def clamp_log_delta(value: float) -> float:
    return min(5.0, max(-5.0, value))


def cap_deviation(value: float, anchor: float, cap: float) -> float:
    if cap <= 0 or not math.isfinite(value) or not math.isfinite(anchor):
        return value
    return anchor + min(cap, max(-cap, value - anchor))


def weighted_rls_update(
    beta: np.ndarray,
    p: np.ndarray,
    x: np.ndarray,
    target: float,
    *,
    weight: float,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    w = max(0.0, min(1.0, float(weight)))
    if w <= 1e-6:
        return beta, p
    scale = math.sqrt(w)
    xw = x * scale
    yw = target * scale
    px = p @ xw
    denom = lam + float(xw @ px)
    gain = px / denom
    beta = beta + gain * (yw - float(xw @ beta))
    p = (p - np.outer(gain, xw) @ p) / lam
    p = 0.5 * (p + p.T)
    return beta, p


def online_decomposed_soft_mixture_vwap_model(
    rows: Sequence[MarketRow],
    *,
    min_rows: int,
    decay: float,
    initial_sigma: float,
    vwap_depth: float,
    gate_learning_rate: float,
    market_deviation_cap_mult: float = 0.0,
    market_guard_mode: str = "cap",
    diagnostics: list[dict[str, object]] | None = None,
    diagnostic_section: str = "",
) -> DecomposedHurdleEvaluation | None:
    if len(rows) < max(min_rows, 3):
        return None
    depth = max(0.0, float(vwap_depth))
    if depth <= 1e-12:
        return None

    lam = min(0.9999, max(0.50, float(decay)))
    alpha = 1.0 - lam
    eps = 1e-9
    init_var = max(float(initial_sigma) ** 2, eps)
    component_names = ("mid", "log_spread", "log_buy_shape", "log_sell_shape")

    market_beta = {name: np.zeros(2, dtype=np.float64) for name in component_names}
    market_p = {name: np.eye(2, dtype=np.float64) * 1_000_000.0 for name in component_names}
    refresh_mean = {name: 0.0 for name in component_names}
    gate_beta = np.zeros(4, dtype=np.float64)
    gate_beta[0] = -1.0
    market_buy_var = init_var
    market_sell_var = init_var
    refresh_buy_var = init_var
    refresh_sell_var = init_var
    zero_buy_var = init_var
    zero_sell_var = init_var

    seen = 0
    market_prob_sum = 0.0
    buy_sq: list[float] = []
    sell_sq: list[float] = []
    buy_zero_sq: list[float] = []
    sell_zero_sq: list[float] = []
    buy_mech_sq: list[float] = []
    sell_mech_sq: list[float] = []
    buy_nll: list[float] = []
    sell_nll: list[float] = []
    buy_zero_nll: list[float] = []
    sell_zero_nll: list[float] = []

    for idx in range(1, len(rows)):
        prev = rows[idx - 1]
        curr = rows[idx]

        prev_buy = vwap_to_depth(prev.asks, depth, ask_side=True)
        prev_sell = vwap_to_depth(prev.bids, depth, ask_side=False)
        curr_buy = vwap_to_depth(curr.asks, depth, ask_side=True)
        curr_sell = vwap_to_depth(curr.bids, depth, ask_side=False)
        ask_after = consume_levels(prev.asks, curr.buy_volume, ask_side=True)
        bid_after = consume_levels(prev.bids, curr.sell_volume, ask_side=False)
        mech_buy = vwap_to_depth(ask_after, depth, ask_side=True)
        mech_sell = vwap_to_depth(bid_after, depth, ask_side=False)
        mech_bid = consume_best_price(prev.bids, curr.sell_volume, side="bid")
        mech_ask = consume_best_price(prev.asks, curr.buy_volume, side="ask")

        values = [prev_buy, prev_sell, curr_buy, curr_sell, mech_buy, mech_sell, mech_bid, mech_ask]
        if not all(math.isfinite(value) for value in values):
            continue

        prev_mid = prev.mid
        obs_mid = curr.mid
        mech_mid = 0.5 * (mech_bid + mech_ask)
        prev_spread = prev.spread
        obs_spread = curr.spread
        mech_spread = mech_ask - mech_bid
        prev_buy_shape = prev_buy - prev_mid
        prev_sell_shape = prev_mid - prev_sell
        obs_buy_shape = curr_buy - obs_mid
        obs_sell_shape = obs_mid - curr_sell
        mech_buy_shape = mech_buy - mech_mid
        mech_sell_shape = mech_mid - mech_sell
        if min(
            prev_spread,
            obs_spread,
            mech_spread,
            prev_buy_shape,
            prev_sell_shape,
            obs_buy_shape,
            obs_sell_shape,
            mech_buy_shape,
            mech_sell_shape,
        ) <= eps:
            continue

        obs_components = {
            "mid": obs_mid - prev_mid,
            "log_spread": math.log(obs_spread) - math.log(prev_spread),
            "log_buy_shape": math.log(obs_buy_shape) - math.log(prev_buy_shape),
            "log_sell_shape": math.log(obs_sell_shape) - math.log(prev_sell_shape),
        }
        mech_components = {
            "mid": mech_mid - prev_mid,
            "log_spread": math.log(mech_spread) - math.log(prev_spread),
            "log_buy_shape": math.log(mech_buy_shape) - math.log(prev_buy_shape),
            "log_sell_shape": math.log(mech_sell_shape) - math.log(prev_sell_shape),
        }
        if not all(math.isfinite(value) for value in (*obs_components.values(), *mech_components.values())):
            continue

        total_flow = curr.buy_volume + curr.sell_volume
        gate_x = np.array(
            [
                1.0,
                1.0 if total_flow <= eps else 0.0,
                min(abs(mech_components["mid"]) / 1_000.0, 10.0),
                min(max(prev_spread, 0.0) / 1_000.0, 10.0),
            ],
            dtype=np.float64,
        )
        q_refresh = sigmoid(float(gate_x @ gate_beta))
        q_refresh = min(0.999, max(0.001, q_refresh))
        q_market = 1.0 - q_refresh
        market_prob_sum += q_market

        market_components: dict[str, float] = {}
        refresh_components: dict[str, float] = {}
        for name in component_names:
            x = np.array([1.0, mech_components[name]], dtype=np.float64)
            market_components[name] = float(x @ market_beta[name])
            refresh_components[name] = refresh_mean[name]

        market_mid = prev_mid + market_components["mid"]
        market_buy = market_mid + prev_buy_shape * math.exp(clamp_log_delta(market_components["log_buy_shape"]))
        market_sell = market_mid - prev_sell_shape * math.exp(clamp_log_delta(market_components["log_sell_shape"]))
        refresh_mid = prev_mid + refresh_components["mid"]
        refresh_buy = refresh_mid + prev_buy_shape * math.exp(clamp_log_delta(refresh_components["log_buy_shape"]))
        refresh_sell = refresh_mid - prev_sell_shape * math.exp(clamp_log_delta(refresh_components["log_sell_shape"]))

        market_buy_delta = market_buy - prev_buy
        market_sell_delta = market_sell - prev_sell
        refresh_buy_delta = refresh_buy - prev_buy
        refresh_sell_delta = refresh_sell - prev_sell
        obs_buy_delta = curr_buy - prev_buy
        obs_sell_delta = curr_sell - prev_sell
        mech_buy_delta = mech_buy - prev_buy
        mech_sell_delta = mech_sell - prev_sell
        point_market_buy_delta = market_buy_delta
        point_market_sell_delta = market_sell_delta
        point_q_market = q_market
        point_q_refresh = q_refresh
        market_point_guarded = False
        if market_deviation_cap_mult > 0:
            point_cap = float(market_deviation_cap_mult) * max(prev_spread, 1.0)
            market_point_guarded = (
                abs(market_buy_delta - mech_buy_delta) > point_cap
                or abs(market_sell_delta - mech_sell_delta) > point_cap
            )
            if market_guard_mode == "drop" and market_point_guarded:
                point_q_market = 0.0
                point_q_refresh = 1.0
            else:
                point_market_buy_delta = cap_deviation(market_buy_delta, mech_buy_delta, point_cap)
                point_market_sell_delta = cap_deviation(market_sell_delta, mech_sell_delta, point_cap)
        pred_buy_delta = point_q_market * point_market_buy_delta + point_q_refresh * refresh_buy_delta
        pred_sell_delta = point_q_market * point_market_sell_delta + point_q_refresh * refresh_sell_delta

        buy_err = obs_buy_delta - pred_buy_delta
        sell_err = obs_sell_delta - pred_sell_delta
        market_buy_err = obs_buy_delta - market_buy_delta
        market_sell_err = obs_sell_delta - market_sell_delta
        refresh_buy_err = obs_buy_delta - refresh_buy_delta
        refresh_sell_err = obs_sell_delta - refresh_sell_delta

        log_market = (
            math.log(q_market)
            + normal_logpdf(market_buy_err, market_buy_var)
            + normal_logpdf(market_sell_err, market_sell_var)
        )
        log_refresh = (
            math.log(q_refresh)
            + normal_logpdf(refresh_buy_err, refresh_buy_var)
            + normal_logpdf(refresh_sell_err, refresh_sell_var)
        )
        log_norm = logaddexp(log_market, log_refresh)
        refresh_resp = math.exp(log_refresh - log_norm)
        market_resp = 1.0 - refresh_resp

        if seen >= min_rows:
            buy_sq.append(buy_err * buy_err)
            sell_sq.append(sell_err * sell_err)
            buy_zero_sq.append(obs_buy_delta * obs_buy_delta)
            sell_zero_sq.append(obs_sell_delta * obs_sell_delta)
            buy_mech_sq.append((obs_buy_delta - mech_buy_delta) ** 2)
            sell_mech_sq.append((obs_sell_delta - mech_sell_delta) ** 2)
            buy_mix_log = logaddexp(
                math.log(q_market) + normal_logpdf(market_buy_err, market_buy_var),
                math.log(q_refresh) + normal_logpdf(refresh_buy_err, refresh_buy_var),
            )
            sell_mix_log = logaddexp(
                math.log(q_market) + normal_logpdf(market_sell_err, market_sell_var),
                math.log(q_refresh) + normal_logpdf(refresh_sell_err, refresh_sell_var),
            )
            buy_nll.append(-buy_mix_log)
            sell_nll.append(-sell_mix_log)
            buy_zero_nll.append(gaussian_nll(obs_buy_delta, max(zero_buy_var, eps)))
            sell_zero_nll.append(gaussian_nll(obs_sell_delta, max(zero_sell_var, eps)))
            if diagnostics is not None:
                diagnostics.append(
                    {
                        "section": diagnostic_section,
                        "ts": curr.ts,
                        "score": max(abs(buy_err), abs(sell_err)),
                        "buy_err": buy_err,
                        "sell_err": sell_err,
                        "obs_buy_delta": obs_buy_delta,
                        "pred_buy_delta": pred_buy_delta,
                        "mech_buy_delta": mech_buy_delta,
                        "market_buy_delta": market_buy_delta,
                        "point_market_buy_delta": point_market_buy_delta,
                        "refresh_buy_delta": refresh_buy_delta,
                        "obs_sell_delta": obs_sell_delta,
                        "pred_sell_delta": pred_sell_delta,
                        "mech_sell_delta": mech_sell_delta,
                        "market_sell_delta": market_sell_delta,
                        "point_market_sell_delta": point_market_sell_delta,
                        "refresh_sell_delta": refresh_sell_delta,
                        "q_market": q_market,
                        "point_q_market": point_q_market,
                        "market_point_guarded": market_point_guarded,
                        "market_resp": market_resp,
                        "total_flow": total_flow,
                        "buy_volume": curr.buy_volume,
                        "sell_volume": curr.sell_volume,
                        "prev_spread": prev_spread,
                        "obs_spread": obs_spread,
                        "mech_mid_delta": mech_components["mid"],
                        "prev_buy_shape": prev_buy_shape,
                        "prev_sell_shape": prev_sell_shape,
                        "market_log_buy_shape_delta": market_components["log_buy_shape"],
                        "market_log_sell_shape_delta": market_components["log_sell_shape"],
                        "refresh_log_buy_shape_delta": refresh_components["log_buy_shape"],
                        "refresh_log_sell_shape_delta": refresh_components["log_sell_shape"],
                    }
                )

        lr = max(0.0, float(gate_learning_rate))
        gate_beta = gate_beta + lr * (refresh_resp - q_refresh) * gate_x
        gate_beta = np.clip(gate_beta, -20.0, 20.0)

        for name in component_names:
            target = obs_components[name]
            x = np.array([1.0, mech_components[name]], dtype=np.float64)
            market_beta[name], market_p[name] = weighted_rls_update(
                market_beta[name],
                market_p[name],
                x,
                target,
                weight=market_resp,
                lam=lam,
            )
            refresh_mean[name] = refresh_mean[name] + alpha * refresh_resp * (target - refresh_mean[name])

        market_buy_var = (1.0 - alpha * market_resp) * market_buy_var + alpha * market_resp * (market_buy_err * market_buy_err)
        market_sell_var = (1.0 - alpha * market_resp) * market_sell_var + alpha * market_resp * (market_sell_err * market_sell_err)
        refresh_buy_var = (1.0 - alpha * refresh_resp) * refresh_buy_var + alpha * refresh_resp * (refresh_buy_err * refresh_buy_err)
        refresh_sell_var = (1.0 - alpha * refresh_resp) * refresh_sell_var + alpha * refresh_resp * (refresh_sell_err * refresh_sell_err)
        zero_buy_var = lam * zero_buy_var + alpha * (obs_buy_delta * obs_buy_delta)
        zero_sell_var = lam * zero_sell_var + alpha * (obs_sell_delta * obs_sell_delta)
        seen += 1

    if not buy_sq:
        return None
    return DecomposedHurdleEvaluation(
        n_scored=len(buy_sq),
        active_fraction=market_prob_sum / max(1, seen),
        buy_rmse=math.sqrt(float(np.mean(buy_sq))),
        buy_nll=float(np.mean(buy_nll)),
        buy_zero_rmse=math.sqrt(float(np.mean(buy_zero_sq))),
        buy_zero_nll=float(np.mean(buy_zero_nll)),
        buy_mechanical_rmse=math.sqrt(float(np.mean(buy_mech_sq))),
        sell_rmse=math.sqrt(float(np.mean(sell_sq))),
        sell_nll=float(np.mean(sell_nll)),
        sell_zero_rmse=math.sqrt(float(np.mean(sell_zero_sq))),
        sell_zero_nll=float(np.mean(sell_zero_nll)),
        sell_mechanical_rmse=math.sqrt(float(np.mean(sell_mech_sq))),
    )


def online_decomposed_em_mixture_vwap_model(
    rows: Sequence[MarketRow],
    *,
    min_rows: int,
    decay: float,
    initial_sigma: float,
    vwap_depth: float,
    point_mode: str,
) -> DecomposedHurdleEvaluation | None:
    if len(rows) < max(min_rows, 3):
        return None
    depth = max(0.0, float(vwap_depth))
    if depth <= 1e-12:
        return None

    lam = min(0.9999, max(0.50, float(decay)))
    alpha = 1.0 - lam
    eps = 1e-9
    init_var = max(float(initial_sigma) ** 2, eps)
    component_names = ("mid", "log_buy_shape", "log_sell_shape")

    market_beta = {name: np.zeros(2, dtype=np.float64) for name in component_names}
    market_p = {name: np.eye(2, dtype=np.float64) * 1_000_000.0 for name in component_names}
    refresh_mean = {name: 0.0 for name in component_names}
    pi_market = 0.5
    market_buy_var = init_var
    market_sell_var = init_var
    refresh_buy_var = init_var
    refresh_sell_var = init_var
    zero_buy_var = init_var
    zero_sell_var = init_var

    seen = 0
    market_prob_sum = 0.0
    buy_sq: list[float] = []
    sell_sq: list[float] = []
    buy_zero_sq: list[float] = []
    sell_zero_sq: list[float] = []
    buy_mech_sq: list[float] = []
    sell_mech_sq: list[float] = []
    buy_nll: list[float] = []
    sell_nll: list[float] = []
    buy_zero_nll: list[float] = []
    sell_zero_nll: list[float] = []

    for idx in range(1, len(rows)):
        prev = rows[idx - 1]
        curr = rows[idx]

        prev_buy = vwap_to_depth(prev.asks, depth, ask_side=True)
        prev_sell = vwap_to_depth(prev.bids, depth, ask_side=False)
        curr_buy = vwap_to_depth(curr.asks, depth, ask_side=True)
        curr_sell = vwap_to_depth(curr.bids, depth, ask_side=False)
        ask_after = consume_levels(prev.asks, curr.buy_volume, ask_side=True)
        bid_after = consume_levels(prev.bids, curr.sell_volume, ask_side=False)
        mech_buy = vwap_to_depth(ask_after, depth, ask_side=True)
        mech_sell = vwap_to_depth(bid_after, depth, ask_side=False)
        mech_bid = consume_best_price(prev.bids, curr.sell_volume, side="bid")
        mech_ask = consume_best_price(prev.asks, curr.buy_volume, side="ask")

        values = [prev_buy, prev_sell, curr_buy, curr_sell, mech_buy, mech_sell, mech_bid, mech_ask]
        if not all(math.isfinite(value) for value in values):
            continue

        prev_mid = prev.mid
        obs_mid = curr.mid
        mech_mid = 0.5 * (mech_bid + mech_ask)
        prev_spread = prev.spread
        obs_spread = curr.spread
        prev_buy_shape = prev_buy - prev_mid
        prev_sell_shape = prev_mid - prev_sell
        obs_buy_shape = curr_buy - obs_mid
        obs_sell_shape = obs_mid - curr_sell
        mech_buy_shape = mech_buy - mech_mid
        mech_sell_shape = mech_mid - mech_sell
        if min(
            prev_spread,
            obs_spread,
            prev_buy_shape,
            prev_sell_shape,
            obs_buy_shape,
            obs_sell_shape,
            mech_buy_shape,
            mech_sell_shape,
        ) <= eps:
            continue

        obs_components = {
            "mid": obs_mid - prev_mid,
            "log_buy_shape": math.log(obs_buy_shape) - math.log(prev_buy_shape),
            "log_sell_shape": math.log(obs_sell_shape) - math.log(prev_sell_shape),
        }
        mech_components = {
            "mid": mech_mid - prev_mid,
            "log_buy_shape": math.log(mech_buy_shape) - math.log(prev_buy_shape),
            "log_sell_shape": math.log(mech_sell_shape) - math.log(prev_sell_shape),
        }
        if not all(math.isfinite(value) for value in (*obs_components.values(), *mech_components.values())):
            continue

        q_market = min(0.999, max(0.001, pi_market))
        q_refresh = 1.0 - q_market
        market_prob_sum += q_market

        market_components: dict[str, float] = {}
        refresh_components: dict[str, float] = {}
        for name in component_names:
            x = np.array([1.0, mech_components[name]], dtype=np.float64)
            market_components[name] = float(x @ market_beta[name])
            refresh_components[name] = refresh_mean[name]

        market_mid = prev_mid + market_components["mid"]
        market_buy = market_mid + prev_buy_shape * math.exp(clamp_log_delta(market_components["log_buy_shape"]))
        market_sell = market_mid - prev_sell_shape * math.exp(clamp_log_delta(market_components["log_sell_shape"]))
        refresh_mid = prev_mid + refresh_components["mid"]
        refresh_buy = refresh_mid + prev_buy_shape * math.exp(clamp_log_delta(refresh_components["log_buy_shape"]))
        refresh_sell = refresh_mid - prev_sell_shape * math.exp(clamp_log_delta(refresh_components["log_sell_shape"]))

        market_buy_delta = market_buy - prev_buy
        market_sell_delta = market_sell - prev_sell
        refresh_buy_delta = refresh_buy - prev_buy
        refresh_sell_delta = refresh_sell - prev_sell
        obs_buy_delta = curr_buy - prev_buy
        obs_sell_delta = curr_sell - prev_sell
        mech_buy_delta = mech_buy - prev_buy
        mech_sell_delta = mech_sell - prev_sell

        if point_mode == "map":
            pred_buy_delta = market_buy_delta if q_market >= q_refresh else refresh_buy_delta
            pred_sell_delta = market_sell_delta if q_market >= q_refresh else refresh_sell_delta
        else:
            pred_buy_delta = q_market * market_buy_delta + q_refresh * refresh_buy_delta
            pred_sell_delta = q_market * market_sell_delta + q_refresh * refresh_sell_delta

        buy_err = obs_buy_delta - pred_buy_delta
        sell_err = obs_sell_delta - pred_sell_delta
        market_buy_err = obs_buy_delta - market_buy_delta
        market_sell_err = obs_sell_delta - market_sell_delta
        refresh_buy_err = obs_buy_delta - refresh_buy_delta
        refresh_sell_err = obs_sell_delta - refresh_sell_delta

        log_market = (
            math.log(q_market)
            + normal_logpdf(market_buy_err, market_buy_var)
            + normal_logpdf(market_sell_err, market_sell_var)
        )
        log_refresh = (
            math.log(q_refresh)
            + normal_logpdf(refresh_buy_err, refresh_buy_var)
            + normal_logpdf(refresh_sell_err, refresh_sell_var)
        )
        log_norm = logaddexp(log_market, log_refresh)
        market_resp = math.exp(log_market - log_norm)
        refresh_resp = 1.0 - market_resp

        if seen >= min_rows:
            buy_sq.append(buy_err * buy_err)
            sell_sq.append(sell_err * sell_err)
            buy_zero_sq.append(obs_buy_delta * obs_buy_delta)
            sell_zero_sq.append(obs_sell_delta * obs_sell_delta)
            buy_mech_sq.append((obs_buy_delta - mech_buy_delta) ** 2)
            sell_mech_sq.append((obs_sell_delta - mech_sell_delta) ** 2)
            buy_mix_log = logaddexp(
                math.log(q_market) + normal_logpdf(market_buy_err, market_buy_var),
                math.log(q_refresh) + normal_logpdf(refresh_buy_err, refresh_buy_var),
            )
            sell_mix_log = logaddexp(
                math.log(q_market) + normal_logpdf(market_sell_err, market_sell_var),
                math.log(q_refresh) + normal_logpdf(refresh_sell_err, refresh_sell_var),
            )
            buy_nll.append(-buy_mix_log)
            sell_nll.append(-sell_mix_log)
            buy_zero_nll.append(gaussian_nll(obs_buy_delta, max(zero_buy_var, eps)))
            sell_zero_nll.append(gaussian_nll(obs_sell_delta, max(zero_sell_var, eps)))

        pi_market = min(0.999, max(0.001, lam * pi_market + alpha * market_resp))

        for name in component_names:
            target = obs_components[name]
            x = np.array([1.0, mech_components[name]], dtype=np.float64)
            market_beta[name], market_p[name] = weighted_rls_update(
                market_beta[name],
                market_p[name],
                x,
                target,
                weight=market_resp,
                lam=lam,
            )
            refresh_mean[name] = refresh_mean[name] + alpha * refresh_resp * (target - refresh_mean[name])

        market_buy_var = (1.0 - alpha * market_resp) * market_buy_var + alpha * market_resp * (market_buy_err * market_buy_err)
        market_sell_var = (1.0 - alpha * market_resp) * market_sell_var + alpha * market_resp * (market_sell_err * market_sell_err)
        refresh_buy_var = (1.0 - alpha * refresh_resp) * refresh_buy_var + alpha * refresh_resp * (refresh_buy_err * refresh_buy_err)
        refresh_sell_var = (1.0 - alpha * refresh_resp) * refresh_sell_var + alpha * refresh_resp * (refresh_sell_err * refresh_sell_err)
        zero_buy_var = lam * zero_buy_var + alpha * (obs_buy_delta * obs_buy_delta)
        zero_sell_var = lam * zero_sell_var + alpha * (obs_sell_delta * obs_sell_delta)
        seen += 1

    if not buy_sq:
        return None
    return DecomposedHurdleEvaluation(
        n_scored=len(buy_sq),
        active_fraction=market_prob_sum / max(1, seen),
        buy_rmse=math.sqrt(float(np.mean(buy_sq))),
        buy_nll=float(np.mean(buy_nll)),
        buy_zero_rmse=math.sqrt(float(np.mean(buy_zero_sq))),
        buy_zero_nll=float(np.mean(buy_zero_nll)),
        buy_mechanical_rmse=math.sqrt(float(np.mean(buy_mech_sq))),
        sell_rmse=math.sqrt(float(np.mean(sell_sq))),
        sell_nll=float(np.mean(sell_nll)),
        sell_zero_rmse=math.sqrt(float(np.mean(sell_zero_sq))),
        sell_zero_nll=float(np.mean(sell_zero_nll)),
        sell_mechanical_rmse=math.sqrt(float(np.mean(sell_mech_sq))),
    )


def online_decomposed_flow_gated_mixture_vwap_model(
    rows: Sequence[MarketRow],
    *,
    min_rows: int,
    decay: float,
    initial_sigma: float,
    vwap_depth: float,
    gate_learning_rate: float,
    gate_kind: str,
    gate_bins: Sequence[float],
    point_mode: str,
    refresh_distribution: str,
    refresh_student_df: float,
    refresh_jump_sigma_mult: float,
    spread_features: str,
) -> DecomposedHurdleEvaluation | None:
    if len(rows) < max(min_rows, 3):
        return None
    depth = max(0.0, float(vwap_depth))
    if depth <= 1e-12:
        return None

    lam = min(0.9999, max(0.50, float(decay)))
    alpha = 1.0 - lam
    eps = 1e-9
    init_var = max(float(initial_sigma) ** 2, eps)
    component_names = ("mid", "log_buy_shape", "log_sell_shape")

    market_beta = {name: np.zeros(2, dtype=np.float64) for name in component_names}
    market_p = {name: np.eye(2, dtype=np.float64) * 1_000_000.0 for name in component_names}
    refresh_mean = {name: 0.0 for name in component_names}
    gate_feature_count = 2
    if spread_features == "prev":
        gate_feature_count = 3
    elif spread_features == "both":
        gate_feature_count = 4
    gate_beta = np.zeros(gate_feature_count, dtype=np.float64)
    sorted_gate_bins = sorted(float(edge) for edge in gate_bins)
    gate_bin_probs = np.full(len(sorted_gate_bins) + 1, 0.5, dtype=np.float64)
    market_buy_var = init_var
    market_sell_var = init_var
    refresh_buy_var = init_var
    refresh_sell_var = init_var
    jump_mult = max(1.01, float(refresh_jump_sigma_mult))
    refresh_jump_prob = 0.05
    refresh_jump_buy_var = init_var * jump_mult * jump_mult
    refresh_jump_sell_var = init_var * jump_mult * jump_mult
    zero_buy_var = init_var
    zero_sell_var = init_var

    seen = 0
    market_prob_sum = 0.0
    buy_sq: list[float] = []
    sell_sq: list[float] = []
    buy_zero_sq: list[float] = []
    sell_zero_sq: list[float] = []
    buy_mech_sq: list[float] = []
    sell_mech_sq: list[float] = []
    buy_nll: list[float] = []
    sell_nll: list[float] = []
    buy_zero_nll: list[float] = []
    sell_zero_nll: list[float] = []

    for idx in range(1, len(rows)):
        prev = rows[idx - 1]
        curr = rows[idx]

        prev_buy = vwap_to_depth(prev.asks, depth, ask_side=True)
        prev_sell = vwap_to_depth(prev.bids, depth, ask_side=False)
        curr_buy = vwap_to_depth(curr.asks, depth, ask_side=True)
        curr_sell = vwap_to_depth(curr.bids, depth, ask_side=False)
        ask_after = consume_levels(prev.asks, curr.buy_volume, ask_side=True)
        bid_after = consume_levels(prev.bids, curr.sell_volume, ask_side=False)
        mech_buy = vwap_to_depth(ask_after, depth, ask_side=True)
        mech_sell = vwap_to_depth(bid_after, depth, ask_side=False)
        mech_bid = consume_best_price(prev.bids, curr.sell_volume, side="bid")
        mech_ask = consume_best_price(prev.asks, curr.buy_volume, side="ask")

        values = [prev_buy, prev_sell, curr_buy, curr_sell, mech_buy, mech_sell, mech_bid, mech_ask]
        if not all(math.isfinite(value) for value in values):
            continue

        prev_mid = prev.mid
        obs_mid = curr.mid
        mech_mid = 0.5 * (mech_bid + mech_ask)
        prev_spread = prev.spread
        obs_spread = curr.spread
        mech_spread = mech_ask - mech_bid
        prev_buy_shape = prev_buy - prev_mid
        prev_sell_shape = prev_mid - prev_sell
        obs_buy_shape = curr_buy - obs_mid
        obs_sell_shape = obs_mid - curr_sell
        mech_buy_shape = mech_buy - mech_mid
        mech_sell_shape = mech_mid - mech_sell
        if min(
            prev_spread,
            obs_spread,
            mech_spread,
            prev_buy_shape,
            prev_sell_shape,
            obs_buy_shape,
            obs_sell_shape,
            mech_buy_shape,
            mech_sell_shape,
        ) <= eps:
            continue

        obs_components = {
            "mid": obs_mid - prev_mid,
            "log_buy_shape": math.log(obs_buy_shape) - math.log(prev_buy_shape),
            "log_sell_shape": math.log(obs_sell_shape) - math.log(prev_sell_shape),
        }
        mech_log_spread_delta = math.log(mech_spread) - math.log(prev_spread)
        mech_components = {
            "mid": mech_mid - prev_mid,
            "log_buy_shape": math.log(mech_buy_shape) - math.log(prev_buy_shape),
            "log_sell_shape": math.log(mech_sell_shape) - math.log(prev_sell_shape),
        }
        if not all(
            math.isfinite(value)
            for value in (*obs_components.values(), *mech_components.values(), mech_log_spread_delta)
        ):
            continue

        total_flow = max(0.0, curr.buy_volume + curr.sell_volume)
        gate_values = [
            1.0,
            min(math.log1p(total_flow), 12.0),
        ]
        if spread_features in {"prev", "both"}:
            gate_values.append(min(math.log1p(max(prev_spread, 0.0)), 12.0))
        if spread_features == "both":
            gate_values.append(min(abs(mech_log_spread_delta), 5.0))
        gate_x = np.array(gate_values, dtype=np.float64)
        gate_bin_idx = flow_bin_index(gate_values[1], sorted_gate_bins)
        if gate_kind == "binned":
            q_market = float(gate_bin_probs[gate_bin_idx])
        else:
            q_market = sigmoid(float(gate_x @ gate_beta))
        q_market = min(0.999, max(0.001, q_market))
        q_refresh = 1.0 - q_market
        market_prob_sum += q_market

        market_components: dict[str, float] = {}
        refresh_components: dict[str, float] = {}
        for name in component_names:
            x = np.array([1.0, mech_components[name]], dtype=np.float64)
            market_components[name] = float(x @ market_beta[name])
            refresh_components[name] = refresh_mean[name]

        market_mid = prev_mid + market_components["mid"]
        market_buy = market_mid + prev_buy_shape * math.exp(clamp_log_delta(market_components["log_buy_shape"]))
        market_sell = market_mid - prev_sell_shape * math.exp(clamp_log_delta(market_components["log_sell_shape"]))
        refresh_mid = prev_mid + refresh_components["mid"]
        refresh_buy = refresh_mid + prev_buy_shape * math.exp(clamp_log_delta(refresh_components["log_buy_shape"]))
        refresh_sell = refresh_mid - prev_sell_shape * math.exp(clamp_log_delta(refresh_components["log_sell_shape"]))

        market_buy_delta = market_buy - prev_buy
        market_sell_delta = market_sell - prev_sell
        refresh_buy_delta = refresh_buy - prev_buy
        refresh_sell_delta = refresh_sell - prev_sell
        obs_buy_delta = curr_buy - prev_buy
        obs_sell_delta = curr_sell - prev_sell
        mech_buy_delta = mech_buy - prev_buy
        mech_sell_delta = mech_sell - prev_sell

        if point_mode == "map":
            pred_buy_delta = market_buy_delta if q_market >= q_refresh else refresh_buy_delta
            pred_sell_delta = market_sell_delta if q_market >= q_refresh else refresh_sell_delta
        else:
            pred_buy_delta = q_market * market_buy_delta + q_refresh * refresh_buy_delta
            pred_sell_delta = q_market * market_sell_delta + q_refresh * refresh_sell_delta

        buy_err = obs_buy_delta - pred_buy_delta
        sell_err = obs_sell_delta - pred_sell_delta
        market_buy_err = obs_buy_delta - market_buy_delta
        market_sell_err = obs_sell_delta - market_sell_delta
        refresh_buy_err = obs_buy_delta - refresh_buy_delta
        refresh_sell_err = obs_sell_delta - refresh_sell_delta

        log_market = (
            math.log(q_market)
            + normal_logpdf(market_buy_err, market_buy_var)
            + normal_logpdf(market_sell_err, market_sell_var)
        )
        if refresh_distribution == "two_gaussian":
            refresh_buy_log = two_gaussian_logpdf(
                refresh_buy_err,
                refresh_buy_var,
                refresh_jump_buy_var,
                refresh_jump_prob,
            )
            refresh_sell_log = two_gaussian_logpdf(
                refresh_sell_err,
                refresh_sell_var,
                refresh_jump_sell_var,
                refresh_jump_prob,
            )
        else:
            refresh_buy_log = refresh_logpdf(
                refresh_buy_err,
                refresh_buy_var,
                refresh_distribution,
                refresh_student_df,
            )
            refresh_sell_log = refresh_logpdf(
                refresh_sell_err,
                refresh_sell_var,
                refresh_distribution,
                refresh_student_df,
            )
        log_refresh = math.log(q_refresh) + refresh_buy_log + refresh_sell_log
        log_norm = logaddexp(log_market, log_refresh)
        market_resp = math.exp(log_market - log_norm)
        refresh_resp = 1.0 - market_resp

        if seen >= min_rows:
            buy_sq.append(buy_err * buy_err)
            sell_sq.append(sell_err * sell_err)
            buy_zero_sq.append(obs_buy_delta * obs_buy_delta)
            sell_zero_sq.append(obs_sell_delta * obs_sell_delta)
            buy_mech_sq.append((obs_buy_delta - mech_buy_delta) ** 2)
            sell_mech_sq.append((obs_sell_delta - mech_sell_delta) ** 2)
            if refresh_distribution == "two_gaussian":
                buy_refresh_log = two_gaussian_logpdf(
                    refresh_buy_err,
                    refresh_buy_var,
                    refresh_jump_buy_var,
                    refresh_jump_prob,
                )
                sell_refresh_log = two_gaussian_logpdf(
                    refresh_sell_err,
                    refresh_sell_var,
                    refresh_jump_sell_var,
                    refresh_jump_prob,
                )
            else:
                buy_refresh_log = refresh_logpdf(
                    refresh_buy_err,
                    refresh_buy_var,
                    refresh_distribution,
                    refresh_student_df,
                )
                sell_refresh_log = refresh_logpdf(
                    refresh_sell_err,
                    refresh_sell_var,
                    refresh_distribution,
                    refresh_student_df,
                )
            buy_mix_log = logaddexp(
                math.log(q_market) + normal_logpdf(market_buy_err, market_buy_var),
                math.log(q_refresh) + buy_refresh_log,
            )
            sell_mix_log = logaddexp(
                math.log(q_market) + normal_logpdf(market_sell_err, market_sell_var),
                math.log(q_refresh) + sell_refresh_log,
            )
            buy_nll.append(-buy_mix_log)
            sell_nll.append(-sell_mix_log)
            buy_zero_nll.append(gaussian_nll(obs_buy_delta, max(zero_buy_var, eps)))
            sell_zero_nll.append(gaussian_nll(obs_sell_delta, max(zero_sell_var, eps)))

        if gate_kind == "binned":
            gate_bin_probs[gate_bin_idx] = min(
                0.999,
                max(0.001, lam * gate_bin_probs[gate_bin_idx] + alpha * market_resp),
            )
        else:
            lr = max(0.0, float(gate_learning_rate))
            gate_beta = gate_beta + lr * (market_resp - q_market) * gate_x
            gate_beta = np.clip(gate_beta, -20.0, 20.0)

        for name in component_names:
            target = obs_components[name]
            x = np.array([1.0, mech_components[name]], dtype=np.float64)
            market_beta[name], market_p[name] = weighted_rls_update(
                market_beta[name],
                market_p[name],
                x,
                target,
                weight=market_resp,
                lam=lam,
            )
            refresh_mean[name] = refresh_mean[name] + alpha * refresh_resp * (target - refresh_mean[name])

        market_buy_var = (1.0 - alpha * market_resp) * market_buy_var + alpha * market_resp * (market_buy_err * market_buy_err)
        market_sell_var = (1.0 - alpha * market_resp) * market_sell_var + alpha * market_resp * (market_sell_err * market_sell_err)
        if refresh_distribution == "two_gaussian":
            jump_buy_resp = two_gaussian_jump_resp(
                refresh_buy_err,
                refresh_buy_var,
                refresh_jump_buy_var,
                refresh_jump_prob,
            )
            jump_sell_resp = two_gaussian_jump_resp(
                refresh_sell_err,
                refresh_sell_var,
                refresh_jump_sell_var,
                refresh_jump_prob,
            )
            avg_jump_resp = 0.5 * (jump_buy_resp + jump_sell_resp)
            refresh_jump_prob = min(
                0.999,
                max(0.001, refresh_jump_prob + alpha * refresh_resp * (avg_jump_resp - refresh_jump_prob)),
            )
            quiet_buy_weight = refresh_resp * (1.0 - jump_buy_resp)
            quiet_sell_weight = refresh_resp * (1.0 - jump_sell_resp)
            jump_buy_weight = refresh_resp * jump_buy_resp
            jump_sell_weight = refresh_resp * jump_sell_resp
            refresh_buy_var = (1.0 - alpha * quiet_buy_weight) * refresh_buy_var + alpha * quiet_buy_weight * (
                refresh_buy_err * refresh_buy_err
            )
            refresh_sell_var = (1.0 - alpha * quiet_sell_weight) * refresh_sell_var + alpha * quiet_sell_weight * (
                refresh_sell_err * refresh_sell_err
            )
            refresh_jump_buy_var = (
                (1.0 - alpha * jump_buy_weight) * refresh_jump_buy_var
                + alpha * jump_buy_weight * (refresh_buy_err * refresh_buy_err)
            )
            refresh_jump_sell_var = (
                (1.0 - alpha * jump_sell_weight) * refresh_jump_sell_var
                + alpha * jump_sell_weight * (refresh_sell_err * refresh_sell_err)
            )
        else:
            refresh_buy_var = (1.0 - alpha * refresh_resp) * refresh_buy_var + alpha * refresh_resp * (refresh_buy_err * refresh_buy_err)
            refresh_sell_var = (1.0 - alpha * refresh_resp) * refresh_sell_var + alpha * refresh_resp * (refresh_sell_err * refresh_sell_err)
        zero_buy_var = lam * zero_buy_var + alpha * (obs_buy_delta * obs_buy_delta)
        zero_sell_var = lam * zero_sell_var + alpha * (obs_sell_delta * obs_sell_delta)
        seen += 1

    if not buy_sq:
        return None
    return DecomposedHurdleEvaluation(
        n_scored=len(buy_sq),
        active_fraction=market_prob_sum / max(1, seen),
        buy_rmse=math.sqrt(float(np.mean(buy_sq))),
        buy_nll=float(np.mean(buy_nll)),
        buy_zero_rmse=math.sqrt(float(np.mean(buy_zero_sq))),
        buy_zero_nll=float(np.mean(buy_zero_nll)),
        buy_mechanical_rmse=math.sqrt(float(np.mean(buy_mech_sq))),
        sell_rmse=math.sqrt(float(np.mean(sell_sq))),
        sell_nll=float(np.mean(sell_nll)),
        sell_zero_rmse=math.sqrt(float(np.mean(sell_zero_sq))),
        sell_zero_nll=float(np.mean(sell_zero_nll)),
        sell_mechanical_rmse=math.sqrt(float(np.mean(sell_mech_sq))),
    )


def fit_vwap_residual_model(
    rows: Sequence[MarketRow],
    *,
    side: str,
    lags: int,
    min_rows: int,
    vwap_depth: float,
) -> FitResult | None:
    design = build_vwap_residual_design_matrix(
        rows,
        side=side,
        lags=lags,
        min_rows=min_rows,
        vwap_depth=vwap_depth,
    )
    if design is None:
        return None
    y, x_without_intercept, names, y_name = design
    x = np.column_stack([np.ones(y.size), x_without_intercept])
    beta = np.linalg.pinv(x) @ y
    residuals = y - x @ beta
    sse = float(residuals @ residuals)
    sigma2_mle = sse / y.size
    rmse = math.sqrt(sigma2_mle)
    centered = y - float(np.mean(y))
    sst = float(centered @ centered)
    r2 = 1.0 - sse / sst if sst > 0 else float("nan")
    dof = max(1, y.size - x.shape[1])
    sigma2_unbiased = sse / dof
    xtx_inv = np.linalg.pinv(x.T @ x)
    stderr = np.sqrt(np.maximum(0.0, np.diag(xtx_inv) * sigma2_unbiased))
    t_stats = np.divide(beta, stderr, out=np.full_like(beta, np.nan), where=stderr > 0)
    corr = [pearson(x_without_intercept[:, i], y) for i in range(x_without_intercept.shape[1])]
    return FitResult(
        n=int(y.size),
        y_name=y_name,
        feature_names=names,
        beta=beta,
        stderr=stderr,
        t_stats=t_stats,
        sigma2_mle=sigma2_mle,
        rmse=rmse,
        r2=r2,
        corr=corr,
    )


def train_test_vwap_residual_model(
    rows: Sequence[MarketRow],
    *,
    side: str,
    lags: int,
    min_rows: int,
    train_ratio: float,
    vwap_depth: float,
) -> TrainTestResult | None:
    design = build_vwap_residual_design_matrix(
        rows,
        side=side,
        lags=lags,
        min_rows=min_rows,
        vwap_depth=vwap_depth,
    )
    if design is None:
        return None
    y, x_without_intercept, _names, _y_name = design
    x = np.column_stack([np.ones(y.size), x_without_intercept])
    ratio = min(0.95, max(0.05, float(train_ratio)))
    split_idx = int(math.floor(y.size * ratio))
    if split_idx < min_rows or y.size - split_idx < max(5, min_rows // 4):
        return None

    x_train = x[:split_idx, :]
    y_train = y[:split_idx]
    x_test = x[split_idx:, :]
    y_test = y[split_idx:]
    beta = np.linalg.pinv(x_train) @ y_train
    baseline = float(np.mean(y_train))
    return TrainTestResult(
        train=evaluate_predictions(y_train, x_train @ beta, baseline),
        test=evaluate_predictions(y_test, x_test @ beta, baseline),
        split_ratio=ratio,
    )


def evaluate_vwap_full_model(
    rows: Sequence[MarketRow],
    *,
    side: str,
    lags: int,
    min_rows: int,
    train_ratio: float,
    vwap_depth: float,
) -> VwapFullEvaluation | None:
    design = build_vwap_residual_design(
        rows,
        side=side,
        lags=lags,
        min_rows=min_rows,
        vwap_depth=vwap_depth,
    )
    if design is None:
        return None
    x = np.column_stack([np.ones(design.residual.size), design.x_without_intercept])
    ratio = min(0.95, max(0.05, float(train_ratio)))
    split_idx = int(math.floor(design.residual.size * ratio))
    if split_idx < min_rows or design.residual.size - split_idx < max(5, min_rows // 4):
        return None

    x_train = x[:split_idx, :]
    y_train = design.residual[:split_idx]
    x_test = x[split_idx:, :]
    residual_test = design.residual[split_idx:]
    observed_delta_test = design.observed_delta[split_idx:]
    mechanical_delta_test = design.mechanical_delta[split_idx:]

    beta = np.linalg.pinv(x_train) @ y_train
    predicted_residual = x_test @ beta
    full_delta_prediction = mechanical_delta_test + predicted_residual

    full_errors = observed_delta_test - full_delta_prediction
    mechanical_errors = observed_delta_test - mechanical_delta_test
    mean_baseline = float(np.mean(design.observed_delta[:split_idx]))
    mean_errors = observed_delta_test - mean_baseline
    full_sse = float(full_errors @ full_errors)
    mechanical_sse = float(mechanical_errors @ mechanical_errors)
    mean_sse = float(mean_errors @ mean_errors)
    centered = observed_delta_test - float(np.mean(observed_delta_test))
    sst = float(centered @ centered)
    return VwapFullEvaluation(
        n=int(observed_delta_test.size),
        full_rmse=math.sqrt(full_sse / observed_delta_test.size),
        full_r2=1.0 - full_sse / sst if sst > 0 else float("nan"),
        mechanical_rmse=math.sqrt(mechanical_sse / observed_delta_test.size),
        mechanical_r2=1.0 - mechanical_sse / sst if sst > 0 else float("nan"),
        mean_baseline_rmse=math.sqrt(mean_sse / observed_delta_test.size),
    )


def fit_vwap_current_step_model(
    rows: Sequence[MarketRow],
    *,
    side: str,
    lags: int,
    min_rows: int,
    vwap_depth: float,
) -> FitResult | None:
    design = build_vwap_current_step_design(
        rows,
        side=side,
        lags=lags,
        min_rows=min_rows,
        vwap_depth=vwap_depth,
    )
    if design is None:
        return None
    y, x_without_intercept, names, y_name, _mechanical_delta = design
    x = np.column_stack([np.ones(y.size), x_without_intercept])
    beta = np.linalg.pinv(x) @ y
    residuals = y - x @ beta
    sse = float(residuals @ residuals)
    sigma2_mle = sse / y.size
    rmse = math.sqrt(sigma2_mle)
    centered = y - float(np.mean(y))
    sst = float(centered @ centered)
    r2 = 1.0 - sse / sst if sst > 0 else float("nan")
    dof = max(1, y.size - x.shape[1])
    sigma2_unbiased = sse / dof
    xtx_inv = np.linalg.pinv(x.T @ x)
    stderr = np.sqrt(np.maximum(0.0, np.diag(xtx_inv) * sigma2_unbiased))
    t_stats = np.divide(beta, stderr, out=np.full_like(beta, np.nan), where=stderr > 0)
    corr = [pearson(x_without_intercept[:, i], y) for i in range(x_without_intercept.shape[1])]
    return FitResult(
        n=int(y.size),
        y_name=y_name,
        feature_names=names,
        beta=beta,
        stderr=stderr,
        t_stats=t_stats,
        sigma2_mle=sigma2_mle,
        rmse=rmse,
        r2=r2,
        corr=corr,
    )


def train_test_vwap_current_step_model(
    rows: Sequence[MarketRow],
    *,
    side: str,
    lags: int,
    min_rows: int,
    train_ratio: float,
    vwap_depth: float,
) -> tuple[TrainTestResult, EvaluationResult] | None:
    design = build_vwap_current_step_design(
        rows,
        side=side,
        lags=lags,
        min_rows=min_rows,
        vwap_depth=vwap_depth,
    )
    if design is None:
        return None
    y, x_without_intercept, _names, _y_name, mechanical_delta = design
    x = np.column_stack([np.ones(y.size), x_without_intercept])
    ratio = min(0.95, max(0.05, float(train_ratio)))
    split_idx = int(math.floor(y.size * ratio))
    if split_idx < min_rows or y.size - split_idx < max(5, min_rows // 4):
        return None

    x_train = x[:split_idx, :]
    y_train = y[:split_idx]
    x_test = x[split_idx:, :]
    y_test = y[split_idx:]
    beta = np.linalg.pinv(x_train) @ y_train
    baseline = float(np.mean(y_train))
    model_eval = TrainTestResult(
        train=evaluate_predictions(y_train, x_train @ beta, baseline),
        test=evaluate_predictions(y_test, x_test @ beta, baseline),
        split_ratio=ratio,
    )
    mechanical_eval = evaluate_predictions(
        y_test,
        mechanical_delta[split_idx:],
        baseline,
    )
    return model_eval, mechanical_eval


def fit_vwap_book_current_step_model(
    rows: Sequence[MarketRow],
    *,
    side: str,
    min_rows: int,
    vwap_depth: float,
) -> FitResult | None:
    design = build_vwap_book_current_step_design(
        rows,
        side=side,
        min_rows=min_rows,
        vwap_depth=vwap_depth,
    )
    if design is None:
        return None
    y, x_without_intercept, names, y_name = design
    x = np.column_stack([np.ones(y.size), x_without_intercept])
    beta = np.linalg.pinv(x) @ y
    residuals = y - x @ beta
    sse = float(residuals @ residuals)
    sigma2_mle = sse / y.size
    rmse = math.sqrt(sigma2_mle)
    centered = y - float(np.mean(y))
    sst = float(centered @ centered)
    r2 = 1.0 - sse / sst if sst > 0 else float("nan")
    dof = max(1, y.size - x.shape[1])
    sigma2_unbiased = sse / dof
    xtx_inv = np.linalg.pinv(x.T @ x)
    stderr = np.sqrt(np.maximum(0.0, np.diag(xtx_inv) * sigma2_unbiased))
    t_stats = np.divide(beta, stderr, out=np.full_like(beta, np.nan), where=stderr > 0)
    corr = [pearson(x_without_intercept[:, i], y) for i in range(x_without_intercept.shape[1])]
    return FitResult(
        n=int(y.size),
        y_name=y_name,
        feature_names=names,
        beta=beta,
        stderr=stderr,
        t_stats=t_stats,
        sigma2_mle=sigma2_mle,
        rmse=rmse,
        r2=r2,
        corr=corr,
    )


def train_test_vwap_book_current_step_model(
    rows: Sequence[MarketRow],
    *,
    side: str,
    min_rows: int,
    train_ratio: float,
    vwap_depth: float,
) -> TrainTestResult | None:
    design = build_vwap_book_current_step_design(
        rows,
        side=side,
        min_rows=min_rows,
        vwap_depth=vwap_depth,
    )
    if design is None:
        return None
    y, x_without_intercept, _names, _y_name = design
    x = np.column_stack([np.ones(y.size), x_without_intercept])
    ratio = min(0.95, max(0.05, float(train_ratio)))
    split_idx = int(math.floor(y.size * ratio))
    if split_idx < min_rows or y.size - split_idx < max(5, min_rows // 4):
        return None

    x_train = x[:split_idx, :]
    y_train = y[:split_idx]
    x_test = x[split_idx:, :]
    y_test = y[split_idx:]
    beta = np.linalg.pinv(x_train) @ y_train
    baseline = float(np.mean(y_train))
    return TrainTestResult(
        train=evaluate_predictions(y_train, x_train @ beta, baseline),
        test=evaluate_predictions(y_test, x_test @ beta, baseline),
        split_ratio=ratio,
    )


def fit_vwap_complete_current_step_model(
    rows: Sequence[MarketRow],
    *,
    side: str,
    lags: int,
    min_rows: int,
    vwap_depth: float,
    include_depth_ratios: bool,
) -> FitResult | None:
    design = build_vwap_complete_current_step_design(
        rows,
        side=side,
        lags=lags,
        min_rows=min_rows,
        vwap_depth=vwap_depth,
        include_depth_ratios=include_depth_ratios,
    )
    if design is None:
        return None
    y, x_without_intercept, names, y_name, _mechanical_delta = design
    x = np.column_stack([np.ones(y.size), x_without_intercept])
    beta = np.linalg.pinv(x) @ y
    residuals = y - x @ beta
    sse = float(residuals @ residuals)
    sigma2_mle = sse / y.size
    rmse = math.sqrt(sigma2_mle)
    centered = y - float(np.mean(y))
    sst = float(centered @ centered)
    r2 = 1.0 - sse / sst if sst > 0 else float("nan")
    dof = max(1, y.size - x.shape[1])
    sigma2_unbiased = sse / dof
    xtx_inv = np.linalg.pinv(x.T @ x)
    stderr = np.sqrt(np.maximum(0.0, np.diag(xtx_inv) * sigma2_unbiased))
    t_stats = np.divide(beta, stderr, out=np.full_like(beta, np.nan), where=stderr > 0)
    corr = [pearson(x_without_intercept[:, i], y) for i in range(x_without_intercept.shape[1])]
    return FitResult(
        n=int(y.size),
        y_name=y_name,
        feature_names=names,
        beta=beta,
        stderr=stderr,
        t_stats=t_stats,
        sigma2_mle=sigma2_mle,
        rmse=rmse,
        r2=r2,
        corr=corr,
    )


def train_test_vwap_complete_current_step_model(
    rows: Sequence[MarketRow],
    *,
    side: str,
    lags: int,
    min_rows: int,
    train_ratio: float,
    vwap_depth: float,
    include_depth_ratios: bool,
) -> tuple[TrainTestResult, EvaluationResult] | None:
    design = build_vwap_complete_current_step_design(
        rows,
        side=side,
        lags=lags,
        min_rows=min_rows,
        vwap_depth=vwap_depth,
        include_depth_ratios=include_depth_ratios,
    )
    if design is None:
        return None
    y, x_without_intercept, _names, _y_name, mechanical_delta = design
    x = np.column_stack([np.ones(y.size), x_without_intercept])
    ratio = min(0.95, max(0.05, float(train_ratio)))
    split_idx = int(math.floor(y.size * ratio))
    if split_idx < min_rows or y.size - split_idx < max(5, min_rows // 4):
        return None

    x_train = x[:split_idx, :]
    y_train = y[:split_idx]
    x_test = x[split_idx:, :]
    y_test = y[split_idx:]
    beta = np.linalg.pinv(x_train) @ y_train
    baseline = float(np.mean(y_train))
    model_eval = TrainTestResult(
        train=evaluate_predictions(y_train, x_train @ beta, baseline),
        test=evaluate_predictions(y_test, x_test @ beta, baseline),
        split_ratio=ratio,
    )
    mechanical_eval = evaluate_predictions(y_test, mechanical_delta[split_idx:], baseline)
    return model_eval, mechanical_eval


def build_design_matrix(
    rows: Sequence[MarketRow],
    *,
    target: str,
    feature_kind: str,
    lags: int,
    min_rows: int,
) -> tuple[np.ndarray, np.ndarray, list[str], str] | None:
    if len(rows) < max(min_rows, lags + 3):
        return None

    mids = np.array([row.mid for row in rows], dtype=np.float64)
    if target == "log_return":
        y_all = np.diff(np.log(mids))
        y_name = "log_return"
    elif target == "delta":
        y_all = np.diff(mids)
        y_name = "delta_mid"
    else:
        raise ValueError(f"unknown target {target}")

    feature_matrix, feature_names = build_lagged_features(rows, feature_kind=feature_kind, lags=lags)
    # y_all[k - 1] is the price move from row k-1 to row k. Use flow observed at row k and its lags.
    start = max(1, lags)
    y = y_all[start - 1 :]
    x_without_intercept = feature_matrix[start:, :]
    valid = np.isfinite(y) & np.all(np.isfinite(x_without_intercept), axis=1)
    y = y[valid]
    x_without_intercept = x_without_intercept[valid]
    if y.size < min_rows:
        return None

    return y, x_without_intercept, ["intercept", *feature_names], y_name


def build_mechanical_residual_design_matrix(
    rows: Sequence[MarketRow],
    *,
    side: str,
    lags: int,
    min_rows: int,
    book_features: bool = False,
    current_step_book_features: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str], str] | None:
    if len(rows) < max(min_rows, lags + 3):
        return None

    residuals = np.full(len(rows), np.nan, dtype=np.float64)
    if side == "bid_from_sell":
        if book_features:
            base, feature_names_base = build_book_feature_base(rows, side=side)
        else:
            base = np.array([[row.sell_volume] for row in rows], dtype=np.float64)
            feature_names_base = ["sell_volume"]
        y_name = "best_bid_mechanical_residual"
        for idx in range(1, len(rows)):
            prev = rows[idx - 1]
            curr = rows[idx]
            mechanical_bid = consume_best_price(prev.bids, curr.sell_volume, side="bid")
            residuals[idx] = curr.best_bid - mechanical_bid
    elif side == "ask_from_buy":
        if book_features:
            base, feature_names_base = build_book_feature_base(rows, side=side)
        else:
            base = np.array([[row.buy_volume] for row in rows], dtype=np.float64)
            feature_names_base = ["buy_volume"]
        y_name = "best_ask_mechanical_residual"
        for idx in range(1, len(rows)):
            prev = rows[idx - 1]
            curr = rows[idx]
            mechanical_ask = consume_best_price(prev.asks, curr.buy_volume, side="ask")
            residuals[idx] = curr.best_ask - mechanical_ask
    else:
        raise ValueError(f"unknown side model {side}")

    feature_matrix, feature_names = build_base_lagged_features(base, feature_names_base, lags=lags)
    start = max(1, lags)
    y = residuals[start:]
    x_without_intercept = feature_matrix[start:, :]
    valid = np.isfinite(y) & np.all(np.isfinite(x_without_intercept), axis=1)
    y = y[valid]
    x_without_intercept = x_without_intercept[valid]
    if y.size < min_rows:
        return None
    return y, x_without_intercept, ["intercept", *feature_names], y_name


def build_side_mechanical_delta_design(
    rows: Sequence[MarketRow],
    *,
    side: str,
    min_rows: int,
) -> tuple[np.ndarray, np.ndarray, list[str], str, np.ndarray] | None:
    if len(rows) < max(min_rows, 3):
        return None

    observed_delta = np.full(len(rows), np.nan, dtype=np.float64)
    mechanical_delta = np.full(len(rows), np.nan, dtype=np.float64)
    for idx in range(1, len(rows)):
        prev = rows[idx - 1]
        curr = rows[idx]
        if side == "bid_from_sell":
            mechanical_bid = consume_best_price(prev.bids, curr.sell_volume, side="bid")
            observed_delta[idx] = curr.best_bid - prev.best_bid
            mechanical_delta[idx] = mechanical_bid - prev.best_bid
            y_name = "current_best_bid_delta_from_mechanical_delta"
        elif side == "ask_from_buy":
            mechanical_ask = consume_best_price(prev.asks, curr.buy_volume, side="ask")
            observed_delta[idx] = curr.best_ask - prev.best_ask
            mechanical_delta[idx] = mechanical_ask - prev.best_ask
            y_name = "current_best_ask_delta_from_mechanical_delta"
        else:
            raise ValueError(f"unknown side model {side}")

    start = 1
    y = observed_delta[start:]
    mechanical = mechanical_delta[start:]
    x_without_intercept = mechanical[:, None]
    valid = np.isfinite(y) & np.isfinite(mechanical)
    y = y[valid]
    mechanical = mechanical[valid]
    x_without_intercept = x_without_intercept[valid]
    if y.size < min_rows:
        return None
    return y, x_without_intercept, ["intercept", "mechanical_delta"], y_name, mechanical


def build_fixed_depth_vwap_mechanical_delta_design(
    rows: Sequence[MarketRow],
    *,
    side: str,
    min_rows: int,
    vwap_depth: float,
    inventory_cash_depth: bool = False,
    flow_spread_features: bool = False,
    spread_features: bool = False,
    current_spread_feature: bool = False,
    current_flow_spread_delta_lag: bool = False,
    current_flow_prev_book_features: bool = False,
    signed_total_spread_features: bool = False,
    current_flow_features: bool = False,
    mechanical_delta_lags: int = 0,
    delta_price_lags: int = 0,
) -> tuple[np.ndarray, np.ndarray, list[str], str, np.ndarray] | None:
    if len(rows) < max(min_rows, 3):
        return None

    fixed_depth = max(0.0, float(vwap_depth))
    if not inventory_cash_depth and fixed_depth <= 1e-12:
        return None
    observed_delta = np.full(len(rows), np.nan, dtype=np.float64)
    mechanical_delta = np.full(len(rows), np.nan, dtype=np.float64)
    extra_features = np.full((len(rows), 8), np.nan, dtype=np.float64)
    for idx in range(1, len(rows)):
        prev = rows[idx - 1]
        curr = rows[idx]
        if side == "buy_vwap":
            depth = cash_depth(prev) if inventory_cash_depth else fixed_depth
            if depth <= 1e-12:
                continue
            prev_vwap = vwap_to_depth(prev.asks, depth, ask_side=True)
            curr_vwap = vwap_to_depth(curr.asks, depth, ask_side=True)
            mechanical_vwap = vwap_to_depth(
                consume_levels(prev.asks, curr.buy_volume, ask_side=True),
                depth,
                ask_side=True,
            )
            y_name = (
                "cash_depth_buy_vwap_delta_from_mechanical_delta"
                if inventory_cash_depth
                else "fixed_depth_buy_vwap_delta_from_mechanical_delta"
            )
        elif side == "sell_vwap":
            depth = inventory_depth(prev) if inventory_cash_depth else fixed_depth
            if depth <= 1e-12:
                continue
            prev_vwap = vwap_to_depth(prev.bids, depth, ask_side=False)
            curr_vwap = vwap_to_depth(curr.bids, depth, ask_side=False)
            mechanical_vwap = vwap_to_depth(
                consume_levels(prev.bids, curr.sell_volume, ask_side=False),
                depth,
                ask_side=False,
            )
            y_name = (
                "inventory_depth_sell_vwap_delta_from_mechanical_delta"
                if inventory_cash_depth
                else "fixed_depth_sell_vwap_delta_from_mechanical_delta"
            )
        else:
            raise ValueError(f"unknown fixed-depth vwap side {side}")

        observed_delta[idx] = curr_vwap - prev_vwap
        mechanical_delta[idx] = mechanical_vwap - prev_vwap
        extra_features[idx, :] = [
            curr.buy_volume,
            curr.sell_volume,
            curr.spread,
            prev.buy_volume,
            prev.sell_volume,
            prev.spread,
            prev.queue_imbalance_5,
            prev.microprice_minus_mid,
        ]

    lag_count = max(0, int(mechanical_delta_lags))
    delta_lag_count = max(0, int(delta_price_lags))
    start = max(1, lag_count + 1, delta_lag_count + 1)
    if current_flow_spread_delta_lag:
        start = max(start, 2)
    y = observed_delta[start:]
    mechanical = mechanical_delta[start:]
    lagged_mechanical_blocks = [mechanical]
    lagged_mechanical_names = ["mechanical_delta_t"]
    for lag in range(1, lag_count + 1):
        lagged_mechanical_blocks.append(mechanical_delta[start - lag : len(rows) - lag])
        lagged_mechanical_names.append(f"mechanical_delta_t-{lag}")
    lagged_delta_blocks: list[np.ndarray] = []
    lagged_delta_names: list[str] = []
    for lag in range(1, delta_lag_count + 1):
        lagged_delta_blocks.append(observed_delta[start - lag : len(rows) - lag])
        lagged_delta_names.append(f"delta_price_t-{lag}")
    if flow_spread_features:
        extras = extra_features[start:, :]
        x_without_intercept = np.column_stack([*lagged_mechanical_blocks, *lagged_delta_blocks, extras])
        valid = (
            np.isfinite(y)
            & np.all(np.isfinite(np.column_stack(lagged_mechanical_blocks)), axis=1)
            & (not lagged_delta_blocks or np.all(np.isfinite(np.column_stack(lagged_delta_blocks)), axis=1))
            & np.all(np.isfinite(extras), axis=1)
        )
        names = [
            "intercept",
            *lagged_mechanical_names,
            *lagged_delta_names,
            "buy_volume_t",
            "sell_volume_t",
            "spread_t",
            "buy_volume_t-1",
            "sell_volume_t-1",
            "spread_t-1",
        ]
    elif spread_features:
        extras = extra_features[start:, [2, 5]]
        x_without_intercept = np.column_stack([*lagged_mechanical_blocks, *lagged_delta_blocks, extras])
        valid = (
            np.isfinite(y)
            & np.all(np.isfinite(np.column_stack(lagged_mechanical_blocks)), axis=1)
            & (not lagged_delta_blocks or np.all(np.isfinite(np.column_stack(lagged_delta_blocks)), axis=1))
            & np.all(np.isfinite(extras), axis=1)
        )
        names = [
            "intercept",
            *lagged_mechanical_names,
            *lagged_delta_names,
            "spread_t",
            "spread_t-1",
        ]
    elif current_spread_feature:
        extras = extra_features[start:, [2]]
        x_without_intercept = np.column_stack([*lagged_mechanical_blocks, *lagged_delta_blocks, extras])
        valid = (
            np.isfinite(y)
            & np.all(np.isfinite(np.column_stack(lagged_mechanical_blocks)), axis=1)
            & (not lagged_delta_blocks or np.all(np.isfinite(np.column_stack(lagged_delta_blocks)), axis=1))
            & np.all(np.isfinite(extras), axis=1)
        )
        names = [
            "intercept",
            *lagged_mechanical_names,
            *lagged_delta_names,
            "spread_t",
        ]
    elif current_flow_spread_delta_lag:
        extras = np.column_stack(
            [
                extra_features[start:, 0],
                extra_features[start:, 1],
                extra_features[start:, 2],
                observed_delta[start - 1 : len(rows) - 1],
            ]
        )
        x_without_intercept = np.column_stack([*lagged_mechanical_blocks, *lagged_delta_blocks, extras])
        valid = (
            np.isfinite(y)
            & np.all(np.isfinite(np.column_stack(lagged_mechanical_blocks)), axis=1)
            & (not lagged_delta_blocks or np.all(np.isfinite(np.column_stack(lagged_delta_blocks)), axis=1))
            & np.all(np.isfinite(extras), axis=1)
        )
        names = [
            "intercept",
            *lagged_mechanical_names,
            *lagged_delta_names,
            "buy_volume_t",
            "sell_volume_t",
            "spread_t",
            "delta_price_t-1",
        ]
    elif current_flow_prev_book_features:
        extras = extra_features[start:, [0, 1, 5, 6, 7]]
        x_without_intercept = np.column_stack([*lagged_mechanical_blocks, *lagged_delta_blocks, extras])
        valid = (
            np.isfinite(y)
            & np.all(np.isfinite(np.column_stack(lagged_mechanical_blocks)), axis=1)
            & (not lagged_delta_blocks or np.all(np.isfinite(np.column_stack(lagged_delta_blocks)), axis=1))
            & np.all(np.isfinite(extras), axis=1)
        )
        names = [
            "intercept",
            *lagged_mechanical_names,
            *lagged_delta_names,
            "buy_volume_t",
            "sell_volume_t",
            "spread_t-1",
            "queue_imbalance_5_t-1",
            "microprice_minus_mid_t-1",
        ]
    elif signed_total_spread_features:
        buy = extra_features[start:, 0]
        sell = extra_features[start:, 1]
        extras = np.column_stack([buy - sell, buy + sell, extra_features[start:, 5]])
        x_without_intercept = np.column_stack([*lagged_mechanical_blocks, *lagged_delta_blocks, extras])
        valid = (
            np.isfinite(y)
            & np.all(np.isfinite(np.column_stack(lagged_mechanical_blocks)), axis=1)
            & (not lagged_delta_blocks or np.all(np.isfinite(np.column_stack(lagged_delta_blocks)), axis=1))
            & np.all(np.isfinite(extras), axis=1)
        )
        names = [
            "intercept",
            *lagged_mechanical_names,
            *lagged_delta_names,
            "signed_flow_t",
            "total_flow_t",
            "spread_t-1",
        ]
    elif current_flow_features:
        extras = extra_features[start:, [0, 1, 3, 4]]
        x_without_intercept = np.column_stack([*lagged_mechanical_blocks, *lagged_delta_blocks, extras])
        valid = (
            np.isfinite(y)
            & np.all(np.isfinite(np.column_stack(lagged_mechanical_blocks)), axis=1)
            & (not lagged_delta_blocks or np.all(np.isfinite(np.column_stack(lagged_delta_blocks)), axis=1))
            & np.all(np.isfinite(extras), axis=1)
        )
        names = [
            "intercept",
            *lagged_mechanical_names,
            *lagged_delta_names,
            "buy_volume_t",
            "sell_volume_t",
            "buy_volume_t-1",
            "sell_volume_t-1",
        ]
    else:
        x_without_intercept = np.column_stack([*lagged_mechanical_blocks, *lagged_delta_blocks])
        valid = np.isfinite(y) & np.all(np.isfinite(x_without_intercept), axis=1)
        names = ["intercept", *lagged_mechanical_names, *lagged_delta_names]
    y = y[valid]
    mechanical = mechanical[valid]
    x_without_intercept = x_without_intercept[valid]
    if y.size < min_rows:
        return None
    return y, x_without_intercept, names, y_name, mechanical


def build_cross_venue_design(
    target_rows: Sequence[MarketRow],
    source_rows: Sequence[MarketRow],
    *,
    target_kind: str,
    min_rows: int,
    vwap_depth: float,
    mechanical_delta_lags: int = 1,
) -> tuple[np.ndarray, np.ndarray, list[str], str, np.ndarray] | None:
    source_by_ts = {row.ts: row for row in source_rows}
    aligned: list[tuple[MarketRow, MarketRow]] = [
        (target_row, source_by_ts[target_row.ts])
        for target_row in target_rows
        if target_row.ts in source_by_ts
    ]
    if len(aligned) < max(min_rows, 4):
        return None

    depth = max(0.0, float(vwap_depth))
    if depth <= 1e-12:
        return None

    n = len(aligned)
    observed_delta = np.full(n, np.nan, dtype=np.float64)
    target_mechanical_delta = np.full(n, np.nan, dtype=np.float64)
    source_mechanical_delta = np.full(n, np.nan, dtype=np.float64)
    source_extras = np.full((n, 6), np.nan, dtype=np.float64)

    for idx in range(1, n):
        target_prev, source_prev = aligned[idx - 1]
        target_curr, source_curr = aligned[idx]

        if target_kind == "best_bid":
            observed_delta[idx] = target_curr.best_bid - target_prev.best_bid
            target_mechanical_delta[idx] = (
                consume_best_price(target_prev.bids, target_curr.sell_volume, side="bid")
                - target_prev.best_bid
            )
            source_mechanical_delta[idx] = (
                consume_best_price(source_prev.bids, source_curr.sell_volume, side="bid")
                - source_prev.best_bid
            )
            y_name = "wallex_best_bid_delta_from_nobitex_features"
        elif target_kind == "best_ask":
            observed_delta[idx] = target_curr.best_ask - target_prev.best_ask
            target_mechanical_delta[idx] = (
                consume_best_price(target_prev.asks, target_curr.buy_volume, side="ask")
                - target_prev.best_ask
            )
            source_mechanical_delta[idx] = (
                consume_best_price(source_prev.asks, source_curr.buy_volume, side="ask")
                - source_prev.best_ask
            )
            y_name = "wallex_best_ask_delta_from_nobitex_features"
        elif target_kind == "buy_vwap":
            target_prev_vwap = vwap_to_depth(target_prev.asks, depth, ask_side=True)
            target_curr_vwap = vwap_to_depth(target_curr.asks, depth, ask_side=True)
            target_mech_vwap = vwap_to_depth(
                consume_levels(target_prev.asks, target_curr.buy_volume, ask_side=True),
                depth,
                ask_side=True,
            )
            source_prev_vwap = vwap_to_depth(source_prev.asks, depth, ask_side=True)
            source_mech_vwap = vwap_to_depth(
                consume_levels(source_prev.asks, source_curr.buy_volume, ask_side=True),
                depth,
                ask_side=True,
            )
            observed_delta[idx] = target_curr_vwap - target_prev_vwap
            target_mechanical_delta[idx] = target_mech_vwap - target_prev_vwap
            source_mechanical_delta[idx] = source_mech_vwap - source_prev_vwap
            y_name = "wallex_buy_vwap_delta_from_nobitex_features"
        elif target_kind == "sell_vwap":
            target_prev_vwap = vwap_to_depth(target_prev.bids, depth, ask_side=False)
            target_curr_vwap = vwap_to_depth(target_curr.bids, depth, ask_side=False)
            target_mech_vwap = vwap_to_depth(
                consume_levels(target_prev.bids, target_curr.sell_volume, ask_side=False),
                depth,
                ask_side=False,
            )
            source_prev_vwap = vwap_to_depth(source_prev.bids, depth, ask_side=False)
            source_mech_vwap = vwap_to_depth(
                consume_levels(source_prev.bids, source_curr.sell_volume, ask_side=False),
                depth,
                ask_side=False,
            )
            observed_delta[idx] = target_curr_vwap - target_prev_vwap
            target_mechanical_delta[idx] = target_mech_vwap - target_prev_vwap
            source_mechanical_delta[idx] = source_mech_vwap - source_prev_vwap
            y_name = "wallex_sell_vwap_delta_from_nobitex_features"
        else:
            raise ValueError(f"unknown cross-venue target kind {target_kind}")

        source_extras[idx, :] = [
            source_curr.buy_volume,
            source_curr.sell_volume,
            source_curr.spread,
            source_prev.buy_volume,
            source_prev.sell_volume,
            source_prev.spread,
        ]

    lag_count = max(0, int(mechanical_delta_lags))
    start = max(1, lag_count + 1)
    y = observed_delta[start:]
    mechanical = target_mechanical_delta[start:]

    source_mech_blocks = [source_mechanical_delta[start:]]
    source_mech_names = ["nobitex_mechanical_delta_t"]
    for lag in range(1, lag_count + 1):
        source_mech_blocks.append(source_mechanical_delta[start - lag : n - lag])
        source_mech_names.append(f"nobitex_mechanical_delta_t-{lag}")

    extras = source_extras[start:, :]
    x_without_intercept = np.column_stack([*source_mech_blocks, extras])
    valid = (
        np.isfinite(y)
        & np.isfinite(mechanical)
        & np.all(np.isfinite(x_without_intercept), axis=1)
    )
    y = y[valid]
    mechanical = mechanical[valid]
    x_without_intercept = x_without_intercept[valid]
    names = [
        "intercept",
        *source_mech_names,
        "nobitex_buy_volume_t",
        "nobitex_sell_volume_t",
        "nobitex_spread_t",
        "nobitex_buy_volume_t-1",
        "nobitex_sell_volume_t-1",
        "nobitex_spread_t-1",
    ]
    if y.size < min_rows:
        return None
    return y, x_without_intercept, names, y_name, mechanical


def price_value_for_kind(row: MarketRow, kind: str, *, vwap_depth: float) -> float:
    if kind == "mid":
        return row.mid
    if kind == "best_bid":
        return row.best_bid
    if kind == "best_ask":
        return row.best_ask
    if kind == "buy_vwap":
        return vwap_to_depth(row.asks, vwap_depth, ask_side=True)
    if kind == "sell_vwap":
        return vwap_to_depth(row.bids, vwap_depth, ask_side=False)
    raise ValueError(f"unknown price kind {kind}")


def build_cross_venue_delta_lag_design(
    target_rows: Sequence[MarketRow],
    source_rows: Sequence[MarketRow],
    *,
    target_kind: str,
    source_kind: str,
    lags: int,
    include_current: bool,
    min_rows: int,
    vwap_depth: float,
) -> tuple[np.ndarray, np.ndarray, list[str], str] | None:
    source_by_ts = {row.ts: row for row in source_rows}
    aligned: list[tuple[MarketRow, MarketRow]] = [
        (target_row, source_by_ts[target_row.ts])
        for target_row in target_rows
        if target_row.ts in source_by_ts
    ]
    if len(aligned) < max(min_rows, 4):
        return None

    lag_count = max(0, int(lags))
    n = len(aligned)
    target_delta = np.full(n, np.nan, dtype=np.float64)
    source_delta = np.full(n, np.nan, dtype=np.float64)
    for idx in range(1, n):
        target_prev, source_prev = aligned[idx - 1]
        target_curr, source_curr = aligned[idx]
        target_delta[idx] = price_value_for_kind(target_curr, target_kind, vwap_depth=vwap_depth) - price_value_for_kind(
            target_prev,
            target_kind,
            vwap_depth=vwap_depth,
        )
        source_delta[idx] = price_value_for_kind(source_curr, source_kind, vwap_depth=vwap_depth) - price_value_for_kind(
            source_prev,
            source_kind,
            vwap_depth=vwap_depth,
        )

    start = max(1, lag_count + 1)
    y = target_delta[start:]
    source_blocks = []
    feature_names = ["intercept"]
    if include_current:
        source_blocks.append(source_delta[start:])
        feature_names.append(f"nobitex_{source_kind}_delta_t")
    first_lag = 1
    for lag in range(first_lag, lag_count + 1):
        source_blocks.append(source_delta[start - lag : n - lag])
        feature_names.append(f"nobitex_{source_kind}_delta_t-{lag}")
    if not source_blocks:
        return None

    x_without_intercept = np.column_stack(source_blocks)
    valid = np.isfinite(y) & np.all(np.isfinite(x_without_intercept), axis=1)
    y = y[valid]
    x_without_intercept = x_without_intercept[valid]
    y_name = f"wallex_{target_kind}_delta_from_nobitex_{source_kind}_delta_lags"
    if y.size < min_rows:
        return None
    return y, x_without_intercept, feature_names, y_name


def fit_cross_venue_delta_lag_model(
    target_rows: Sequence[MarketRow],
    source_rows: Sequence[MarketRow],
    *,
    target_kind: str,
    source_kind: str,
    lags: int,
    include_current: bool,
    min_rows: int,
    train_ratio: float,
    vwap_depth: float,
) -> tuple[FitResult, EvaluationResult] | None:
    design = build_cross_venue_delta_lag_design(
        target_rows,
        source_rows,
        target_kind=target_kind,
        source_kind=source_kind,
        lags=lags,
        include_current=include_current,
        min_rows=min_rows,
        vwap_depth=vwap_depth,
    )
    if design is None:
        return None
    y, x_without_intercept, names, y_name = design
    x = np.column_stack([np.ones(y.size), x_without_intercept])
    beta = np.linalg.pinv(x) @ y
    residuals = y - x @ beta
    sse = float(residuals @ residuals)
    sigma2_mle = sse / y.size
    rmse = math.sqrt(sigma2_mle)
    centered = y - float(np.mean(y))
    sst = float(centered @ centered)
    r2 = 1.0 - sse / sst if sst > 0 else float("nan")
    dof = max(1, y.size - x.shape[1])
    sigma2_unbiased = sse / dof
    xtx_inv = np.linalg.pinv(x.T @ x)
    stderr = np.sqrt(np.maximum(0.0, np.diag(xtx_inv) * sigma2_unbiased))
    t_stats = np.divide(beta, stderr, out=np.full_like(beta, np.nan), where=stderr > 0)
    corr = [pearson(x_without_intercept[:, i], y) for i in range(x_without_intercept.shape[1])]
    fit = FitResult(
        n=int(y.size),
        y_name=y_name,
        feature_names=names,
        beta=beta,
        stderr=stderr,
        t_stats=t_stats,
        sigma2_mle=sigma2_mle,
        rmse=rmse,
        r2=r2,
        corr=corr,
    )

    split_idx = int(y.size * min(0.95, max(0.05, float(train_ratio))))
    split_idx = min(max(split_idx, x.shape[1]), y.size - 1)
    x_train = x[:split_idx]
    y_train = y[:split_idx]
    x_test = x[split_idx:]
    y_test = y[split_idx:]
    beta_train = np.linalg.pinv(x_train) @ y_train
    pred = x_test @ beta_train
    errors = y_test - pred
    rmse_test = math.sqrt(float(errors @ errors) / y_test.size)
    mean_train = float(np.mean(y_train))
    mean_errors = y_test - mean_train
    zero_errors = y_test
    baseline_rmse = math.sqrt(float(mean_errors @ mean_errors) / y_test.size)
    zero_rmse = math.sqrt(float(zero_errors @ zero_errors) / y_test.size)
    centered_test = y_test - float(np.mean(y_test))
    sst_test = float(centered_test @ centered_test)
    sse_test = float(errors @ errors)
    test_r2 = 1.0 - sse_test / sst_test if sst_test > 0 else float("nan")
    direction_accuracy = float(np.mean(np.sign(pred) == np.sign(y_test)))
    baseline_direction_accuracy = float(np.mean(np.sign(np.full_like(y_test, mean_train)) == np.sign(y_test)))
    zero_direction_accuracy = float(np.mean(np.sign(np.zeros_like(y_test)) == np.sign(y_test)))
    evaluation = EvaluationResult(
        n=int(y_test.size),
        rmse=rmse_test,
        r2=test_r2,
        baseline_rmse=baseline_rmse,
        zero_baseline_rmse=zero_rmse,
        direction_accuracy=direction_accuracy,
        baseline_direction_accuracy=baseline_direction_accuracy,
        zero_baseline_direction_accuracy=zero_direction_accuracy,
    )
    return fit, evaluation


def evaluate_cross_venue_gated_delta_model(
    target_rows: Sequence[MarketRow],
    source_rows: Sequence[MarketRow],
    *,
    target_kind: str,
    source_kind: str,
    gate_percentile: float,
    min_rows: int,
    train_ratio: float,
    vwap_depth: float,
) -> dict[str, float] | None:
    design = build_cross_venue_delta_lag_design(
        target_rows,
        source_rows,
        target_kind=target_kind,
        source_kind=source_kind,
        lags=0,
        include_current=True,
        min_rows=min_rows,
        vwap_depth=vwap_depth,
    )
    if design is None:
        return None
    y, x_without_intercept, _names, _y_name = design
    source_delta = x_without_intercept[:, 0]
    split_idx = int(y.size * min(0.95, max(0.05, float(train_ratio))))
    split_idx = min(max(split_idx, 3), y.size - 1)
    y_train = y[:split_idx]
    x_train_source = source_delta[:split_idx]
    y_test = y[split_idx:]
    x_test_source = source_delta[split_idx:]

    pct = min(100.0, max(0.0, float(gate_percentile)))
    threshold = float(np.percentile(np.abs(x_train_source), pct))
    train_active = np.abs(x_train_source) >= threshold
    test_active = np.abs(x_test_source) >= threshold
    if int(np.sum(train_active)) < 3:
        return None

    x_train_active = np.column_stack([np.ones(np.sum(train_active)), x_train_source[train_active]])
    beta = np.linalg.pinv(x_train_active) @ y_train[train_active]

    pred = np.zeros_like(y_test)
    if np.any(test_active):
        x_test_active = np.column_stack([np.ones(np.sum(test_active)), x_test_source[test_active]])
        pred[test_active] = x_test_active @ beta

    errors = y_test - pred
    zero_errors = y_test
    mean_train = float(np.mean(y_train))
    mean_errors = y_test - mean_train
    sse = float(errors @ errors)
    rmse = math.sqrt(sse / y_test.size)
    zero_rmse = math.sqrt(float(zero_errors @ zero_errors) / y_test.size)
    mean_rmse = math.sqrt(float(mean_errors @ mean_errors) / y_test.size)
    centered_test = y_test - float(np.mean(y_test))
    sst = float(centered_test @ centered_test)
    r2 = 1.0 - sse / sst if sst > 0 else float("nan")
    return {
        "n": float(y.size),
        "test_n": float(y_test.size),
        "gate_percentile": pct,
        "threshold": threshold,
        "train_active_fraction": float(np.mean(train_active)),
        "test_active_fraction": float(np.mean(test_active)),
        "rmse": rmse,
        "zero_rmse": zero_rmse,
        "mean_rmse": mean_rmse,
        "r2": r2,
        "beta0": float(beta[0]),
        "beta1": float(beta[1]),
        "direction_accuracy": float(np.mean(np.sign(pred) == np.sign(y_test))),
    }


def price_and_mechanical_delta(row_prev: MarketRow, row_curr: MarketRow, kind: str, *, vwap_depth: float) -> tuple[float, float]:
    if kind == "mid":
        observed = row_curr.mid - row_prev.mid
        mech_bid = consume_best_price(row_prev.bids, row_curr.sell_volume, side="bid")
        mech_ask = consume_best_price(row_prev.asks, row_curr.buy_volume, side="ask")
        mechanical = 0.5 * (mech_bid + mech_ask) - row_prev.mid
        return observed, mechanical
    if kind == "best_bid":
        observed = row_curr.best_bid - row_prev.best_bid
        mechanical = consume_best_price(row_prev.bids, row_curr.sell_volume, side="bid") - row_prev.best_bid
        return observed, mechanical
    if kind == "best_ask":
        observed = row_curr.best_ask - row_prev.best_ask
        mechanical = consume_best_price(row_prev.asks, row_curr.buy_volume, side="ask") - row_prev.best_ask
        return observed, mechanical
    if kind == "buy_vwap":
        prev_vwap = vwap_to_depth(row_prev.asks, vwap_depth, ask_side=True)
        curr_vwap = vwap_to_depth(row_curr.asks, vwap_depth, ask_side=True)
        mech_vwap = vwap_to_depth(
            consume_levels(row_prev.asks, row_curr.buy_volume, ask_side=True),
            vwap_depth,
            ask_side=True,
        )
        return curr_vwap - prev_vwap, mech_vwap - prev_vwap
    if kind == "sell_vwap":
        prev_vwap = vwap_to_depth(row_prev.bids, vwap_depth, ask_side=False)
        curr_vwap = vwap_to_depth(row_curr.bids, vwap_depth, ask_side=False)
        mech_vwap = vwap_to_depth(
            consume_levels(row_prev.bids, row_curr.sell_volume, ask_side=False),
            vwap_depth,
            ask_side=False,
        )
        return curr_vwap - prev_vwap, mech_vwap - prev_vwap
    raise ValueError(f"unknown price kind {kind}")


def quantile_summary(values: Sequence[float]) -> dict[str, float]:
    arr = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    if arr.size == 0:
        return {"n": 0.0, "mean": float("nan"), "p25": float("nan"), "median": float("nan"), "p75": float("nan")}
    return {
        "n": float(arr.size),
        "mean": float(np.mean(arr)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
    }


def format_quantiles(summary: dict[str, float]) -> str:
    return (
        f"mean={summary['mean']:.6g}, "
        f"p25={summary['p25']:.6g}, "
        f"median={summary['median']:.6g}, "
        f"p75={summary['p75']:.6g}"
    )


def lag1_binary_autocorr(flags: Sequence[bool]) -> float:
    if len(flags) < 3:
        return float("nan")
    x = np.asarray(flags[:-1], dtype=np.float64)
    y = np.asarray(flags[1:], dtype=np.float64)
    return pearson(x, y)


def hour_histogram(rows: Sequence[MarketRow], event_indices: Sequence[int]) -> list[tuple[int, int]]:
    counts: dict[int, int] = {}
    for idx in event_indices:
        hour = rows[idx].ts.hour
        counts[hour] = counts.get(hour, 0) + 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))


def cash_depth(row: MarketRow) -> float:
    mid = row.mid
    if not math.isfinite(row.cash) or not math.isfinite(mid) or mid <= 1e-12:
        return float("nan")
    return max(0.0, row.cash / mid)


def inventory_depth(row: MarketRow) -> float:
    if not math.isfinite(row.inventory):
        return float("nan")
    return max(0.0, row.inventory)


def build_vwap_residual_design_matrix(
    rows: Sequence[MarketRow],
    *,
    side: str,
    lags: int,
    min_rows: int,
    vwap_depth: float,
) -> tuple[np.ndarray, np.ndarray, list[str], str] | None:
    design = build_vwap_residual_design(
        rows,
        side=side,
        lags=lags,
        min_rows=min_rows,
        vwap_depth=vwap_depth,
    )
    if design is None:
        return None
    return design.residual, design.x_without_intercept, design.feature_names, design.y_name


def build_vwap_residual_design(
    rows: Sequence[MarketRow],
    *,
    side: str,
    lags: int,
    min_rows: int,
    vwap_depth: float,
) -> VwapDesign | None:
    if len(rows) < max(min_rows, lags + 3):
        return None

    residuals = np.full(len(rows), np.nan, dtype=np.float64)
    observed_delta = np.full(len(rows), np.nan, dtype=np.float64)
    mechanical_delta = np.full(len(rows), np.nan, dtype=np.float64)
    for idx in range(1, len(rows)):
        prev = rows[idx - 1]
        curr = rows[idx]
        depth_used = comparable_vwap_depth(prev, curr, vwap_depth)
        if depth_used <= 1e-12:
            continue

        curr_buy_vwap = vwap_to_depth(curr.asks, depth_used, ask_side=True)
        curr_sell_vwap = vwap_to_depth(curr.bids, depth_used, ask_side=False)
        ask_after = consume_levels(prev.asks, curr.buy_volume, ask_side=True)
        bid_after = consume_levels(prev.bids, curr.sell_volume, ask_side=False)
        mechanical_buy_vwap = vwap_to_depth(ask_after, depth_used, ask_side=True)
        mechanical_sell_vwap = vwap_to_depth(bid_after, depth_used, ask_side=False)

        if side == "buy_vwap_from_flow":
            prev_buy_vwap = vwap_to_depth(prev.asks, depth_used, ask_side=True)
            observed_delta[idx] = curr_buy_vwap - prev_buy_vwap
            mechanical_delta[idx] = mechanical_buy_vwap - prev_buy_vwap
            residuals[idx] = curr_buy_vwap - mechanical_buy_vwap
        elif side == "sell_vwap_from_flow":
            prev_sell_vwap = vwap_to_depth(prev.bids, depth_used, ask_side=False)
            observed_delta[idx] = curr_sell_vwap - prev_sell_vwap
            mechanical_delta[idx] = mechanical_sell_vwap - prev_sell_vwap
            residuals[idx] = curr_sell_vwap - mechanical_sell_vwap
        else:
            raise ValueError(f"unknown vwap side model {side}")

    base = np.array([[row.buy_volume, row.sell_volume] for row in rows], dtype=np.float64)
    feature_matrix, feature_names = build_base_lagged_features(
        base,
        ["buy_volume", "sell_volume"],
        lags=lags,
    )
    start = max(1, lags)
    y = residuals[start:]
    observed = observed_delta[start:]
    mechanical = mechanical_delta[start:]
    x_without_intercept = feature_matrix[start:, :]
    valid = (
        np.isfinite(y)
        & np.isfinite(observed)
        & np.isfinite(mechanical)
        & np.all(np.isfinite(x_without_intercept), axis=1)
    )
    y = y[valid]
    observed = observed[valid]
    mechanical = mechanical[valid]
    x_without_intercept = x_without_intercept[valid]
    if y.size < min_rows:
        return None
    y_name = "buy_vwap_mechanical_residual" if side == "buy_vwap_from_flow" else "sell_vwap_mechanical_residual"
    return VwapDesign(
        residual=y,
        x_without_intercept=x_without_intercept,
        feature_names=["intercept", *feature_names],
        y_name=y_name,
        observed_delta=observed,
        mechanical_delta=mechanical,
    )


def build_vwap_current_step_design(
    rows: Sequence[MarketRow],
    *,
    side: str,
    lags: int,
    min_rows: int,
    vwap_depth: float,
) -> tuple[np.ndarray, np.ndarray, list[str], str, np.ndarray] | None:
    if len(rows) < max(min_rows, lags + 3):
        return None

    observed_delta = np.full(len(rows), np.nan, dtype=np.float64)
    mechanical_delta = np.full(len(rows), np.nan, dtype=np.float64)
    for idx in range(1, len(rows)):
        prev = rows[idx - 1]
        curr = rows[idx]
        depth_used = comparable_vwap_depth(prev, curr, vwap_depth)
        if depth_used <= 1e-12:
            continue

        prev_buy_vwap = vwap_to_depth(prev.asks, depth_used, ask_side=True)
        prev_sell_vwap = vwap_to_depth(prev.bids, depth_used, ask_side=False)
        curr_buy_vwap = vwap_to_depth(curr.asks, depth_used, ask_side=True)
        curr_sell_vwap = vwap_to_depth(curr.bids, depth_used, ask_side=False)
        mechanical_buy_vwap = vwap_to_depth(
            consume_levels(prev.asks, curr.buy_volume, ask_side=True),
            depth_used,
            ask_side=True,
        )
        mechanical_sell_vwap = vwap_to_depth(
            consume_levels(prev.bids, curr.sell_volume, ask_side=False),
            depth_used,
            ask_side=False,
        )

        if side == "buy_vwap_current":
            observed_delta[idx] = curr_buy_vwap - prev_buy_vwap
            mechanical_delta[idx] = mechanical_buy_vwap - prev_buy_vwap
        elif side == "sell_vwap_current":
            observed_delta[idx] = curr_sell_vwap - prev_sell_vwap
            mechanical_delta[idx] = mechanical_sell_vwap - prev_sell_vwap
        else:
            raise ValueError(f"unknown vwap current side {side}")

    flow_base = np.array([[row.buy_volume, row.sell_volume] for row in rows], dtype=np.float64)
    flow_features, flow_names = build_base_lagged_features(
        flow_base,
        ["buy_volume", "sell_volume"],
        lags=lags,
    )
    feature_matrix = np.column_stack([mechanical_delta, flow_features])
    feature_names = ["mechanical_delta", *flow_names]
    start = max(1, lags)
    y = observed_delta[start:]
    x_without_intercept = feature_matrix[start:, :]
    mechanical = mechanical_delta[start:]
    valid = np.isfinite(y) & np.isfinite(mechanical) & np.all(np.isfinite(x_without_intercept), axis=1)
    y = y[valid]
    mechanical = mechanical[valid]
    x_without_intercept = x_without_intercept[valid]
    if y.size < min_rows:
        return None
    y_name = "current_buy_vwap_delta" if side == "buy_vwap_current" else "current_sell_vwap_delta"
    return y, x_without_intercept, ["intercept", *feature_names], y_name, mechanical


def build_vwap_book_current_step_design(
    rows: Sequence[MarketRow],
    *,
    side: str,
    min_rows: int,
    vwap_depth: float,
) -> tuple[np.ndarray, np.ndarray, list[str], str] | None:
    if len(rows) < max(min_rows, 3):
        return None

    y_values: list[float] = []
    features: list[list[float]] = []
    eps = 1e-9
    for idx in range(1, len(rows)):
        prev = rows[idx - 1]
        curr = rows[idx]
        depth_used = min(
            max(0.0, float(vwap_depth)),
            total_depth(prev.asks),
            total_depth(prev.bids),
            total_depth(curr.asks),
            total_depth(curr.bids),
        )
        if depth_used <= 1e-12:
            continue

        prev_buy_vwap = vwap_to_depth(prev.asks, depth_used, ask_side=True)
        prev_sell_vwap = vwap_to_depth(prev.bids, depth_used, ask_side=False)
        curr_buy_vwap = vwap_to_depth(curr.asks, depth_used, ask_side=True)
        curr_sell_vwap = vwap_to_depth(curr.bids, depth_used, ask_side=False)

        if side == "buy_vwap_book_current":
            y = curr_buy_vwap - prev_buy_vwap
            flow = prev.buy_volume
            depth_1 = prev.best_ask_qty
            depth_5 = prev.ask_depth_5
            y_name = "current_buy_vwap_delta_from_prev_book"
            flow_name = "buy_volume_t-1"
            ratio_1_name = "buy_over_ask_qty1_t-1"
            ratio_5_name = "buy_over_ask_depth5_t-1"
        elif side == "sell_vwap_book_current":
            y = curr_sell_vwap - prev_sell_vwap
            flow = prev.sell_volume
            depth_1 = prev.best_bid_qty
            depth_5 = prev.bid_depth_5
            y_name = "current_sell_vwap_delta_from_prev_book"
            flow_name = "sell_volume_t-1"
            ratio_1_name = "sell_over_bid_qty1_t-1"
            ratio_5_name = "sell_over_bid_depth5_t-1"
        else:
            raise ValueError(f"unknown vwap book current side {side}")

        row = [
            flow,
            flow / max(depth_1, eps),
            flow / max(depth_5, eps),
            prev.spread,
            prev.queue_imbalance_5,
            prev.microprice_minus_mid,
        ]
        if math.isfinite(y) and all(math.isfinite(value) for value in row):
            y_values.append(float(y))
            features.append(row)

    if len(y_values) < min_rows:
        return None
    feature_names = [
        "intercept",
        flow_name,
        ratio_1_name,
        ratio_5_name,
        "spread_t-1",
        "queue_imbalance_5_t-1",
        "microprice_minus_mid_t-1",
    ]
    return (
        np.array(y_values, dtype=np.float64),
        np.array(features, dtype=np.float64),
        feature_names,
        y_name,
    )


def build_vwap_complete_current_step_design(
    rows: Sequence[MarketRow],
    *,
    side: str,
    lags: int,
    min_rows: int,
    vwap_depth: float,
    include_depth_ratios: bool,
) -> tuple[np.ndarray, np.ndarray, list[str], str, np.ndarray] | None:
    if len(rows) < max(min_rows, lags + 3):
        return None

    observed_delta = np.full(len(rows), np.nan, dtype=np.float64)
    mechanical_delta = np.full(len(rows), np.nan, dtype=np.float64)
    for idx in range(1, len(rows)):
        prev = rows[idx - 1]
        curr = rows[idx]
        depth_used = comparable_vwap_depth(prev, curr, vwap_depth)
        if depth_used <= 1e-12:
            continue

        prev_buy_vwap = vwap_to_depth(prev.asks, depth_used, ask_side=True)
        prev_sell_vwap = vwap_to_depth(prev.bids, depth_used, ask_side=False)
        curr_buy_vwap = vwap_to_depth(curr.asks, depth_used, ask_side=True)
        curr_sell_vwap = vwap_to_depth(curr.bids, depth_used, ask_side=False)
        mechanical_buy_vwap = vwap_to_depth(
            consume_levels(prev.asks, curr.buy_volume, ask_side=True),
            depth_used,
            ask_side=True,
        )
        mechanical_sell_vwap = vwap_to_depth(
            consume_levels(prev.bids, curr.sell_volume, ask_side=False),
            depth_used,
            ask_side=False,
        )

        if side == "buy_vwap_complete_current":
            observed_delta[idx] = curr_buy_vwap - prev_buy_vwap
            mechanical_delta[idx] = mechanical_buy_vwap - prev_buy_vwap
        elif side == "sell_vwap_complete_current":
            observed_delta[idx] = curr_sell_vwap - prev_sell_vwap
            mechanical_delta[idx] = mechanical_sell_vwap - prev_sell_vwap
        else:
            raise ValueError(f"unknown complete vwap side {side}")

    base, base_names = build_complete_book_feature_base(rows, include_depth_ratios=include_depth_ratios)
    lagged_features, lagged_names = build_base_lagged_features(base, base_names, lags=lags)
    feature_matrix = np.column_stack([mechanical_delta, lagged_features])
    feature_names = ["mechanical_delta", *lagged_names]
    start = max(1, lags)
    y = observed_delta[start:]
    x_without_intercept = feature_matrix[start:, :]
    mechanical = mechanical_delta[start:]
    valid = np.isfinite(y) & np.isfinite(mechanical) & np.all(np.isfinite(x_without_intercept), axis=1)
    y = y[valid]
    mechanical = mechanical[valid]
    x_without_intercept = x_without_intercept[valid]
    if y.size < min_rows:
        return None
    y_name = "complete_current_buy_vwap_delta" if side == "buy_vwap_complete_current" else "complete_current_sell_vwap_delta"
    return y, x_without_intercept, ["intercept", *feature_names], y_name, mechanical


def build_complete_book_feature_base(
    rows: Sequence[MarketRow],
    *,
    include_depth_ratios: bool,
) -> tuple[np.ndarray, list[str]]:
    eps = 1e-9
    buy = np.array([row.buy_volume for row in rows], dtype=np.float64)
    sell = np.array([row.sell_volume for row in rows], dtype=np.float64)
    ask_qty1 = np.array([row.best_ask_qty for row in rows], dtype=np.float64)
    bid_qty1 = np.array([row.best_bid_qty for row in rows], dtype=np.float64)
    ask_depth5 = np.array([row.ask_depth_5 for row in rows], dtype=np.float64)
    bid_depth5 = np.array([row.bid_depth_5 for row in rows], dtype=np.float64)
    spread = np.array([row.spread for row in rows], dtype=np.float64)
    queue_imbalance = np.array([row.queue_imbalance_5 for row in rows], dtype=np.float64)
    microprice_minus_mid = np.array([row.microprice_minus_mid for row in rows], dtype=np.float64)
    arrays = [buy, sell]
    names = [
        "buy_volume",
        "sell_volume",
    ]
    if include_depth_ratios:
        arrays.extend(
            [
                buy / np.maximum(ask_qty1, eps),
                buy / np.maximum(ask_depth5, eps),
                sell / np.maximum(bid_qty1, eps),
                sell / np.maximum(bid_depth5, eps),
            ]
        )
        names.extend(
            [
                "buy_over_ask_qty1",
                "buy_over_ask_depth5",
                "sell_over_bid_qty1",
                "sell_over_bid_depth5",
            ]
        )
    arrays.extend([spread, queue_imbalance, microprice_minus_mid])
    names.extend([
        "spread",
        "queue_imbalance_5",
        "microprice_minus_mid",
    ])
    return np.column_stack(arrays), names


def consume_best_price(levels: Sequence[tuple[float, float]], qty: float, *, side: str) -> float:
    if not levels:
        return float("nan")
    remaining = max(0.0, float(qty))
    for price, level_qty in levels:
        if remaining < max(0.0, level_qty):
            return price
        remaining -= max(0.0, level_qty)
    return levels[-1][0]


def consume_levels(
    levels: Sequence[tuple[float, float]],
    qty: float,
    *,
    ask_side: bool,
) -> list[tuple[float, float]]:
    remaining = max(0.0, float(qty))
    out: list[tuple[float, float]] = []
    sorted_levels = sorted(levels, key=lambda item: item[0], reverse=not ask_side)
    for price, level_qty in sorted_levels:
        q = max(0.0, float(level_qty))
        if remaining > 0:
            take = min(q, remaining)
            q -= take
            remaining -= take
        if q > 1e-12:
            out.append((float(price), q))
    return out


def total_depth(levels: Sequence[tuple[float, float]]) -> float:
    return float(sum(max(0.0, qty) for _price, qty in levels))


def comparable_vwap_depth(prev: MarketRow, curr: MarketRow, vwap_depth: float) -> float:
    ask_after = consume_levels(prev.asks, curr.buy_volume, ask_side=True)
    bid_after = consume_levels(prev.bids, curr.sell_volume, ask_side=False)
    return min(
        max(0.0, float(vwap_depth)),
        total_depth(curr.asks),
        total_depth(curr.bids),
        total_depth(ask_after),
        total_depth(bid_after),
    )


def vwap_to_depth(levels: Sequence[tuple[float, float]], depth: float, *, ask_side: bool) -> float:
    remaining = max(0.0, float(depth))
    if remaining <= 1e-12:
        return float("nan")
    notional = 0.0
    used = remaining
    sorted_levels = sorted(levels, key=lambda item: item[0], reverse=not ask_side)
    for price, qty in sorted_levels:
        if remaining <= 1e-12:
            break
        q = max(0.0, float(qty))
        if q <= 1e-12:
            continue
        take = min(q, remaining)
        notional += float(price) * take
        remaining -= take
    if remaining > 1e-9:
        return float("nan")
    return float(notional / used)


def build_side_design_matrix(
    rows: Sequence[MarketRow],
    *,
    side: str,
    target: str,
    lags: int,
    min_rows: int,
    book_features: bool = False,
    current_step_book_features: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str], str] | None:
    if len(rows) < max(min_rows, lags + 3):
        return None

    if side == "bid_from_sell":
        prices = np.array([row.best_bid for row in rows], dtype=np.float64)
        if book_features:
            base, base_names = build_book_feature_base(rows, side=side)
        else:
            base = np.array([[row.sell_volume] for row in rows], dtype=np.float64)
            base_names = ["sell_volume"]
        price_name = "best_bid"
    elif side == "ask_from_buy":
        prices = np.array([row.best_ask for row in rows], dtype=np.float64)
        if book_features:
            base, base_names = build_book_feature_base(rows, side=side)
        else:
            base = np.array([[row.buy_volume] for row in rows], dtype=np.float64)
            base_names = ["buy_volume"]
        price_name = "best_ask"
    else:
        raise ValueError(f"unknown side model {side}")

    if target == "log_return":
        y_all = np.diff(np.log(prices))
        y_name = f"{price_name}_log_return"
    elif target == "delta":
        y_all = np.diff(prices)
        y_name = f"{price_name}_delta"
    else:
        raise ValueError(f"unknown target {target}")

    feature_matrix, feature_names = build_base_lagged_features(base, base_names, lags=lags)
    start = max(1, lags)
    if book_features and not current_step_book_features:
        # Predict the next book move using features known at t.
        y = y_all[start:]
        x_without_intercept = feature_matrix[start:-1, :]
        y_name = f"next_{y_name}"
    else:
        if book_features and current_step_book_features:
            y_name = f"current_{y_name}"
        y = y_all[start - 1 :]
        x_without_intercept = feature_matrix[start:, :]
    valid = np.isfinite(y) & np.all(np.isfinite(x_without_intercept), axis=1)
    y = y[valid]
    x_without_intercept = x_without_intercept[valid]
    if y.size < min_rows:
        return None

    return y, x_without_intercept, ["intercept", *feature_names], y_name


def build_book_feature_base(rows: Sequence[MarketRow], *, side: str) -> tuple[np.ndarray, list[str]]:
    eps = 1e-9
    if side == "bid_from_sell":
        flow = np.array([row.sell_volume for row in rows], dtype=np.float64)
        depth_1 = np.array([row.best_bid_qty for row in rows], dtype=np.float64)
        depth_5 = np.array([row.bid_depth_5 for row in rows], dtype=np.float64)
        names = [
            "sell_volume",
            "sell_over_bid_qty1",
            "sell_over_bid_depth5",
            "spread",
            "queue_imbalance_5",
            "microprice_minus_mid",
        ]
    elif side == "ask_from_buy":
        flow = np.array([row.buy_volume for row in rows], dtype=np.float64)
        depth_1 = np.array([row.best_ask_qty for row in rows], dtype=np.float64)
        depth_5 = np.array([row.ask_depth_5 for row in rows], dtype=np.float64)
        names = [
            "buy_volume",
            "buy_over_ask_qty1",
            "buy_over_ask_depth5",
            "spread",
            "queue_imbalance_5",
            "microprice_minus_mid",
        ]
    else:
        raise ValueError(f"unknown side model {side}")
    spread = np.array([row.spread for row in rows], dtype=np.float64)
    queue_imbalance = np.array([row.queue_imbalance_5 for row in rows], dtype=np.float64)
    microprice_minus_mid = np.array([row.microprice_minus_mid for row in rows], dtype=np.float64)
    base = np.column_stack(
        [
            flow,
            flow / np.maximum(depth_1, eps),
            flow / np.maximum(depth_5, eps),
            spread,
            queue_imbalance,
            microprice_minus_mid,
        ]
    )
    return base, names


def evaluate_predictions(y: np.ndarray, y_hat: np.ndarray, baseline: float) -> EvaluationResult:
    residuals = y - y_hat
    baseline_residuals = y - baseline
    zero_baseline_residuals = y
    sse = float(residuals @ residuals)
    baseline_sse = float(baseline_residuals @ baseline_residuals)
    zero_baseline_sse = float(zero_baseline_residuals @ zero_baseline_residuals)
    rmse = math.sqrt(sse / y.size)
    baseline_rmse = math.sqrt(baseline_sse / y.size)
    zero_baseline_rmse = math.sqrt(zero_baseline_sse / y.size)
    centered = y - float(np.mean(y))
    sst = float(centered @ centered)
    r2 = 1.0 - sse / sst if sst > 0 else float("nan")
    direction_accuracy = float(np.mean(np.sign(y_hat) == np.sign(y)))
    baseline_direction = float(np.mean(np.sign(np.full_like(y, baseline)) == np.sign(y)))
    zero_baseline_direction = float(np.mean(np.sign(np.zeros_like(y)) == np.sign(y)))
    return EvaluationResult(
        n=int(y.size),
        rmse=rmse,
        r2=r2,
        baseline_rmse=baseline_rmse,
        zero_baseline_rmse=zero_baseline_rmse,
        direction_accuracy=direction_accuracy,
        baseline_direction_accuracy=baseline_direction,
        zero_baseline_direction_accuracy=zero_baseline_direction,
    )


def build_lagged_features(
    rows: Sequence[MarketRow],
    *,
    feature_kind: str,
    lags: int,
) -> tuple[np.ndarray, list[str]]:
    if feature_kind == "signed":
        base = np.array([[row.signed_flow] for row in rows], dtype=np.float64)
        base_names = ["signed_flow"]
    elif feature_kind == "imbalance":
        base = np.array([[row.imbalance] for row in rows], dtype=np.float64)
        base_names = ["imbalance"]
    elif feature_kind == "buy_sell":
        base = np.array([[row.buy_volume, row.sell_volume] for row in rows], dtype=np.float64)
        base_names = ["buy_volume", "sell_volume"]
    else:
        raise ValueError(f"unknown feature kind {feature_kind}")

    return build_base_lagged_features(base, base_names, lags=lags)


def build_base_lagged_features(
    base: np.ndarray,
    base_names: list[str],
    *,
    lags: int,
) -> tuple[np.ndarray, list[str]]:
    blocks: list[np.ndarray] = []
    names: list[str] = []
    for lag in range(lags + 1):
        shifted = np.full_like(base, np.nan)
        if lag == 0:
            shifted[:] = base
        else:
            shifted[lag:, :] = base[:-lag, :]
        blocks.append(shifted)
        suffix = "t" if lag == 0 else f"t-{lag}"
        names.extend(f"{name}_{suffix}" for name in base_names)
    return np.column_stack(blocks), names


def print_header(
    role: str,
    exchange: str,
    symbol: str,
    interval_ms: int,
    topic: str,
    latest_ts: datetime,
) -> None:
    latest = latest_ts.astimezone(timezone.utc).isoformat()
    print(f"\n=== {role.upper()} {exchange} {symbol} interval={interval_ms}ms ===")
    print(f"topic: {topic}")
    print(f"latest event: {latest}")


def print_window_result(
    label: str,
    rows: Sequence[MarketRow],
    result: FitResult | None,
    split_result: TrainTestResult | None,
) -> None:
    if result is None:
        print(f"\n[{label}] rows={len(rows)}: not enough usable rows")
        return
    print(f"\n[{label}] rows={len(rows)} fit_n={result.n} target={result.y_name}")
    print(f"R2={result.r2:.6g}  RMSE={result.rmse:.6g}  sigma2_mle={result.sigma2_mle:.6g}")
    if split_result is not None:
        print(
            f"train/test {split_result.split_ratio:.0%}/{1 - split_result.split_ratio:.0%}: "
            f"train_n={split_result.train.n} test_n={split_result.test.n}  "
            f"test_R2={split_result.test.r2:.6g}  "
            f"test_RMSE={split_result.test.rmse:.6g} "
            f"(mean_baseline={split_result.test.baseline_rmse:.6g}, "
            f"zero_baseline={split_result.test.zero_baseline_rmse:.6g})  "
            f"test_dir_acc={split_result.test.direction_accuracy:.3f} "
            f"(mean_baseline={split_result.test.baseline_direction_accuracy:.3f}, "
            f"zero_baseline={split_result.test.zero_baseline_direction_accuracy:.3f})"
        )
    print("coefficient                         beta          stderr        t_stat       corr_y")
    for idx, name in enumerate(result.feature_names):
        corr_text = ""
        if idx > 0:
            corr_text = f"{result.corr[idx - 1]: .6g}"
        print(
            f"{name:<28} "
            f"{result.beta[idx]:>13.6g} "
            f"{result.stderr[idx]:>13.6g} "
            f"{result.t_stats[idx]:>12.4g} "
            f"{corr_text:>12}"
        )


def print_side_window_results(
    label: str,
    rows: Sequence[MarketRow],
    *,
    lags: int,
    min_rows: int,
    train_ratio: float,
    target: str,
    book_features: bool,
    current_step_book_features: bool,
    mechanical_residual: bool,
) -> None:
    if len(rows) < min_rows:
        print(f"\n[{label}] rows={len(rows)}: not enough usable rows")
        return
    for title, side in (("best_bid ~ sell volume", "bid_from_sell"), ("best_ask ~ buy volume", "ask_from_buy")):
        result = fit_side_model(
            rows,
            side=side,
            target=target,
            lags=lags,
            min_rows=min_rows,
            book_features=book_features,
            current_step_book_features=current_step_book_features,
            mechanical_residual=mechanical_residual,
        )
        split_result = train_test_side_model(
            rows,
            side=side,
            target=target,
            lags=lags,
            min_rows=min_rows,
            train_ratio=train_ratio,
            book_features=book_features,
            current_step_book_features=current_step_book_features,
            mechanical_residual=mechanical_residual,
        )
        if mechanical_residual:
            suffix = " mechanical residual, current-step target"
            if book_features:
                suffix += ", book features"
        elif book_features:
            if current_step_book_features:
                suffix = " with book features, current-step target"
            else:
                suffix = " with book features, next-step target"
        else:
            suffix = ""
        print(f"\n[{label}] {title}{suffix}")
        print_window_result("side", rows, result, split_result)


def print_side_mechanical_delta_window_results(
    label: str,
    rows: Sequence[MarketRow],
    *,
    min_rows: int,
    train_ratio: float,
) -> None:
    if len(rows) < min_rows:
        print(f"\n[{label}] rows={len(rows)}: not enough usable rows")
        return
    specs = (
        ("best_bid current delta ~ mechanical_delta only", "bid_from_sell"),
        ("best_ask current delta ~ mechanical_delta only", "ask_from_buy"),
    )
    for title, side in specs:
        result = fit_side_mechanical_delta_model(rows, side=side, min_rows=min_rows)
        evals = train_test_side_mechanical_delta_model(
            rows,
            side=side,
            min_rows=min_rows,
            train_ratio=train_ratio,
        )
        split_result = evals[0] if evals is not None else None
        mechanical_eval = evals[1] if evals is not None else None
        print(f"\n[{label}] {title}")
        print_window_result("side-mechanical-delta", rows, result, split_result)
        if mechanical_eval is not None:
            print(
                "mechanical-only test: "
                f"R2={mechanical_eval.r2:.6g}  "
                f"RMSE={mechanical_eval.rmse:.6g}  "
                f"mean_baseline_RMSE={mechanical_eval.baseline_rmse:.6g}  "
                f"zero_baseline_RMSE={mechanical_eval.zero_baseline_rmse:.6g}"
            )


def print_online_side_mechanical_delta_window_results(
    label: str,
    rows: Sequence[MarketRow],
    *,
    min_rows: int,
    decay: float,
    initial_sigma: float,
) -> None:
    if len(rows) < min_rows:
        print(f"\n[{label}] rows={len(rows)}: not enough usable rows")
        return
    specs = (
        ("best_bid current delta", "bid_from_sell"),
        ("best_ask current delta", "ask_from_buy"),
    )
    for title, side in specs:
        result = online_side_mechanical_delta_model(
            rows,
            side=side,
            min_rows=min_rows,
            decay=decay,
            initial_sigma=initial_sigma,
        )
        if result is None:
            print(f"\n[{label}] {title} online mechanical-delta-only: not enough scored rows")
            continue
        beta_text = ", ".join(f"{value:.6g}" for value in result.final_beta)
        print(
            f"\n[{label}] {title} online mechanical-delta-only "
            f"decay={decay:g} scored_n={result.n_scored}"
        )
        print(
            f"model_RMSE={result.rmse:.6g}  "
            f"model_NLL={result.nll:.6g}  "
            f"mean_gaussian_RMSE={result.mean_rmse:.6g}  "
            f"mean_gaussian_NLL={result.mean_nll:.6g}  "
            f"zero_RMSE={result.zero_rmse:.6g}  "
            f"zero_NLL={result.zero_nll:.6g}  "
            f"mechanical_RMSE={result.mechanical_rmse:.6g}"
        )
        print(
            f"final_beta=[{beta_text}]  "
            f"final_mean={result.final_mean:.6g}  "
            f"final_sigma={result.final_sigma:.6g}  "
            f"final_mean_sigma={result.final_mean_sigma:.6g}  "
            f"final_zero_sigma={result.final_zero_sigma:.6g}"
        )


def print_online_decomposed_hurdle_vwap_results(
    label: str,
    rows: Sequence[MarketRow],
    *,
    min_rows: int,
    decay: float,
    initial_sigma: float,
    vwap_depth: float,
    flow_threshold: float,
    mechanical_threshold: float,
    thin_depth_threshold: float,
    hard_gated_quiet: bool,
    gate_mode: str,
    depth_percentile: float,
    flow_depth_ratio: float,
) -> None:
    result = online_decomposed_hurdle_vwap_model(
        rows,
        min_rows=min_rows,
        decay=decay,
        initial_sigma=initial_sigma,
        vwap_depth=vwap_depth,
        flow_threshold=flow_threshold,
        mechanical_threshold=mechanical_threshold,
        thin_depth_threshold=thin_depth_threshold,
        hard_gated_quiet=hard_gated_quiet,
        gate_mode=gate_mode,
        depth_percentile=depth_percentile,
        flow_depth_ratio=flow_depth_ratio,
    )
    if result is None:
        print(f"\n[{label}] decomposed hurdle VWAP: not enough usable rows")
        return
    gate_label = "hard-gated quiet" if hard_gated_quiet else "mechanical quiet"
    print(
        f"\n[{label}] decomposed hurdle VWAP ({gate_label}, gate={gate_mode}) depth={vwap_depth:g} decay={decay:g} "
        f"scored_n={result.n_scored} active_fraction={result.active_fraction:.3f}"
    )
    print(
        "buy_vwap: "
        f"model_RMSE={result.buy_rmse:.6g}  "
        f"model_NLL={result.buy_nll:.6g}  "
        f"zero_RMSE={result.buy_zero_rmse:.6g}  "
        f"zero_NLL={result.buy_zero_nll:.6g}  "
        f"mechanical_RMSE={result.buy_mechanical_rmse:.6g}"
    )
    print(
        "sell_vwap: "
        f"model_RMSE={result.sell_rmse:.6g}  "
        f"model_NLL={result.sell_nll:.6g}  "
        f"zero_RMSE={result.sell_zero_rmse:.6g}  "
        f"zero_NLL={result.sell_zero_nll:.6g}  "
        f"mechanical_RMSE={result.sell_mechanical_rmse:.6g}"
        )


def print_online_decomposed_two_regime_vwap_results(
    label: str,
    rows: Sequence[MarketRow],
    *,
    min_rows: int,
    decay: float,
    initial_sigma: float,
    vwap_depth: float,
) -> None:
    if len(rows) < min_rows:
        print(f"\n[{label}] rows={len(rows)}: not enough usable rows")
        return
    result = online_decomposed_two_regime_vwap_model(
        rows,
        min_rows=min_rows,
        decay=decay,
        initial_sigma=initial_sigma,
        vwap_depth=vwap_depth,
    )
    if result is None:
        print(f"\n[{label}] decomposed two-regime VWAP: not enough scored rows")
        return
    print(
        f"\n[{label}] decomposed two-regime VWAP "
        f"(market linear mechanical_delta, refresh adaptive Gaussian) "
        f"depth={vwap_depth:g} decay={decay:g} scored_n={result.n_scored} "
        f"market_fraction={result.active_fraction:.3f}"
    )
    print(
        f"buy_vwap: model_RMSE={result.buy_rmse:.6g}  "
        f"model_NLL={result.buy_nll:.6g}  "
        f"zero_RMSE={result.buy_zero_rmse:.6g}  "
        f"zero_NLL={result.buy_zero_nll:.6g}  "
        f"mechanical_RMSE={result.buy_mechanical_rmse:.6g}"
    )
    print(
        f"sell_vwap: model_RMSE={result.sell_rmse:.6g}  "
        f"model_NLL={result.sell_nll:.6g}  "
        f"zero_RMSE={result.sell_zero_rmse:.6g}  "
        f"zero_NLL={result.sell_zero_nll:.6g}  "
        f"mechanical_RMSE={result.sell_mechanical_rmse:.6g}"
    )


def print_online_decomposed_flow_gated_mixture_vwap_results(
    label: str,
    rows: Sequence[MarketRow],
    *,
    min_rows: int,
    decay: float,
    initial_sigma: float,
    vwap_depth: float,
    interval_ms: int,
    continuous_sections: bool,
    max_gap_factor: float,
    gate_learning_rate: float,
    gate_kind: str,
    gate_bins: Sequence[float],
    point_mode: str,
    refresh_distribution: str,
    refresh_student_df: float,
    refresh_jump_sigma_mult: float,
    spread_features: str,
) -> None:
    if len(rows) < min_rows:
        print(f"\n[{label}] rows={len(rows)}: not enough usable rows")
        return

    sections = [list(rows)]
    if continuous_sections:
        sections = split_continuous_sections(
            rows,
            interval_ms=interval_ms,
            max_gap_factor=max_gap_factor,
            min_rows=min_rows,
        )
    if not sections:
        print(f"\n[{label}] flow-gated mixture VWAP: no continuous section with enough rows")
        return

    scored_results: list[DecomposedHurdleEvaluation] = []
    for section_idx, section in enumerate(sections, start=1):
        result = online_decomposed_flow_gated_mixture_vwap_model(
            section,
            min_rows=min_rows,
            decay=decay,
            initial_sigma=initial_sigma,
            vwap_depth=vwap_depth,
            gate_learning_rate=gate_learning_rate,
            gate_kind=gate_kind,
            gate_bins=gate_bins,
            point_mode=point_mode,
            refresh_distribution=refresh_distribution,
            refresh_student_df=refresh_student_df,
            refresh_jump_sigma_mult=refresh_jump_sigma_mult,
            spread_features=spread_features,
        )
        section_label = label
        if continuous_sections:
            section_label = (
                f"{label} section={section_idx}/{len(sections)} "
                f"{section[0].ts.isoformat()}..{section[-1].ts.isoformat()}"
            )
        if result is None:
            print(f"\n[{section_label}] flow-gated mixture VWAP: not enough scored rows")
            continue
        scored_results.append(result)
        gate_desc = "pi_market=sigmoid(a0+a1 log1p(total_flow))"
        gate_update_desc = f"gate_lr={gate_learning_rate:g}"
        if gate_kind == "binned":
            formatted_bins = ",".join(f"{edge:g}" for edge in sorted(gate_bins))
            gate_desc = f"pi_market=EWMA_bin(log1p(total_flow); edges={formatted_bins})"
            gate_update_desc = "gate_update=ewma_decay"
        print(
            f"\n[{section_label}] flow-gated mixture VWAP "
            f"{gate_desc} "
            f"components=mid+buy_shape+sell_shape refresh={refresh_distribution} "
            f"df={refresh_student_df:g} jump_mult={refresh_jump_sigma_mult:g} "
            f"depth={vwap_depth:g} decay={decay:g} "
            f"gate_kind={gate_kind} {gate_update_desc} point_mode={point_mode} "
            f"spread_features={spread_features} "
            f"scored_n={result.n_scored} mean_market_prob={result.active_fraction:.3f}"
        )
        print(
            f"buy_vwap: model_RMSE={result.buy_rmse:.6g}  "
            f"model_NLL={result.buy_nll:.6g}  "
            f"zero_RMSE={result.buy_zero_rmse:.6g}  "
            f"zero_NLL={result.buy_zero_nll:.6g}  "
            f"mechanical_RMSE={result.buy_mechanical_rmse:.6g}"
        )
        print(
            f"sell_vwap: model_RMSE={result.sell_rmse:.6g}  "
            f"model_NLL={result.sell_nll:.6g}  "
            f"zero_RMSE={result.sell_zero_rmse:.6g}  "
            f"zero_NLL={result.sell_zero_nll:.6g}  "
            f"mechanical_RMSE={result.sell_mechanical_rmse:.6g}"
        )

    if continuous_sections and scored_results:
        total_n = sum(result.n_scored for result in scored_results)
        buy_sse = sum(result.n_scored * result.buy_rmse**2 for result in scored_results)
        sell_sse = sum(result.n_scored * result.sell_rmse**2 for result in scored_results)
        buy_zero_sse = sum(result.n_scored * result.buy_zero_rmse**2 for result in scored_results)
        sell_zero_sse = sum(result.n_scored * result.sell_zero_rmse**2 for result in scored_results)
        buy_mech_sse = sum(result.n_scored * result.buy_mechanical_rmse**2 for result in scored_results)
        sell_mech_sse = sum(result.n_scored * result.sell_mechanical_rmse**2 for result in scored_results)
        buy_nll = sum(result.n_scored * result.buy_nll for result in scored_results) / total_n
        sell_nll = sum(result.n_scored * result.sell_nll for result in scored_results) / total_n
        buy_zero_nll = sum(result.n_scored * result.buy_zero_nll for result in scored_results) / total_n
        sell_zero_nll = sum(result.n_scored * result.sell_zero_nll for result in scored_results) / total_n
        mean_market_prob = sum(result.n_scored * result.active_fraction for result in scored_results) / total_n
        buy_zero_wins = sum(result.buy_rmse < result.buy_zero_rmse for result in scored_results)
        sell_zero_wins = sum(result.sell_rmse < result.sell_zero_rmse for result in scored_results)
        buy_mech_wins = sum(result.buy_rmse < result.buy_mechanical_rmse for result in scored_results)
        sell_mech_wins = sum(result.sell_rmse < result.sell_mechanical_rmse for result in scored_results)
        section_count = len(scored_results)
        print(
            f"\n[{label} aggregate over {section_count} scored continuous sections] "
            f"flow-gated mixture VWAP gate_kind={gate_kind} "
            f"scored_n={total_n} mean_market_prob={mean_market_prob:.3f}"
        )
        print(
            f"buy_vwap: model_RMSE={math.sqrt(buy_sse / total_n):.6g}  "
            f"model_NLL={buy_nll:.6g}  "
            f"zero_RMSE={math.sqrt(buy_zero_sse / total_n):.6g}  "
            f"zero_NLL={buy_zero_nll:.6g}  "
            f"mechanical_RMSE={math.sqrt(buy_mech_sse / total_n):.6g}  "
            f"wins_vs_zero={buy_zero_wins}/{section_count}  "
            f"wins_vs_mechanical={buy_mech_wins}/{section_count}"
        )
        print(
            f"sell_vwap: model_RMSE={math.sqrt(sell_sse / total_n):.6g}  "
            f"model_NLL={sell_nll:.6g}  "
            f"zero_RMSE={math.sqrt(sell_zero_sse / total_n):.6g}  "
            f"zero_NLL={sell_zero_nll:.6g}  "
            f"mechanical_RMSE={math.sqrt(sell_mech_sse / total_n):.6g}  "
            f"wins_vs_zero={sell_zero_wins}/{section_count}  "
            f"wins_vs_mechanical={sell_mech_wins}/{section_count}"
        )


def print_online_decomposed_em_mixture_vwap_results(
    label: str,
    rows: Sequence[MarketRow],
    *,
    min_rows: int,
    decay: float,
    initial_sigma: float,
    vwap_depth: float,
    interval_ms: int,
    continuous_sections: bool,
    max_gap_factor: float,
    point_mode: str,
) -> None:
    if len(rows) < min_rows:
        print(f"\n[{label}] rows={len(rows)}: not enough usable rows")
        return

    sections = [list(rows)]
    if continuous_sections:
        sections = split_continuous_sections(
            rows,
            interval_ms=interval_ms,
            max_gap_factor=max_gap_factor,
            min_rows=min_rows,
        )
    if not sections:
        print(f"\n[{label}] decomposed EM-mixture VWAP: no continuous section with enough rows")
        return

    scored_results: list[DecomposedHurdleEvaluation] = []
    for section_idx, section in enumerate(sections, start=1):
        result = online_decomposed_em_mixture_vwap_model(
            section,
            min_rows=min_rows,
            decay=decay,
            initial_sigma=initial_sigma,
            vwap_depth=vwap_depth,
            point_mode=point_mode,
        )
        section_label = label
        if continuous_sections:
            section_label = (
                f"{label} section={section_idx}/{len(sections)} "
                f"{section[0].ts.isoformat()}..{section[-1].ts.isoformat()}"
            )
        if result is None:
            print(f"\n[{section_label}] decomposed EM-mixture VWAP: not enough scored rows")
            continue
        scored_results.append(result)
        print(
            f"\n[{section_label}] decomposed EM-mixture VWAP "
            f"pi_market=EWMA posterior, components=mid+buy_shape+sell_shape "
            f"depth={vwap_depth:g} decay={decay:g} point_mode={point_mode} "
            f"scored_n={result.n_scored} mean_market_prob={result.active_fraction:.3f}"
        )
        print(
            f"buy_vwap: model_RMSE={result.buy_rmse:.6g}  "
            f"model_NLL={result.buy_nll:.6g}  "
            f"zero_RMSE={result.buy_zero_rmse:.6g}  "
            f"zero_NLL={result.buy_zero_nll:.6g}  "
            f"mechanical_RMSE={result.buy_mechanical_rmse:.6g}"
        )
        print(
            f"sell_vwap: model_RMSE={result.sell_rmse:.6g}  "
            f"model_NLL={result.sell_nll:.6g}  "
            f"zero_RMSE={result.sell_zero_rmse:.6g}  "
            f"zero_NLL={result.sell_zero_nll:.6g}  "
            f"mechanical_RMSE={result.sell_mechanical_rmse:.6g}"
        )

    if continuous_sections and scored_results:
        total_n = sum(result.n_scored for result in scored_results)
        buy_sse = sum(result.n_scored * result.buy_rmse**2 for result in scored_results)
        sell_sse = sum(result.n_scored * result.sell_rmse**2 for result in scored_results)
        buy_zero_sse = sum(result.n_scored * result.buy_zero_rmse**2 for result in scored_results)
        sell_zero_sse = sum(result.n_scored * result.sell_zero_rmse**2 for result in scored_results)
        buy_mech_sse = sum(result.n_scored * result.buy_mechanical_rmse**2 for result in scored_results)
        sell_mech_sse = sum(result.n_scored * result.sell_mechanical_rmse**2 for result in scored_results)
        buy_nll = sum(result.n_scored * result.buy_nll for result in scored_results) / total_n
        sell_nll = sum(result.n_scored * result.sell_nll for result in scored_results) / total_n
        buy_zero_nll = sum(result.n_scored * result.buy_zero_nll for result in scored_results) / total_n
        sell_zero_nll = sum(result.n_scored * result.sell_zero_nll for result in scored_results) / total_n
        mean_market_prob = sum(result.n_scored * result.active_fraction for result in scored_results) / total_n
        buy_zero_wins = sum(result.buy_rmse < result.buy_zero_rmse for result in scored_results)
        sell_zero_wins = sum(result.sell_rmse < result.sell_zero_rmse for result in scored_results)
        buy_mech_wins = sum(result.buy_rmse < result.buy_mechanical_rmse for result in scored_results)
        sell_mech_wins = sum(result.sell_rmse < result.sell_mechanical_rmse for result in scored_results)
        section_count = len(scored_results)
        print(
            f"\n[{label} aggregate over {section_count} scored continuous sections] "
            f"EM-mixture VWAP scored_n={total_n} mean_market_prob={mean_market_prob:.3f}"
        )
        print(
            f"buy_vwap: model_RMSE={math.sqrt(buy_sse / total_n):.6g}  "
            f"model_NLL={buy_nll:.6g}  "
            f"zero_RMSE={math.sqrt(buy_zero_sse / total_n):.6g}  "
            f"zero_NLL={buy_zero_nll:.6g}  "
            f"mechanical_RMSE={math.sqrt(buy_mech_sse / total_n):.6g}  "
            f"wins_vs_zero={buy_zero_wins}/{section_count}  "
            f"wins_vs_mechanical={buy_mech_wins}/{section_count}"
        )
        print(
            f"sell_vwap: model_RMSE={math.sqrt(sell_sse / total_n):.6g}  "
            f"model_NLL={sell_nll:.6g}  "
            f"zero_RMSE={math.sqrt(sell_zero_sse / total_n):.6g}  "
            f"zero_NLL={sell_zero_nll:.6g}  "
            f"mechanical_RMSE={math.sqrt(sell_mech_sse / total_n):.6g}  "
            f"wins_vs_zero={sell_zero_wins}/{section_count}  "
            f"wins_vs_mechanical={sell_mech_wins}/{section_count}"
        )


def print_online_decomposed_soft_mixture_vwap_results(
    label: str,
    rows: Sequence[MarketRow],
    *,
    min_rows: int,
    decay: float,
    initial_sigma: float,
    vwap_depth: float,
    interval_ms: int,
    continuous_sections: bool,
    max_gap_factor: float,
    gate_learning_rate: float,
    diagnostic_count: int,
    market_deviation_cap_mult: float,
    market_guard_mode: str,
) -> None:
    if len(rows) < min_rows:
        print(f"\n[{label}] rows={len(rows)}: not enough usable rows")
        return

    sections = [list(rows)]
    if continuous_sections:
        sections = split_continuous_sections(
            rows,
            interval_ms=interval_ms,
            max_gap_factor=max_gap_factor,
            min_rows=min_rows,
        )
    if not sections:
        print(f"\n[{label}] decomposed soft-mixture VWAP: no continuous section with enough rows")
        return

    scored_results: list[DecomposedHurdleEvaluation] = []
    diagnostics: list[dict[str, object]] = []
    for section_idx, section in enumerate(sections, start=1):
        section_label = label
        if continuous_sections:
            section_label = (
                f"{label} section={section_idx}/{len(sections)} "
                f"{section[0].ts.isoformat()}..{section[-1].ts.isoformat()}"
            )
        result = online_decomposed_soft_mixture_vwap_model(
            section,
            min_rows=min_rows,
            decay=decay,
            initial_sigma=initial_sigma,
            vwap_depth=vwap_depth,
            gate_learning_rate=gate_learning_rate,
            market_deviation_cap_mult=market_deviation_cap_mult,
            market_guard_mode=market_guard_mode,
            diagnostics=diagnostics if diagnostic_count > 0 else None,
            diagnostic_section=section_label,
        )
        if result is None:
            print(f"\n[{section_label}] decomposed soft-mixture VWAP: not enough scored rows")
            continue
        scored_results.append(result)
        print(
            f"\n[{section_label}] decomposed soft-mixture VWAP "
            f"q=sigmoid(a0+a1 zero_flow+a2 abs_mech_mid+a3 spread_prev) "
            f"depth={vwap_depth:g} decay={decay:g} gate_lr={gate_learning_rate:g} "
            f"market_cap_mult={market_deviation_cap_mult:g} "
            f"market_guard_mode={market_guard_mode} "
            f"scored_n={result.n_scored} mean_market_prob={result.active_fraction:.3f}"
        )
        print(
            f"buy_vwap: model_RMSE={result.buy_rmse:.6g}  "
            f"model_NLL={result.buy_nll:.6g}  "
            f"zero_RMSE={result.buy_zero_rmse:.6g}  "
            f"zero_NLL={result.buy_zero_nll:.6g}  "
            f"mechanical_RMSE={result.buy_mechanical_rmse:.6g}"
        )
        print(
            f"sell_vwap: model_RMSE={result.sell_rmse:.6g}  "
            f"model_NLL={result.sell_nll:.6g}  "
            f"zero_RMSE={result.sell_zero_rmse:.6g}  "
            f"zero_NLL={result.sell_zero_nll:.6g}  "
            f"mechanical_RMSE={result.sell_mechanical_rmse:.6g}"
        )

    if continuous_sections and scored_results:
        total_n = sum(result.n_scored for result in scored_results)
        buy_sse = sum(result.n_scored * result.buy_rmse**2 for result in scored_results)
        sell_sse = sum(result.n_scored * result.sell_rmse**2 for result in scored_results)
        buy_zero_sse = sum(result.n_scored * result.buy_zero_rmse**2 for result in scored_results)
        sell_zero_sse = sum(result.n_scored * result.sell_zero_rmse**2 for result in scored_results)
        buy_mech_sse = sum(result.n_scored * result.buy_mechanical_rmse**2 for result in scored_results)
        sell_mech_sse = sum(result.n_scored * result.sell_mechanical_rmse**2 for result in scored_results)
        buy_nll = sum(result.n_scored * result.buy_nll for result in scored_results) / total_n
        sell_nll = sum(result.n_scored * result.sell_nll for result in scored_results) / total_n
        buy_zero_nll = sum(result.n_scored * result.buy_zero_nll for result in scored_results) / total_n
        sell_zero_nll = sum(result.n_scored * result.sell_zero_nll for result in scored_results) / total_n
        mean_market_prob = sum(result.n_scored * result.active_fraction for result in scored_results) / total_n
        buy_zero_wins = sum(result.buy_rmse < result.buy_zero_rmse for result in scored_results)
        sell_zero_wins = sum(result.sell_rmse < result.sell_zero_rmse for result in scored_results)
        buy_mech_wins = sum(result.buy_rmse < result.buy_mechanical_rmse for result in scored_results)
        sell_mech_wins = sum(result.sell_rmse < result.sell_mechanical_rmse for result in scored_results)
        section_count = len(scored_results)
        print(
            f"\n[{label} aggregate over {section_count} scored continuous sections] "
            f"soft-mixture VWAP scored_n={total_n} mean_market_prob={mean_market_prob:.3f}"
        )
        print(
            f"buy_vwap: model_RMSE={math.sqrt(buy_sse / total_n):.6g}  "
            f"model_NLL={buy_nll:.6g}  "
            f"zero_RMSE={math.sqrt(buy_zero_sse / total_n):.6g}  "
            f"zero_NLL={buy_zero_nll:.6g}  "
            f"mechanical_RMSE={math.sqrt(buy_mech_sse / total_n):.6g}  "
            f"wins_vs_zero={buy_zero_wins}/{section_count}  "
            f"wins_vs_mechanical={buy_mech_wins}/{section_count}"
        )
        print(
            f"sell_vwap: model_RMSE={math.sqrt(sell_sse / total_n):.6g}  "
            f"model_NLL={sell_nll:.6g}  "
            f"zero_RMSE={math.sqrt(sell_zero_sse / total_n):.6g}  "
            f"zero_NLL={sell_zero_nll:.6g}  "
            f"mechanical_RMSE={math.sqrt(sell_mech_sse / total_n):.6g}  "
            f"wins_vs_zero={sell_zero_wins}/{section_count}  "
            f"wins_vs_mechanical={sell_mech_wins}/{section_count}"
        )

    if diagnostic_count > 0 and diagnostics:
        print_soft_mixture_error_diagnostics(label, diagnostics, diagnostic_count)


def as_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def print_soft_mixture_error_diagnostics(
    label: str,
    diagnostics: Sequence[dict[str, object]],
    diagnostic_count: int,
) -> None:
    worst = sorted(diagnostics, key=lambda item: as_float(item.get("score")), reverse=True)[
        : max(0, diagnostic_count)
    ]
    if not worst:
        return
    print(f"\n[{label} worst soft-mixture point errors]")
    for idx, item in enumerate(worst, start=1):
        print(
            f"{idx}. ts={item.get('ts')} section={item.get('section')}\n"
            f"   buy: obs={as_float(item.get('obs_buy_delta')):.6g} "
            f"pred={as_float(item.get('pred_buy_delta')):.6g} "
            f"err={as_float(item.get('buy_err')):.6g} "
            f"mech={as_float(item.get('mech_buy_delta')):.6g} "
            f"market={as_float(item.get('market_buy_delta')):.6g} "
            f"point_market={as_float(item.get('point_market_buy_delta')):.6g} "
            f"refresh={as_float(item.get('refresh_buy_delta')):.6g}\n"
            f"   sell: obs={as_float(item.get('obs_sell_delta')):.6g} "
            f"pred={as_float(item.get('pred_sell_delta')):.6g} "
            f"err={as_float(item.get('sell_err')):.6g} "
            f"mech={as_float(item.get('mech_sell_delta')):.6g} "
            f"market={as_float(item.get('market_sell_delta')):.6g} "
            f"point_market={as_float(item.get('point_market_sell_delta')):.6g} "
            f"refresh={as_float(item.get('refresh_sell_delta')):.6g}\n"
            f"   gate: q_market={as_float(item.get('q_market')):.3f} "
            f"point_q_market={as_float(item.get('point_q_market')):.3f} "
            f"market_resp={as_float(item.get('market_resp')):.3f} "
            f"guarded={item.get('market_point_guarded')} "
            f"flow={as_float(item.get('total_flow')):.6g} "
            f"buy_flow={as_float(item.get('buy_volume')):.6g} "
            f"sell_flow={as_float(item.get('sell_volume')):.6g} "
            f"mech_mid_delta={as_float(item.get('mech_mid_delta')):.6g}\n"
            f"   book: prev_spread={as_float(item.get('prev_spread')):.6g} "
            f"obs_spread={as_float(item.get('obs_spread')):.6g} "
            f"prev_buy_shape={as_float(item.get('prev_buy_shape')):.6g} "
            f"prev_sell_shape={as_float(item.get('prev_sell_shape')):.6g}\n"
            f"   log_shape: market_buy={as_float(item.get('market_log_buy_shape_delta')):.6g} "
            f"market_sell={as_float(item.get('market_log_sell_shape_delta')):.6g} "
            f"refresh_buy={as_float(item.get('refresh_log_buy_shape_delta')):.6g} "
            f"refresh_sell={as_float(item.get('refresh_log_sell_shape_delta')):.6g}"
        )


def print_cross_venue_delta_lag_results(
    label: str,
    target_rows: Sequence[MarketRow],
    source_rows: Sequence[MarketRow],
    *,
    target_exchange: str,
    source_exchange: str,
    lags: int,
    include_current: bool,
    min_rows: int,
    train_ratio: float,
    vwap_depth: float,
) -> None:
    aligned_count = len({row.ts for row in target_rows} & {row.ts for row in source_rows})
    if aligned_count < min_rows:
        print(
            f"\n[{label}] cross delta {target_exchange}<-{source_exchange}: "
            f"aligned_rows={aligned_count}, not enough usable rows"
        )
        return

    specs = (
        ("mid", "mid"),
        ("best bid", "best_bid"),
        ("best ask", "best_ask"),
        ("buy VWAP", "buy_vwap"),
        ("sell VWAP", "sell_vwap"),
    )
    for title, kind in specs:
        built = fit_cross_venue_delta_lag_model(
            target_rows,
            source_rows,
            target_kind=kind,
            source_kind=kind,
            lags=lags,
            include_current=include_current,
            min_rows=min_rows,
            train_ratio=train_ratio,
            vwap_depth=vwap_depth,
        )
        if built is None:
            print(f"\n[{label}] {title} cross delta-lag: not enough scored rows")
            continue
        fit, test = built
        coef_text = ", ".join(
            f"{name}={value:.6g}"
            for name, value in zip(fit.feature_names, fit.beta, strict=False)
        )
        corr_text = ", ".join(
            f"{name}={value:.6g}"
            for name, value in zip(fit.feature_names[1:], fit.corr, strict=False)
        )
        t_text = ", ".join(
            f"{name}={value:.3g}"
            for name, value in zip(fit.feature_names, fit.t_stats, strict=False)
        )
        lag_label = f"lags=0..{lags}" if include_current else f"lags=1..{lags}"
        print(
            f"\n[{label}] {target_exchange} {title} delta from {source_exchange} {title} delta lags "
            f"{lag_label} aligned_rows={aligned_count} n={fit.n}"
        )
        print(
            f"in_sample_R2={fit.r2:.6g}  in_sample_RMSE={fit.rmse:.6g}  "
            f"test_R2={test.r2:.6g}  test_RMSE={test.rmse:.6g}  "
            f"test_mean_RMSE={test.baseline_rmse:.6g}  test_zero_RMSE={test.zero_baseline_rmse:.6g}  "
            f"test_dir_acc={test.direction_accuracy:.6g}"
        )
        print(f"corr=[{corr_text}]")
        print(f"beta=[{coef_text}]")
        print(f"t_stats=[{t_text}]")


def print_cross_venue_gated_delta_results(
    label: str,
    target_rows: Sequence[MarketRow],
    source_rows: Sequence[MarketRow],
    *,
    target_exchange: str,
    source_exchange: str,
    gate_percentiles: Sequence[float],
    min_rows: int,
    train_ratio: float,
    vwap_depth: float,
) -> None:
    aligned_count = len({row.ts for row in target_rows} & {row.ts for row in source_rows})
    if aligned_count < min_rows:
        print(
            f"\n[{label}] gated cross delta {target_exchange}<-{source_exchange}: "
            f"aligned_rows={aligned_count}, not enough usable rows"
        )
        return

    specs = (
        ("mid", "mid"),
        ("best bid", "best_bid"),
        ("best ask", "best_ask"),
        ("buy VWAP", "buy_vwap"),
        ("sell VWAP", "sell_vwap"),
    )
    for title, kind in specs:
        ungated = fit_cross_venue_delta_lag_model(
            target_rows,
            source_rows,
            target_kind=kind,
            source_kind=kind,
            lags=0,
            include_current=True,
            min_rows=min_rows,
            train_ratio=train_ratio,
            vwap_depth=vwap_depth,
        )
        candidates = [
            result
            for pct in gate_percentiles
            if (
                result := evaluate_cross_venue_gated_delta_model(
                    target_rows,
                    source_rows,
                    target_kind=kind,
                    source_kind=kind,
                    gate_percentile=pct,
                    min_rows=min_rows,
                    train_ratio=train_ratio,
                    vwap_depth=vwap_depth,
                )
            )
            is not None
        ]
        if ungated is None or not candidates:
            print(f"\n[{label}] {title} gated cross delta: not enough scored rows")
            continue
        _fit, ungated_test = ungated
        best = min(candidates, key=lambda item: item["rmse"])
        all_text = "; ".join(
            (
                f"p{candidate['gate_percentile']:.0f}:"
                f"rmse={candidate['rmse']:.4g},"
                f"act={candidate['test_active_fraction']:.3g},"
                f"b1={candidate['beta1']:.4g}"
            )
            for candidate in candidates
        )
        print(
            f"\n[{label}] {target_exchange} {title} gated same-step delta from {source_exchange} "
            f"aligned_rows={aligned_count} test_n={int(best['test_n'])}"
        )
        print(
            f"best_gate_p={best['gate_percentile']:.0f}  threshold={best['threshold']:.6g}  "
            f"test_active_fraction={best['test_active_fraction']:.6g}  "
            f"gated_RMSE={best['rmse']:.6g}  gated_R2={best['r2']:.6g}  "
            f"zero_RMSE={best['zero_rmse']:.6g}  mean_RMSE={best['mean_rmse']:.6g}  "
            f"ungated_RMSE={ungated_test.rmse:.6g}  ungated_R2={ungated_test.r2:.6g}"
        )
        print(
            f"best_beta=[intercept={best['beta0']:.6g}, nobitex_delta_t={best['beta1']:.6g}]  "
            f"all_gates=[{all_text}]"
        )


def print_quote_refresh_diagnostics(
    label: str,
    rows: Sequence[MarketRow],
    *,
    exchange: str,
    min_rows: int,
    vwap_depth: float,
    sigma_multiplier: float,
) -> None:
    if len(rows) < max(min_rows, 4):
        print(f"\n[{label}] {exchange} quote-refresh diagnostics: rows={len(rows)}, not enough usable rows")
        return

    specs = (
        ("mid", "mid"),
        ("best bid", "best_bid"),
        ("best ask", "best_ask"),
        ("buy VWAP", "buy_vwap"),
        ("sell VWAP", "sell_vwap"),
    )
    eps = 1e-9
    for title, kind in specs:
        observed: list[float] = []
        mechanical: list[float] = []
        total_flow: list[float] = []
        spread_before: list[float] = []
        spread_after: list[float] = []
        bid_depth_before: list[float] = []
        bid_depth_after: list[float] = []
        ask_depth_before: list[float] = []
        ask_depth_after: list[float] = []
        event_row_indices: list[int] = []

        for idx in range(1, len(rows)):
            prev = rows[idx - 1]
            curr = rows[idx]
            obs_delta, mech_delta = price_and_mechanical_delta(prev, curr, kind, vwap_depth=vwap_depth)
            if not math.isfinite(obs_delta) or not math.isfinite(mech_delta):
                continue
            observed.append(obs_delta)
            mechanical.append(mech_delta)
            total_flow.append(curr.buy_volume + curr.sell_volume)
            spread_before.append(prev.spread)
            spread_after.append(curr.spread)
            bid_depth_before.append(prev.bid_depth_5)
            bid_depth_after.append(curr.bid_depth_5)
            ask_depth_before.append(prev.ask_depth_5)
            ask_depth_after.append(curr.ask_depth_5)
            event_row_indices.append(idx)

        if len(observed) < max(min_rows, 4):
            print(f"\n[{label}] {exchange} {title} quote-refresh diagnostics: not enough deltas")
            continue

        obs = np.asarray(observed, dtype=np.float64)
        mech = np.asarray(mechanical, dtype=np.float64)
        flow = np.asarray(total_flow, dtype=np.float64)
        mechanically_quiet = np.abs(mech) <= eps
        quiet_obs = obs[mechanically_quiet]
        if quiet_obs.size >= 3:
            quiet_sigma = float(np.std(quiet_obs))
        else:
            quiet_sigma = float(np.std(obs))
        threshold = max(float(sigma_multiplier) * quiet_sigma, eps)
        large_unmechanical = mechanically_quiet & (np.abs(obs) > threshold)
        zero_flow = flow <= eps
        refresh = large_unmechanical & zero_flow
        large_count = int(np.sum(large_unmechanical))
        refresh_count = int(np.sum(refresh))
        total_count = int(obs.size)
        quiet_count = int(np.sum(mechanically_quiet))
        zero_flow_count = int(np.sum(zero_flow))
        refresh_fraction_of_large = refresh_count / large_count if large_count else float("nan")
        refresh_fraction_all = refresh_count / total_count

        spread_before_arr = np.asarray(spread_before, dtype=np.float64)
        spread_after_arr = np.asarray(spread_after, dtype=np.float64)
        bid_before_arr = np.asarray(bid_depth_before, dtype=np.float64)
        bid_after_arr = np.asarray(bid_depth_after, dtype=np.float64)
        ask_before_arr = np.asarray(ask_depth_before, dtype=np.float64)
        ask_after_arr = np.asarray(ask_depth_after, dtype=np.float64)
        refresh_indices = np.flatnonzero(refresh)
        row_indices = [event_row_indices[int(idx)] for idx in refresh_indices]

        spread_before_q = quantile_summary(spread_before_arr[refresh].tolist())
        spread_after_q = quantile_summary(spread_after_arr[refresh].tolist())
        spread_delta_q = quantile_summary((spread_after_arr[refresh] - spread_before_arr[refresh]).tolist())
        bid_before_q = quantile_summary(bid_before_arr[refresh].tolist())
        bid_after_q = quantile_summary(bid_after_arr[refresh].tolist())
        ask_before_q = quantile_summary(ask_before_arr[refresh].tolist())
        ask_after_q = quantile_summary(ask_after_arr[refresh].tolist())

        gaps: list[float] = []
        for prev_idx, curr_idx in zip(row_indices, row_indices[1:], strict=False):
            gaps.append((rows[curr_idx].ts - rows[prev_idx].ts).total_seconds())
        gap_q = quantile_summary(gaps)
        autocorr = lag1_binary_autocorr(refresh.tolist())
        top_hours = ", ".join(f"{hour:02d}:00={count}" for hour, count in hour_histogram(rows, row_indices)[:5])

        print(f"\n[{label}] {exchange} {title} quote-refresh diagnostic depth={vwap_depth:g}")
        print(
            f"n={total_count}  quiet_mechanical={quiet_count} ({quiet_count / total_count:.3g})  "
            f"zero_flow={zero_flow_count} ({zero_flow_count / total_count:.3g})  "
            f"quiet_sigma={quiet_sigma:.6g}  jump_threshold={threshold:.6g}"
        )
        print(
            f"large_unmechanical={large_count} ({large_count / total_count:.3g})  "
            f"zero_flow_large_unmechanical={refresh_count} ({refresh_fraction_all:.3g} of all, "
            f"{refresh_fraction_of_large:.3g} of large_unmechanical)"
        )
        print(
            f"spread_before[{format_quantiles(spread_before_q)}]  "
            f"spread_after[{format_quantiles(spread_after_q)}]  "
            f"spread_delta[{format_quantiles(spread_delta_q)}]"
        )
        print(
            f"bid_depth5_before[{format_quantiles(bid_before_q)}]  "
            f"bid_depth5_after[{format_quantiles(bid_after_q)}]"
        )
        print(
            f"ask_depth5_before[{format_quantiles(ask_before_q)}]  "
            f"ask_depth5_after[{format_quantiles(ask_after_q)}]"
        )
        print(
            f"refresh_lag1_autocorr={autocorr:.6g}  "
            f"refresh_gap_seconds[{format_quantiles(gap_q)}]  top_utc_hours=[{top_hours}]"
        )


def print_online_cross_venue_window_results(
    label: str,
    target_rows: Sequence[MarketRow],
    source_rows: Sequence[MarketRow],
    *,
    target_exchange: str,
    source_exchange: str,
    min_rows: int,
    decay: float,
    initial_sigma: float,
    vwap_depth: float,
    mechanical_delta_lags: int,
) -> None:
    aligned_count = len({row.ts for row in target_rows} & {row.ts for row in source_rows})
    if aligned_count < min_rows:
        print(
            f"\n[{label}] cross venue {target_exchange}<-{source_exchange}: "
            f"aligned_rows={aligned_count}, not enough usable rows"
        )
        return

    specs = (
        ("best bid current delta", "best_bid"),
        ("best ask current delta", "best_ask"),
        ("buy VWAP current delta", "buy_vwap"),
        ("sell VWAP current delta", "sell_vwap"),
    )
    for title, target_kind in specs:
        built = online_cross_venue_model(
            target_rows,
            source_rows,
            target_kind=target_kind,
            min_rows=min_rows,
            decay=decay,
            initial_sigma=initial_sigma,
            vwap_depth=vwap_depth,
            mechanical_delta_lags=mechanical_delta_lags,
        )
        if built is None:
            print(f"\n[{label}] {title} cross venue: not enough scored rows")
            continue
        result, names = built
        beta_text = ", ".join(
            f"{name}={value:.6g}"
            for name, value in zip(names, result.final_beta, strict=False)
        )
        print(
            f"\n[{label}] {target_exchange} {title} from {source_exchange} features "
            f"decay={decay:g} aligned_rows={aligned_count} scored_n={result.n_scored}"
        )
        print(
            f"model_RMSE={result.rmse:.6g}  "
            f"model_NLL={result.nll:.6g}  "
            f"mean_gaussian_RMSE={result.mean_rmse:.6g}  "
            f"mean_gaussian_NLL={result.mean_nll:.6g}  "
            f"zero_RMSE={result.zero_rmse:.6g}  "
            f"zero_NLL={result.zero_nll:.6g}  "
            f"{target_exchange}_mechanical_RMSE={result.mechanical_rmse:.6g}"
        )
        print(
            f"final_beta=[{beta_text}]  "
            f"final_mean={result.final_mean:.6g}  "
            f"final_sigma={result.final_sigma:.6g}  "
            f"final_mean_sigma={result.final_mean_sigma:.6g}  "
            f"final_zero_sigma={result.final_zero_sigma:.6g}"
        )


def print_online_vwap_mechanical_delta_window_results(
    label: str,
    rows: Sequence[MarketRow],
    *,
    min_rows: int,
    decay: float,
    initial_sigma: float,
    vwap_depth: float,
    inventory_cash_depth: bool,
    flow_spread_features: bool,
    spread_features: bool,
    current_spread_feature: bool,
    current_flow_spread_delta_lag: bool,
    current_flow_prev_book_features: bool,
    signed_total_spread_features: bool,
    current_flow_features: bool,
    mechanical_delta_lags: int,
    delta_price_lags: int,
) -> None:
    if len(rows) < min_rows:
        print(f"\n[{label}] rows={len(rows)}: not enough usable rows")
        return
    specs = (
        ("buy_vwap current delta", "buy_vwap"),
        ("sell_vwap current delta", "sell_vwap"),
    )
    for title, side in specs:
        result = online_vwap_mechanical_delta_model(
            rows,
            side=side,
            min_rows=min_rows,
            decay=decay,
            initial_sigma=initial_sigma,
            vwap_depth=vwap_depth,
            inventory_cash_depth=inventory_cash_depth,
            flow_spread_features=flow_spread_features,
            spread_features=spread_features,
            current_spread_feature=current_spread_feature,
            current_flow_spread_delta_lag=current_flow_spread_delta_lag,
            current_flow_prev_book_features=current_flow_prev_book_features,
            signed_total_spread_features=signed_total_spread_features,
            current_flow_features=current_flow_features,
            mechanical_delta_lags=mechanical_delta_lags,
            delta_price_lags=delta_price_lags,
        )
        if result is None:
            print(f"\n[{label}] {title} online fixed-depth mechanical-delta-only: not enough scored rows")
            continue
        beta_text = ", ".join(f"{value:.6g}" for value in result.final_beta)
        depth_label = "cash/inventory depth" if inventory_cash_depth else f"depth={vwap_depth:g}"
        lag_label = f", mech_lags={mechanical_delta_lags}" if mechanical_delta_lags else ""
        delta_lag_label = f", delta_lags={delta_price_lags}" if delta_price_lags else ""
        if flow_spread_features:
            feature_label = "mechanical+flow/spread"
        elif spread_features:
            feature_label = "mechanical+spread"
        elif current_spread_feature:
            feature_label = "mechanical+spread_t"
        elif current_flow_spread_delta_lag:
            feature_label = "mechanical+flow/spread+delta_lag"
        elif current_flow_prev_book_features:
            feature_label = "mechanical+current-flow+prev-book"
        elif signed_total_spread_features:
            feature_label = "mechanical+signed/total/spread_prev"
        elif current_flow_features:
            feature_label = "mechanical+current-flow"
        else:
            feature_label = "mechanical-delta-only"
        feature_label = f"{feature_label}{lag_label}{delta_lag_label}"
        print(f"\n[{label}] {title} online VWAP {feature_label} {depth_label} decay={decay:g} scored_n={result.n_scored}")
        print(
            f"model_RMSE={result.rmse:.6g}  "
            f"model_NLL={result.nll:.6g}  "
            f"mean_gaussian_RMSE={result.mean_rmse:.6g}  "
            f"mean_gaussian_NLL={result.mean_nll:.6g}  "
            f"zero_RMSE={result.zero_rmse:.6g}  "
            f"zero_NLL={result.zero_nll:.6g}  "
            f"mechanical_RMSE={result.mechanical_rmse:.6g}"
        )
        print(
            f"final_beta=[{beta_text}]  "
            f"final_mean={result.final_mean:.6g}  "
            f"final_sigma={result.final_sigma:.6g}  "
            f"final_mean_sigma={result.final_mean_sigma:.6g}  "
            f"final_zero_sigma={result.final_zero_sigma:.6g}"
        )


def print_vwap_residual_window_results(
    label: str,
    rows: Sequence[MarketRow],
    *,
    lags: int,
    min_rows: int,
    train_ratio: float,
    vwap_depth: float,
) -> None:
    if len(rows) < min_rows:
        print(f"\n[{label}] rows={len(rows)}: not enough usable rows")
        return
    specs = (
        ("buy_vwap residual ~ buy/sell volumes", "buy_vwap_from_flow"),
        ("sell_vwap residual ~ buy/sell volumes", "sell_vwap_from_flow"),
    )
    for title, side in specs:
        result = fit_vwap_residual_model(
            rows,
            side=side,
            lags=lags,
            min_rows=min_rows,
            vwap_depth=vwap_depth,
        )
        split_result = train_test_vwap_residual_model(
            rows,
            side=side,
            lags=lags,
            min_rows=min_rows,
            train_ratio=train_ratio,
            vwap_depth=vwap_depth,
        )
        full_eval = evaluate_vwap_full_model(
            rows,
            side=side,
            lags=lags,
            min_rows=min_rows,
            train_ratio=train_ratio,
            vwap_depth=vwap_depth,
        )
        print(f"\n[{label}] {title} mechanical residual, depth={vwap_depth:g}")
        print_window_result("vwap", rows, result, split_result)
        if full_eval is not None:
            print(
                "full delta test: "
                f"n={full_eval.n}  "
                f"full_R2={full_eval.full_r2:.6g}  "
                f"full_RMSE={full_eval.full_rmse:.6g}  "
                f"mechanical_R2={full_eval.mechanical_r2:.6g}  "
                f"mechanical_RMSE={full_eval.mechanical_rmse:.6g}  "
                f"mean_baseline_RMSE={full_eval.mean_baseline_rmse:.6g}"
            )


def print_vwap_current_step_window_results(
    label: str,
    rows: Sequence[MarketRow],
    *,
    lags: int,
    min_rows: int,
    train_ratio: float,
    vwap_depth: float,
) -> None:
    if len(rows) < min_rows:
        print(f"\n[{label}] rows={len(rows)}: not enough usable rows")
        return
    specs = (
        ("buy_vwap current delta ~ mechanical + buy/sell volumes", "buy_vwap_current"),
        ("sell_vwap current delta ~ mechanical + buy/sell volumes", "sell_vwap_current"),
    )
    for title, side in specs:
        result = fit_vwap_current_step_model(
            rows,
            side=side,
            lags=lags,
            min_rows=min_rows,
            vwap_depth=vwap_depth,
        )
        evals = train_test_vwap_current_step_model(
            rows,
            side=side,
            lags=lags,
            min_rows=min_rows,
            train_ratio=train_ratio,
            vwap_depth=vwap_depth,
        )
        split_result = evals[0] if evals is not None else None
        mechanical_eval = evals[1] if evals is not None else None
        print(f"\n[{label}] {title}, depth={vwap_depth:g}")
        print_window_result("vwap-current", rows, result, split_result)
        if mechanical_eval is not None:
            print(
                "mechanical-only test: "
                f"R2={mechanical_eval.r2:.6g}  "
                f"RMSE={mechanical_eval.rmse:.6g}  "
                f"mean_baseline_RMSE={mechanical_eval.baseline_rmse:.6g}  "
                f"zero_baseline_RMSE={mechanical_eval.zero_baseline_rmse:.6g}"
            )


def print_vwap_book_current_step_window_results(
    label: str,
    rows: Sequence[MarketRow],
    *,
    min_rows: int,
    train_ratio: float,
    vwap_depth: float,
) -> None:
    if len(rows) < min_rows:
        print(f"\n[{label}] rows={len(rows)}: not enough usable rows")
        return
    specs = (
        ("buy_vwap current delta ~ previous ask-book features", "buy_vwap_book_current"),
        ("sell_vwap current delta ~ previous bid-book features", "sell_vwap_book_current"),
    )
    for title, side in specs:
        result = fit_vwap_book_current_step_model(
            rows,
            side=side,
            min_rows=min_rows,
            vwap_depth=vwap_depth,
        )
        split_result = train_test_vwap_book_current_step_model(
            rows,
            side=side,
            min_rows=min_rows,
            train_ratio=train_ratio,
            vwap_depth=vwap_depth,
        )
        print(f"\n[{label}] {title}, depth={vwap_depth:g}")
        print_window_result("vwap-book-current", rows, result, split_result)


def print_vwap_complete_current_step_window_results(
    label: str,
    rows: Sequence[MarketRow],
    *,
    lags: int,
    min_rows: int,
    train_ratio: float,
    vwap_depth: float,
    include_depth_ratios: bool,
) -> None:
    if len(rows) < min_rows:
        print(f"\n[{label}] rows={len(rows)}: not enough usable rows")
        return
    specs = (
        ("buy_vwap current delta ~ mechanical + complete flow/book features", "buy_vwap_complete_current"),
        ("sell_vwap current delta ~ mechanical + complete flow/book features", "sell_vwap_complete_current"),
    )
    for title, side in specs:
        result = fit_vwap_complete_current_step_model(
            rows,
            side=side,
            lags=lags,
            min_rows=min_rows,
            vwap_depth=vwap_depth,
            include_depth_ratios=include_depth_ratios,
        )
        evals = train_test_vwap_complete_current_step_model(
            rows,
            side=side,
            lags=lags,
            min_rows=min_rows,
            train_ratio=train_ratio,
            vwap_depth=vwap_depth,
            include_depth_ratios=include_depth_ratios,
        )
        split_result = evals[0] if evals is not None else None
        mechanical_eval = evals[1] if evals is not None else None
        ratio_label = "with depth ratios" if include_depth_ratios else "without depth ratios"
        print(f"\n[{label}] {title}, lags={lags}, depth={vwap_depth:g}, {ratio_label}")
        print_window_result("vwap-complete-current", rows, result, split_result)
        if mechanical_eval is not None:
            print(
                "mechanical-only test: "
                f"R2={mechanical_eval.r2:.6g}  "
                f"RMSE={mechanical_eval.rmse:.6g}  "
                f"mean_baseline_RMSE={mechanical_eval.baseline_rmse:.6g}  "
                f"zero_baseline_RMSE={mechanical_eval.zero_baseline_rmse:.6g}"
            )


def safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def parse_levels(orderbook: object, side: str) -> list[tuple[float, float]]:
    if orderbook is None:
        return []
    if isinstance(orderbook, str):
        try:
            orderbook = json.loads(orderbook)
        except json.JSONDecodeError:
            return []
    if not isinstance(orderbook, dict):
        return []
    raw_levels = orderbook.get(side) or []
    levels: list[tuple[float, float]] = []
    for level in raw_levels:
        if not isinstance(level, dict):
            continue
        price = safe_float(level.get("price"))
        qty = safe_float(level.get("qty"))
        if math.isfinite(price) and math.isfinite(qty) and price > 0 and qty >= 0:
            levels.append((price, qty))
    if side == "bids":
        return sorted(levels, key=lambda item: item[0], reverse=True)
    return sorted(levels, key=lambda item: item[0])


def pearson(x: Iterable[float], y: Iterable[float]) -> float:
    x_arr = np.asarray(list(x), dtype=np.float64)
    y_arr = np.asarray(list(y), dtype=np.float64)
    valid = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[valid]
    y_arr = y_arr[valid]
    if x_arr.size < 2:
        return float("nan")
    x_centered = x_arr - float(np.mean(x_arr))
    y_centered = y_arr - float(np.mean(y_arr))
    denom = math.sqrt(float(x_centered @ x_centered) * float(y_centered @ y_centered))
    if denom <= 0:
        return float("nan")
    return float((x_centered @ y_centered) / denom)


if __name__ == "__main__":
    main()
