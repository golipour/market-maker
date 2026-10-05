/** Field metadata for preset / override advanced forms. */

import { cloneConfig, getAt, setAt, type JsonObject } from "../lib/presetConfig";

export type FieldType = "number" | "text" | "boolean" | "select";

export type FieldOption = { value: string; label: string };

export type FieldWhen = {
  path: string;
  equals?: string | number | boolean;
  in?: ReadonlyArray<string | number | boolean>;
};

export type FieldDef = {
  path: string;
  label: string;
  type: FieldType;
  help?: string;
  defaultValue?: unknown;
  options?: readonly FieldOption[];
  step?: string;
  min?: number;
  max?: number;
  when?: FieldWhen | FieldWhen[];
};

export type SectionDef = {
  id: string;
  title: string;
  help?: string;
  fields: readonly FieldDef[];
};

function opt(values: readonly string[]): FieldOption[] {
  return values.map((value) => ({ value, label: value }));
}

const PRICE_LOCAL_KINDS = ["residual", "conditional_mechanical_mixture"] as const;
const PRICE_GLOBAL_KINDS = ["ewma_gaussian", "conditional_mechanical_mixture", "residual"] as const;
const FLOW_KINDS = [
  "hurdle_lognormal",
  "hurdle_gamma",
  "hurdle_weibull",
  "gamma",
  "weibull",
] as const;

function modelKindWhen(path: string, kinds: readonly string[]): FieldWhen {
  return { path, in: [...kinds] };
}

function venueOptimizerFields(prefix: "local" | "global"): FieldDef[] {
  const base = `controller.optimizer.${prefix}`;
  const fields: FieldDef[] = [
    {
      path: `${base}.n_components`,
      label: "Components",
      type: "number",
      defaultValue: prefix === "local" ? 4 : 2,
      help: "Sigmoid mixture components for depth curves.",
    },
    {
      path: `${base}.objective_weight`,
      label: "Objective weight",
      type: "number",
      defaultValue: 1.0,
      step: "0.01",
    },
    {
      path: `${base}.return_risk_aversion_beta`,
      label: "Return risk aversion β",
      type: "number",
      defaultValue: 0.1,
      step: "0.01",
      help: "Penalty on return variance in the objective.",
    },
    {
      path: `${base}.cumulative_qty_floor_at_spread_max`,
      label: "Qty floor at spread max",
      type: "number",
      defaultValue: 1.0,
      step: "0.01",
      help: "Decoder total base qty floor at the outer spread (per side).",
    },
    {
      path: `${base}.cumulative_qty_cap_at_spread_max`,
      label: "Qty cap at spread max",
      type: "number",
      step: "0.01",
      min: 0,
      help: "Hard cap on total base qty per side at the outer spread. Leave empty for no cap (wallet only). Units: base asset (e.g. USDT), not IRT.",
    },
    {
      path: `${base}.validity.min_score`,
      label: "Validity min score",
      type: "number",
      defaultValue: 0.3,
      step: "0.01",
      help: "Minimum book-quality score before validity penalty.",
    },
    {
      path: `${base}.validity.penalty_scale`,
      label: "Validity penalty scale",
      type: "number",
      defaultValue: 100.0,
      step: "1",
    },
    {
      path: `${base}.viability.min_viable_quote`,
      label: "Min viable quote (IRT)",
      type: "number",
      defaultValue: 50_000,
      min: 0,
      help: "Per-side wallet floor q_min in quote currency. Deficit is zero above this; the penalty only kicks in below it.",
    },
    {
      path: `${base}.viability.recovery_fraction`,
      label: "Recovery fraction ρ",
      type: "number",
      defaultValue: 0.0,
      step: "0.01",
      min: 0,
      max: 1,
      help: "Desired fractional reduction of reserve deficit per step (0 = disabled). When positive, penalizes insufficient deficit recovery.",
    },
    {
      path: `${base}.viability.shortfall_tolerance`,
      label: "Shortfall tolerance ε",
      type: "number",
      defaultValue: 0.05,
      step: "0.01",
      min: 0,
      help: "Dual slack: λ rises when EWMA quadratic deficit S̄ exceeds ε, and falls when S̄ is below ε.",
    },
    {
      path: `${base}.viability.lambda_init`,
      label: "λ init",
      type: "number",
      defaultValue: 0.0,
      step: "0.01",
      min: 0,
      help: "Starting linear multiplier, clipped to [0, λ_max]. This is the actual λ, not log-space θ.",
    },
    {
      path: `${base}.viability.lambda_learning_rate`,
      label: "λ learning rate η",
      type: "number",
      defaultValue: 0.02,
      step: "0.001",
      min: 0,
      help: "Dual step size. λ ← clip(λ + η (S̄ − ε), 0, λ_max). Keep small; the quadratic deficit already supplies a steep gradient.",
    },
    {
      path: `${base}.viability.lambda_max`,
      label: "λ max",
      type: "number",
      defaultValue: 100.0,
      step: "1",
      min: 0,
      help: "Hard roof on λ. Strong enough to widen spread; cannot explode like e^θ.",
    },
    {
      path: `${base}.viability.dual_ewma_decay`,
      label: "Dual EWMA decay α",
      type: "number",
      defaultValue: 0.9,
      step: "0.01",
      min: 0,
      help: "Smoothing on S before the dual step. 0 = use raw frame shortfall; 0.9 ≈ ~10-frame memory. Must be in [0, 1).",
    },
    {
      path: `${base}.book_quality.soft_scoring.tightness_spread_scale`,
      label: "Tightness spread scale",
      type: "number",
      defaultValue: prefix === "local" ? 200 : 100,
      help: "Spread scale for soft tightness scoring.",
    },
    {
      path: `${base}.book_quality.soft_scoring.presence_reference_qty`,
      label: "Presence reference qty",
      type: "number",
      defaultValue: 1.0,
      step: "0.01",
    },
    {
      path: `${base}.book_quality.soft_scoring.measure`,
      label: "Soft measure",
      type: "select",
      defaultValue: "quoted",
      options: opt(["quoted", "combined"]),
      help: "quoted = maker book only; combined includes liquidity side.",
    },
    {
      path: `${base}.book_quality.hard_structure.action_max_spread`,
      label: "Hard max spread",
      type: "number",
      defaultValue: 100,
      help: "Hard structure cap on action spread (global venue typically).",
    },
    {
      path: `${base}.book_quality.hard_structure.action_min_depth`,
      label: "Hard min depth",
      type: "number",
      defaultValue: 0.3,
      step: "0.01",
    },
    {
      path: `${base}.book_quality.hard_structure.action_k_epsilon`,
      label: "Hard k ε",
      type: "number",
      defaultValue: 0.01,
      step: "0.001",
    },
    {
      path: `${base}.soft_fill_temp`,
      label: "Soft-fill temperature",
      type: "number",
      defaultValue: 0.1,
      step: "0.01",
    },
    {
      path: `${base}.soft_fill_activation`,
      label: "Soft-fill activation",
      type: "select",
      defaultValue: "hardsigmoid",
      options: opt(["sigmoid", "hardsigmoid"]),
    },
    {
      path: `${base}.soft_fill_queue_model`,
      label: "Soft-fill queue model",
      type: "select",
      defaultValue: "back_of_queue",
      options: opt(["back_of_queue", "pro_rata"]),
    },
  ];
  if (prefix === "local") {
    fields.splice(
      fields.findIndex((f) => f.path.includes("soft_scoring.measure")) + 1,
      0,
      {
        path: `${base}.reservation_skew.max_abs_skew`,
        label: "Max abs reservation skew",
        type: "number",
        defaultValue: 0.002,
        step: "0.0001",
        help: "Local-only reservation price skew cap.",
      },
    );
  }
  return fields;
}

function priceModelFields(
  prefix: "local_price_model" | "global_price_model",
  kinds: readonly string[],
): FieldDef[] {
  const base = `controller.${prefix}`;
  return [
    {
      path: `${base}.kind`,
      label: "Kind",
      type: "select",
      defaultValue: kinds[0],
      options: opt(kinds),
    },
    {
      path: `${base}.decay`,
      label: "Decay",
      type: "number",
      defaultValue: 0.99,
      step: "0.001",
      when: modelKindWhen(`${base}.kind`, kinds),
    },
    {
      path: `${base}.initial_sigma`,
      label: "Initial σ",
      type: "number",
      defaultValue: 50,
      when: modelKindWhen(`${base}.kind`, ["residual", "conditional_mechanical_mixture"]),
    },
    {
      path: `${base}.reference_vwap_vol`,
      label: "Reference VWAP vol",
      type: "number",
      defaultValue: prefix === "local_price_model" ? 1000 : 100,
      help: "Fixed-depth volume for mechanical mixture shape features.",
      when: modelKindWhen(`${base}.kind`, ["conditional_mechanical_mixture"]),
    },
    {
      path: `${base}.shape_alpha`,
      label: "Shape α",
      type: "number",
      defaultValue: 0.5,
      step: "0.01",
      when: modelKindWhen(`${base}.kind`, ["conditional_mechanical_mixture"]),
    },
    {
      path: `${base}.initial_log_sigma`,
      label: "Initial log σ",
      type: "number",
      defaultValue: 0.25,
      step: "0.01",
      when: modelKindWhen(`${base}.kind`, ["conditional_mechanical_mixture"]),
    },
    {
      path: `${base}.student_df`,
      label: "Student-t df",
      type: "number",
      defaultValue: 3,
      when: modelKindWhen(`${base}.kind`, ["conditional_mechanical_mixture"]),
    },
    {
      path: `${base}.min_refresh_weight`,
      label: "Min refresh weight",
      type: "number",
      defaultValue: 0.15,
      step: "0.01",
      help: "Floor on refresh Student-t weight to prevent gate saturation under heavy flow.",
      when: modelKindWhen(`${base}.kind`, ["conditional_mechanical_mixture"]),
    },
    {
      path: `${base}.point_mode`,
      label: "Point mode",
      type: "select",
      defaultValue: "sample",
      options: opt(["sample", "mean"]),
      when: modelKindWhen(`${base}.kind`, ["conditional_mechanical_mixture"]),
    },
    {
      path: `${base}.sigma_buy`,
      label: "σ buy (EWMA)",
      type: "number",
      defaultValue: 1,
      when: modelKindWhen(`${base}.kind`, ["ewma_gaussian"]),
    },
    {
      path: `${base}.sigma_sell`,
      label: "σ sell (EWMA)",
      type: "number",
      defaultValue: 1,
      when: modelKindWhen(`${base}.kind`, ["ewma_gaussian"]),
    },
  ];
}

function flowModelFields(prefix: "local_market_flow_model" | "global_market_flow_model"): FieldDef[] {
  const base = `controller.${prefix}`;
  return [
    {
      path: `${base}.kind`,
      label: "Kind",
      type: "select",
      defaultValue: "hurdle_lognormal",
      options: opt([...FLOW_KINDS]),
    },
    {
      path: `${base}.decay`,
      label: "Decay",
      type: "number",
      defaultValue: 0.95,
      step: "0.001",
    },
    {
      path: `${base}.init_p0`,
      label: "Init p0",
      type: "number",
      defaultValue: 0.3,
      step: "0.01",
      when: modelKindWhen(`${base}.kind`, ["hurdle_lognormal", "hurdle_gamma", "hurdle_weibull"]),
    },
    {
      path: `${base}.init_mu`,
      label: "Init μ",
      type: "number",
      defaultValue: 0,
      when: modelKindWhen(`${base}.kind`, ["hurdle_lognormal"]),
    },
    {
      path: `${base}.init_sigma`,
      label: "Init σ",
      type: "number",
      defaultValue: 1,
      when: modelKindWhen(`${base}.kind`, ["hurdle_lognormal"]),
    },
    {
      path: `${base}.init_shape`,
      label: "Init shape",
      type: "number",
      defaultValue: 1,
      when: modelKindWhen(`${base}.kind`, ["hurdle_gamma", "gamma"]),
    },
    {
      path: `${base}.init_scale`,
      label: "Init scale",
      type: "number",
      defaultValue: 1,
      when: modelKindWhen(`${base}.kind`, ["hurdle_gamma", "gamma"]),
    },
    {
      path: `${base}.init_k`,
      label: "Init Weibull k",
      type: "number",
      defaultValue: 2,
      when: modelKindWhen(`${base}.kind`, ["hurdle_weibull", "weibull"]),
    },
    {
      path: `${base}.init_lambda`,
      label: "Init Weibull λ",
      type: "number",
      defaultValue: 1,
      when: modelKindWhen(`${base}.kind`, ["hurdle_weibull", "weibull"]),
    },
  ];
}

function riskSideFields(side: "local" | "global"): FieldDef[] {
  const base = `controller.risk.${side === "local" ? "local" : "global"}`;
  const invMax = side === "local" ? 10_000 : 1_000;
  const notional = side === "local" ? 1_000_000_000 : 100_000_000;
  return [
    {
      path: `${base}.min_inventory`,
      label: `${side} min inventory`,
      type: "number",
      defaultValue: 0,
    },
    {
      path: `${base}.max_inventory`,
      label: `${side} max inventory`,
      type: "number",
      defaultValue: invMax,
    },
    {
      path: `${base}.min_cash`,
      label: `${side} min cash`,
      type: "number",
      defaultValue: 0,
    },
    {
      path: `${base}.max_open_notional`,
      label: `${side} max open notional`,
      type: "number",
      defaultValue: notional,
    },
    {
      path: `${base}.max_order_notional`,
      label: `${side} max order notional`,
      type: "number",
      defaultValue: notional,
    },
    {
      path: `${base}.max_market_data_age_ms`,
      label: `${side} max market age (ms)`,
      type: "number",
      defaultValue: 60_000,
    },
    {
      path: `${base}.max_private_state_age_ms`,
      label: `${side} max private age (ms)`,
      type: "number",
      defaultValue: 120_000,
    },
  ];
}

export const PRESET_SECTIONS: readonly SectionDef[] = [
  {
    id: "timing",
    title: "Timing",
    help: "Controller cadence and market-data freshness gates.",
    fields: [
      {
        path: "interval_ms",
        label: "Interval (ms)",
        type: "number",
        defaultValue: 10_000,
        help: "Decision interval between frames.",
      },
      {
        path: "max_global_lag_ms",
        label: "Max global lag (ms)",
        type: "number",
        defaultValue: 2_000,
      },
      {
        path: "global_grace_ms",
        label: "Global grace (ms)",
        type: "number",
        defaultValue: 250,
      },
      {
        path: "max_local_market_age_ms",
        label: "Max local market age (ms)",
        type: "number",
        defaultValue: 10_000,
      },
      {
        path: "max_global_market_age_ms",
        label: "Max global market age (ms)",
        type: "number",
        defaultValue: 10_000,
      },
    ],
  },
  {
    id: "liquidity",
    title: "Liquidity sourcing",
    fields: [
      {
        path: "liquidity_sourcing.mode",
        label: "Mode",
        type: "select",
        defaultValue: "off",
        options: opt(["off", "simulate", "live"]),
        help: "off = observe only; simulate = model/objective only; live = allow liquidity orders.",
      },
    ],
  },
  {
    id: "pricing",
    title: "Pricing",
    fields: [
      {
        path: "pricing.quote_anchor.mode",
        label: "Quote anchor mode",
        type: "select",
        defaultValue: "liquidity_mid",
        options: opt(["quoting_mid", "liquidity_mid", "liquidity_weighted_mid"]),
        help: "Centers maker quotes. liquidity_weighted_mid uses VWAP inside anchor_vwap_bps.",
      },
      {
        path: "pricing.quote_anchor.anchor_vwap_bps",
        label: "Anchor VWAP (bps)",
        type: "number",
        defaultValue: 100,
      },
      {
        path: "pricing.inventory_mark.venue",
        label: "Inventory mark venue",
        type: "select",
        defaultValue: "liquidity",
        options: opt(["quoting", "liquidity"]),
        help: "Venue whose inventory-depth VWAP marks wealth.",
      },
    ],
  },
  {
    id: "execution",
    title: "Execution",
    fields: [
      {
        path: "execution.quoting.enabled",
        label: "Quoting enabled",
        type: "boolean",
        defaultValue: true,
      },
      {
        path: "execution.quoting.planner",
        label: "Quoting planner",
        type: "select",
        defaultValue: "replace",
        options: opt(["reconcile", "replace"]),
      },
      {
        path: "execution.quoting.max_orders_per_side",
        label: "Quoting max orders / side",
        type: "number",
        defaultValue: 10,
      },
      {
        path: "controller.reconcile.price_tol_bps",
        label: "Reconcile price tolerance (bps)",
        type: "number",
        defaultValue: 0,
        step: "0.1",
        help: "Max price difference in basis points to keep a resting order in reconcile mode (0 = exact match).",
      },
      {
        path: "controller.reconcile.qty_tol_rel",
        label: "Reconcile quantity tolerance (rel)",
        type: "number",
        defaultValue: 0,
        step: "0.01",
        help: "Max relative difference (e.g. 0.05 = 5%) in quantity to keep a resting order in reconcile mode.",
      },
      {
        path: "execution.liquidity.planner",
        label: "Liquidity planner",
        type: "select",
        defaultValue: "replace",
        options: opt(["reconcile", "replace"]),
        help: "Live liquidity execution is gated by liquidity_sourcing.mode = live.",
      },
      {
        path: "execution.liquidity.max_orders_per_side",
        label: "Liquidity max orders / side",
        type: "number",
        defaultValue: 2,
      },
    ],
  },
  {
    id: "controller",
    title: "Controller",
    fields: [
      {
        path: "controller.enabled",
        label: "Enabled",
        type: "boolean",
        defaultValue: true,
      },
      {
        path: "controller.dry_run",
        label: "Dry run",
        type: "boolean",
        defaultValue: false,
        help: "When true, decisions run but live order sync is blocked.",
      },
      {
        path: "controller.execution_after_steps",
        label: "Execution after steps",
        type: "number",
        defaultValue: 10,
        help: "Warmup steps before live execution.",
      },
      {
        path: "controller.negative_objective_action",
        label: "Negative objective action",
        type: "select",
        defaultValue: "optimizer_action",
        options: opt(["optimizer_action", "floor_action", "zero_action"]),
        help: "When CE is weak: keep optimizer output, floor depth, or pull quotes (zero_action).",
      },
      {
        path: "controller.negative_objective_zero_threshold",
        label: "Zero-action CE threshold",
        type: "number",
        defaultValue: 0,
        step: "0.01",
        help: "Pull quotes when CE < -threshold (zero_action mode).",
      },
      {
        path: "controller.decision_timeout_ms",
        label: "Decision timeout (ms)",
        type: "number",
        defaultValue: 3_000,
      },
      {
        path: "controller.placement_timeout_ms",
        label: "Placement timeout (ms)",
        type: "number",
        defaultValue: 7_000,
      },
      {
        path: "controller.placement_max_concurrency",
        label: "Placement max concurrency",
        type: "number",
        defaultValue: 32,
      },
      {
        path: "controller.decisions_topic",
        label: "Decisions topic",
        type: "text",
        defaultValue: "strategy:decisions",
      },
      {
        path: "controller.action_topic",
        label: "Action topic",
        type: "text",
        defaultValue: "strategy:actions",
      },
      {
        path: "controller.orderbook_grid.spread_step",
        label: "MPC spread step",
        type: "number",
        defaultValue: 20,
        help: "Internal MPC grid step (quote-per-base). Controls soft-fill / book bucketing — not live order spacing.",
      },
      {
        path: "controller.orderbook_grid.spread_max",
        label: "MPC spread max",
        type: "number",
        defaultValue: 200,
        help: "Internal MPC outer edge. Execution ladder must stay within this domain.",
      },
      {
        path: "controller.execution_grid.mode",
        label: "Execution grid mode",
        type: "select",
        defaultValue: "uniform",
        options: opt(["uniform", "log", "hybrid", "explicit"]),
        help: "How to sample continuous D(δ) into live orders. Default uniform mirrors the MPC grid.",
      },
      {
        path: "controller.execution_grid.spread_max",
        label: "Execution spread max",
        type: "number",
        help: "Optional outer edge ≤ MPC spread max. Empty = use MPC spread max.",
        when: { path: "controller.execution_grid.mode", in: ["uniform", "log", "hybrid", "explicit"] },
      },
      {
        path: "controller.execution_grid.spread_step",
        label: "Execution spread step",
        type: "number",
        help: "Linear step for uniform mode. Empty = use MPC spread step.",
        when: { path: "controller.execution_grid.mode", equals: "uniform" },
      },
      {
        path: "controller.execution_grid.levels",
        label: "Execution levels",
        type: "number",
        defaultValue: 8,
        min: 2,
        help: "Point count for log mode, or outer-band point count for hybrid.",
        when: { path: "controller.execution_grid.mode", in: ["log", "hybrid"] },
      },
      {
        path: "controller.execution_grid.spread_min",
        label: "Execution spread min",
        type: "number",
        help: "Positive lower bound for geometric spacing (required for log; optional join for hybrid).",
        when: { path: "controller.execution_grid.mode", in: ["log", "hybrid"] },
      },
      {
        path: "controller.execution_grid.inner_linear_max",
        label: "Hybrid inner linear max",
        type: "number",
        help: "End of the near-mid linear band for hybrid mode.",
        when: { path: "controller.execution_grid.mode", equals: "hybrid" },
      },
      {
        path: "controller.execution_grid.inner_step",
        label: "Hybrid inner step",
        type: "number",
        help: "Linear step inside the hybrid inner band. Empty = use MPC spread step.",
        when: { path: "controller.execution_grid.mode", equals: "hybrid" },
      },
    ],
  },
  {
    id: "risk",
    title: "Risk",
    help: "Hard pre-execution gates. local = quoting exchange, global = liquidity/reference.",
    fields: [
      {
        path: "controller.risk.enabled",
        label: "Risk enabled",
        type: "boolean",
        defaultValue: true,
      },
      {
        path: "controller.risk.engine",
        label: "Risk engine",
        type: "select",
        defaultValue: "net_delta",
        options: opt(["net_delta"]),
      },
      ...riskSideFields("local"),
      ...riskSideFields("global"),
    ],
  },
  {
    id: "local_price",
    title: "Local price model",
    fields: priceModelFields("local_price_model", PRICE_LOCAL_KINDS),
  },
  {
    id: "global_price",
    title: "Global price model",
    fields: priceModelFields("global_price_model", PRICE_GLOBAL_KINDS),
  },
  {
    id: "local_flow",
    title: "Local market flow model",
    fields: flowModelFields("local_market_flow_model"),
  },
  {
    id: "global_flow",
    title: "Global market flow model",
    fields: flowModelFields("global_market_flow_model"),
  },
  {
    id: "optimizer",
    title: "Optimizer",
    help: "Do not put helper _comment keys inside optimizer.local / optimizer.global.",
    fields: [
      {
        path: "controller.optimizer.kind",
        label: "Kind",
        type: "select",
        defaultValue: "monte_carlo_mpc_tensor_batched",
        options: opt(["monte_carlo_mpc_tensor_batched"]),
      },
      {
        path: "controller.optimizer.n_samples",
        label: "MC samples",
        type: "number",
        defaultValue: 64,
      },
      {
        path: "controller.optimizer.fixed_mc_samples",
        label: "Fixed MC samples",
        type: "boolean",
        defaultValue: true,
      },
      {
        path: "controller.optimizer.compile_soft_fill",
        label: "Compile soft fill (torch.compile)",
        type: "boolean",
        defaultValue: true,
        help: "Fuses the soft-fill chain with torch.compile (Inductor). Default true. The compile cost (3-5 s cold, ~1 s warm cache) is paid once at construction during startup warm-up, never inside a decide, so it does not consume decision_timeout_ms; the first decide afterwards measures ~200 ms. Set false to force eager. Any compile failure falls back to eager automatically. MM_TORCH_COMPILE=0 forces eager globally.",
      },
      {
        path: "controller.optimizer.horizon",
        label: "Horizon",
        type: "number",
        defaultValue: 2,
      },
      {
        path: "controller.optimizer.adam_epochs",
        label: "Adam epochs",
        type: "number",
        defaultValue: 40,
      },
      {
        path: "controller.optimizer.adam_lr",
        label: "Adam learning rate",
        type: "number",
        defaultValue: 1,
        step: "0.01",
      },
      {
        path: "controller.optimizer.device",
        label: "Device",
        type: "select",
        defaultValue: "cpu",
        options: opt(["cpu", "cuda"]),
      },
      {
        path: "controller.optimizer.warm_start.enabled",
        label: "Warm start enabled",
        type: "boolean",
        defaultValue: true,
      },
      {
        path: "controller.optimizer.warm_start.preserve_optimizer_state",
        label: "Preserve optimizer state",
        type: "boolean",
        defaultValue: true,
      },
      {
        path: "controller.optimizer.warm_start.alpha",
        label: "Warm-start α",
        type: "number",
        defaultValue: 0.8,
        step: "0.01",
      },
      {
        path: "controller.optimizer.warm_start.alpha_mode",
        label: "α mode",
        type: "select",
        defaultValue: "fixed",
        options: opt(["fixed", "adaptive"]),
      },
      {
        path: "controller.optimizer.warm_start.alpha_max",
        label: "α max",
        type: "number",
        defaultValue: 0.95,
        step: "0.01",
      },
      {
        path: "controller.optimizer.warm_start.grad_norm_eps",
        label: "Grad norm ε",
        type: "number",
        defaultValue: 1e-12,
        step: "1e-12",
      },
      {
        path: "controller.optimizer.objective_normalize.enabled",
        label: "Objective normalize",
        type: "boolean",
        defaultValue: true,
      },
      {
        path: "controller.optimizer.objective_normalize.ce_ewma_decay",
        label: "CE EWMA decay",
        type: "number",
        defaultValue: 0.9,
        step: "0.01",
      },
    ],
  },
  {
    id: "optimizer_local",
    title: "Optimizer — local venue",
    fields: venueOptimizerFields("local"),
  },
  {
    id: "optimizer_global",
    title: "Optimizer — global venue",
    fields: venueOptimizerFields("global"),
  },
];

export const ALL_FIELD_PATHS: readonly string[] = PRESET_SECTIONS.flatMap((section) =>
  section.fields.map((field) => field.path),
);

export const FIELD_BY_PATH: ReadonlyMap<string, FieldDef> = new Map(
  ALL_FIELD_PATHS.map((path) => {
    const field = PRESET_SECTIONS.flatMap((s) => s.fields).find((f) => f.path === path)!;
    return [path, field];
  }),
);

export function fieldVisible(field: FieldDef, config: unknown): boolean {
  if (!field.when) {
    return true;
  }
  const conditions = Array.isArray(field.when) ? field.when : [field.when];
  return conditions.every((cond) => {
    const actual = getAt(config, cond.path);
    if (cond.in) {
      return cond.in.includes(actual as string | number | boolean);
    }
    if ("equals" in cond) {
      return actual === cond.equals;
    }
    return true;
  });
}

/** Apply schema defaults for missing known paths (UI display only). */
export function applySchemaDefaults(config: JsonObject): JsonObject {
  let out = cloneConfig(config);
  for (const field of FIELD_BY_PATH.values()) {
    if (field.defaultValue === undefined) {
      continue;
    }
    if (getAt(out, field.path) === undefined) {
      out = setAt(out, field.path, field.defaultValue);
    }
  }
  return out;
}
