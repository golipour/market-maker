/** Browser-persisted Grafana embed scope for the Observability tab. */

export const OBSERVABILITY_GRAFANA_STORAGE_KEY = "mm.observability.grafana.v1";

export const TIME_RANGES = [
  { value: "now-15m", label: "Last 15m" },
  { value: "now-1h", label: "Last 1h" },
  { value: "now-6h", label: "Last 6h" },
  { value: "now-24h", label: "Last 24h" },
  { value: "now-7d", label: "Last 7d" },
] as const;

export const SPREAD_MAX_OPTIONS = ["2000", "5000", "10000", "20000", "500", "200", "100"] as const;
export const SPREAD_BUCKET_OPTIONS = ["20", "40", "60", "80"] as const;
export const TARGET_FRAME_MODES = [
  { value: "latest", label: "Latest" },
  { value: "pinned", label: "Pinned" },
] as const;

export const FRAME_SCOPE_DASHBOARD_UIDS = new Set([
  "mm-obs-v2-orderbook-depth",
  "mm-obs-v2-target-order-stems",
]);

const ALL = "$__all";

export type GrafanaScopeState = {
  activeUid: string | null;
  from: string;
  to: string;
  exchange: string;
  symbol: string;
  actor_id: string;
  interval_ms: string;
  run_id: string;
  spread_max: string;
  spread_buckets: string;
  target_frame_mode: string;
};

export const DEFAULT_GRAFANA_SCOPE: GrafanaScopeState = {
  activeUid: null,
  from: "now-1h",
  to: "now",
  exchange: ALL,
  symbol: ALL,
  actor_id: ALL,
  interval_ms: ALL,
  run_id: ALL,
  spread_max: "2000",
  spread_buckets: "20",
  target_frame_mode: "latest",
};

const STRING_KEYS = [
  "activeUid",
  "from",
  "to",
  "exchange",
  "symbol",
  "actor_id",
  "interval_ms",
  "run_id",
  "spread_max",
  "spread_buckets",
  "target_frame_mode",
] as const;

function asNonEmptyString(value: unknown): string | null {
  if (typeof value !== "string") {
    return null;
  }
  const trimmed = value.trim();
  return trimmed ? trimmed : null;
}

export function parseGrafanaScope(raw: unknown): GrafanaScopeState {
  const input = raw && typeof raw === "object" ? (raw as Record<string, unknown>) : {};
  const next: GrafanaScopeState = { ...DEFAULT_GRAFANA_SCOPE };
  for (const key of STRING_KEYS) {
    const value = asNonEmptyString(input[key]);
    if (value !== null) {
      next[key] = value;
    }
  }
  return next;
}

export function loadGrafanaScope(storage: Pick<Storage, "getItem"> | null | undefined): GrafanaScopeState {
  if (!storage) {
    return { ...DEFAULT_GRAFANA_SCOPE };
  }
  try {
    const raw = storage.getItem(OBSERVABILITY_GRAFANA_STORAGE_KEY);
    if (!raw) {
      return { ...DEFAULT_GRAFANA_SCOPE };
    }
    return parseGrafanaScope(JSON.parse(raw) as unknown);
  } catch {
    return { ...DEFAULT_GRAFANA_SCOPE };
  }
}

export function saveGrafanaScope(
  state: GrafanaScopeState,
  storage: Pick<Storage, "setItem"> | null | undefined,
): void {
  if (!storage) {
    return;
  }
  try {
    storage.setItem(OBSERVABILITY_GRAFANA_STORAGE_KEY, JSON.stringify(state));
  } catch {
    /* ignore quota / private-mode failures */
  }
}

const VAR_PARAMS = [
  ["var-exchange", "exchange"],
  ["var-symbol", "symbol"],
  ["var-actor_id", "actor_id"],
  ["var-interval_ms", "interval_ms"],
  ["var-run_id", "run_id"],
  ["var-spread_max", "spread_max"],
  ["var-spread_buckets", "spread_buckets"],
  ["var-target_frame_mode", "target_frame_mode"],
] as const;

export function applyGrafanaScope(url: string, scope: GrafanaScopeState): string {
  let parsed: URL;
  try {
    parsed = new URL(url);
  } catch {
    return url;
  }
  parsed.searchParams.set("from", scope.from);
  parsed.searchParams.set("to", scope.to);
  for (const [param, key] of VAR_PARAMS) {
    parsed.searchParams.delete(param);
    parsed.searchParams.set(param, scope[key]);
  }
  return parsed.toString();
}

export function usesFrameScope(uid: string | null | undefined): boolean {
  return Boolean(uid && FRAME_SCOPE_DASHBOARD_UIDS.has(uid));
}
