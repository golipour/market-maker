import type { AccountCredentials, AuthMode, ExchangeAuthSpec } from "../lib/accountCredentials";

export type { AccountCredentials, AuthMode, ExchangeAuthSpec };

const TOKEN_KEY = "MM_ADMIN_TOKEN";

export function getToken(): string {
  return localStorage.getItem(TOKEN_KEY) ?? "";
}

export function setToken(token: string): void {
  if (token.trim()) {
    localStorage.setItem(TOKEN_KEY, token.trim());
  } else {
    localStorage.removeItem(TOKEN_KEY);
  }
}

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  if (!headers.has("Content-Type") && init.body) {
    headers.set("Content-Type", "application/json");
  }
  const token = getToken();
  if (token) {
    headers.set("Authorization", `Bearer ${token}`);
  }
  let response: Response;
  try {
    response = await fetch(path, { ...init, headers });
  } catch {
    throw new ApiError(
      "Cannot reach admin API — start it with: python3 -m mm.tools.run_local_stack --skip consumer",
      0,
    );
  }
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = (await response.json()) as { detail?: string };
      if (body.detail) {
        detail = body.detail;
      }
    } catch {
      /* ignore */
    }
    throw new ApiError(detail, response.status);
  }
  if (response.status === 204) {
    return undefined as T;
  }
  return (await response.json()) as T;
}

export const api = {
  get: <T>(path: string) => request<T>(path),
  post: <T>(path: string, body?: unknown) =>
    request<T>(path, {
      method: "POST",
      body: body !== undefined ? JSON.stringify(body) : undefined,
    }),
  put: <T>(path: string, body: unknown) =>
    request<T>(path, { method: "PUT", body: JSON.stringify(body) }),
  patch: <T>(path: string, body: unknown) =>
    request<T>(path, { method: "PATCH", body: JSON.stringify(body) }),
};

export interface ServiceProbe {
  id: string;
  name: string;
  metrics_url: string;
  health_url: string | null;
  status: "up" | "down";
  metrics_ok: boolean;
  health_ok: boolean | null;
  detail: string;
}

export interface ServicesResponse {
  checked_at_ms: number;
  services: ServiceProbe[];
}

export interface GrafanaDashboard {
  title: string;
  uid: string;
  embed_url: string;
  open_url: string;
}

export interface GrafanaSoloPanel {
  title: string;
  uid: string;
  panel_id: number;
  embed_url: string;
}

export interface GrafanaConfig {
  base_url: string;
  dashboards: GrafanaDashboard[];
  solo_panels: GrafanaSoloPanel[];
}

export interface RegistryList<T> {
  items: T[];
}

export interface Exchange {
  exchange: string;
  status: string;
  capabilities?: Record<string, unknown>;
  config?: Record<string, unknown>;
  auth_spec?: ExchangeAuthSpec | null;
}

export interface ExchangeUpsertPayload {
  exchange: string;
  status?: string;
  capabilities?: Record<string, unknown>;
  config?: Record<string, unknown>;
}

export interface PresetUpsertPayload {
  preset_id: string;
  config: Record<string, unknown>;
  version?: number;
  status?: string;
}

export interface MarketSource {
  source_id: string;
  exchange: string;
  symbol: string;
  status: string;
  mode: string;
  depth_limit: number;
  trades_limit: number;
  poll_interval_ms: number;
}

export interface MarketSourceUpsertPayload {
  source_id?: string;
  exchange: string;
  symbol: string;
  status?: string;
}

export interface MarketFeed {
  feed_id: string;
  source_id: string;
  interval_ms: number;
  status: string;
  exchange: string;
  symbol: string;
  mode: string;
  depth_limit: number;
  trades_limit: number;
  poll_interval_ms: number;
}

export interface MarketFeedUpsertPayload {
  feed_id?: string;
  source_id: string;
  interval_ms: number;
  status?: string;
}

export interface PlatformSettings {
  settings_id: string;
  bus: Record<string, unknown>;
  persistence: Record<string, unknown>;
  observability: Record<string, unknown>;
  scheduler: Record<string, unknown>;
}

export interface WorkerAssignmentUpsertPayload {
  assignment_id?: string;
  worker_id: string;
  strategy_ids: string[];
  lease_ttl_seconds?: number;
  status?: string;
}

export interface AccountUpsertPayload {
  account_id: string;
  exchange: string;
  credentials?: AccountCredentials;
  credential_ref?: string;
  role?: string;
  status?: string;
}

export interface Strategy {
  strategy_id: string;
  actor_id: string;
  account_id: string;
  symbol: string;
  preset_id: string;
  quoting_exchange: string;
  liquidity_exchange: string;
  status: string;
  config_overrides?: Record<string, unknown>;
  wallet_id?: string | null;
  ledger_source?: string;
}

export interface StrategyRuntimeConfig {
  strategy_id: string;
  runtime_config: Record<string, unknown>;
  resolved_contract: Record<string, unknown>;
}

export interface Account {
  account_id: string;
  exchange: string;
  credential_ref?: string;
  credentials?: AccountCredentials | null;
  credential_storage?: "stored" | "legacy_env";
  role: string;
  status: string;
}

export interface Preset {
  preset_id: string;
  version: number;
  status: string;
  config: Record<string, unknown>;
}

export interface StrategyUpsertPayload {
  strategy_id: string;
  actor_id: string;
  account_id: string;
  symbol: string;
  preset_id: string;
  quoting_exchange: string;
  liquidity_exchange: string;
  status?: string;
  config_overrides?: Record<string, unknown>;
  wallet_id?: string | null;
  ledger_source?: string;
}

export interface AllocationUpsertPayload {
  allocation_id?: string;
  strategy_id?: string | null;
  actor_id?: string;
  account_id: string;
  quote_asset?: string;
  base_asset?: string;
  allocated_quote?: number;
  allocated_base?: number;
  halt_on_breach?: boolean;
  status?: string;
}

export interface Allocation {
  allocation_id: string;
  strategy_id: string | null;
  actor_id: string;
  account_id: string;
  quote_asset: string;
  base_asset: string;
  allocated_quote: number;
  allocated_base: number;
  halt_on_breach: boolean;
  status: string;
}

export interface WorkerAssignment {
  assignment_id: string;
  worker_id: string;
  strategy_ids: string[];
  lease_expires_at?: string | null;
  heartbeat_at?: string | null;
  status: string;
}

export interface AdminStatus {
  status: string;
  bus: string;
  database_url_configured: boolean;
  registry_enabled: boolean;
  grafana_url?: string;
}

export interface AdminCommandPayload {
  command: string;
  actor_id?: string | null;
  exchange?: string | null;
  enabled?: boolean | null;
  reason?: string;
  payload?: Record<string, unknown>;
}

export interface CommandResult {
  status: string;
  command_id: string;
}

export type WorkerLiveStatus = "alive" | "stale" | "offline";

export type TradingStatus =
  | "running"
  | "paused"
  | "breach"
  | "kill_switch"
  | "dry_run"
  | "worker_offline"
  | "unassigned"
  | "registry_inactive";

export interface RuntimeControlState {
  connected: boolean;
  paused_actors: string[];
  allocation_breached_actors: string[];
  paused_venue_roles: string[];
  dry_run: boolean;
  kill_switch: boolean;
  updated_at_ms: number;
}

export interface RuntimeOverviewWorker {
  assignment_id: string;
  worker_id: string;
  strategy_ids: string[];
  registry_status: string;
  live: WorkerLiveStatus;
  lease_remaining_s: number;
  heartbeat_at: string | null;
  lease_expires_at: string | null;
}

export interface RuntimeOverviewStrategy {
  strategy_id: string;
  actor_id: string;
  registry_status: string;
  worker_id: string | null;
  worker_live: WorkerLiveStatus | null;
  trading_status: TradingStatus;
}

export interface RuntimeOverview {
  checked_at_ms: number;
  control: RuntimeControlState;
  workers: RuntimeOverviewWorker[];
  strategies: RuntimeOverviewStrategy[];
}

export interface StrategyControlResult {
  status: string;
  command_id: string;
  actor_id: string;
}
