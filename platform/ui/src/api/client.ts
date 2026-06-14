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
  const response = await fetch(path, { ...init, headers });
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

export interface Strategy {
  strategy_id: string;
  actor_id: string;
  account_id: string;
  symbol: string;
  preset_id: string;
  quoting_exchange: string;
  liquidity_exchange: string;
  status: string;
}

export interface Allocation {
  allocation_id: string;
  strategy_id: string;
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
  lease_ttl_seconds: number;
  status: string;
  heartbeat_at_ms?: number;
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
