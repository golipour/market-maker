import { useCallback, useEffect, useMemo, useState } from "react";
import {
  ApiError,
  api,
  type Exchange,
  type GrafanaConfig,
  type MarketFeed,
  type MarketSource,
  type RegistryList,
  type Strategy,
} from "../api/client";
import { GrafanaPanel } from "../components/GrafanaPanel";
import {
  LADDER_BINNING_OPTIONS,
  SPREAD_BUCKET_OPTIONS,
  SPREAD_MAX_OPTIONS,
  TARGET_FRAME_MODES,
  TIME_RANGES,
  applyGrafanaScope,
  loadGrafanaScope,
  saveGrafanaScope,
  usesFrameScope,
  type GrafanaScopeState,
} from "../lib/observabilityGrafana";

const ALL_VALUE = "$__all";

function browserStorage(): Storage | null {
  try {
    return window.localStorage;
  } catch {
    return null;
  }
}

function uniqueSorted(values: Iterable<string>): string[] {
  return [...new Set(values)].sort((a, b) => a.localeCompare(b));
}

export function ObservabilityPage() {
  const [config, setConfig] = useState<GrafanaConfig | null>(null);
  const [scope, setScope] = useState<GrafanaScopeState>(() => loadGrafanaScope(browserStorage()));
  const [visitedUids, setVisitedUids] = useState<string[]>(() => (scope.activeUid ? [scope.activeUid] : []));
  const [exchanges, setExchanges] = useState<string[]>([]);
  const [symbols, setSymbols] = useState<string[]>([]);
  const [actors, setActors] = useState<string[]>([]);
  const [intervals, setIntervals] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [data, exchangeRows, sourceRows, feedRows, strategyRows] = await Promise.all([
        api.get<GrafanaConfig>("/api/admin/platform/grafana"),
        api.get<RegistryList<Exchange>>("/api/admin/registry/exchanges").catch(() => ({ items: [] })),
        api.get<RegistryList<MarketSource>>("/api/admin/registry/market-sources").catch(() => ({ items: [] })),
        api.get<RegistryList<MarketFeed>>("/api/admin/registry/market-feeds").catch(() => ({ items: [] })),
        api.get<RegistryList<Strategy>>("/api/admin/registry/strategies").catch(() => ({ items: [] })),
      ]);
      setConfig(data);
      setExchanges(uniqueSorted(exchangeRows.items.map((row) => row.exchange)));
      setSymbols(uniqueSorted(sourceRows.items.map((row) => row.symbol)));
      setActors(uniqueSorted(strategyRows.items.map((row) => row.actor_id).filter(Boolean)));
      setIntervals(
        uniqueSorted(feedRows.items.map((row) => String(row.interval_ms)).filter((value) => value !== "NaN")),
      );
      setScope((prev) => {
        if (prev.activeUid && data.dashboards.some((dashboard) => dashboard.uid === prev.activeUid)) {
          return prev;
        }
        return { ...prev, activeUid: data.dashboards[0]?.uid ?? null };
      });
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to load Grafana config");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  useEffect(() => {
    saveGrafanaScope(scope, browserStorage());
  }, [scope]);

  useEffect(() => {
    if (!scope.activeUid) {
      return;
    }
    setVisitedUids((prev) => (prev.includes(scope.activeUid!) ? prev : [...prev, scope.activeUid!]));
  }, [scope.activeUid]);

  const activeDashboard = config?.dashboards.find((dashboard) => dashboard.uid === scope.activeUid);
  const showFrameScope = usesFrameScope(scope.activeUid);

  const scopedDashboards = useMemo(() => {
    if (!config) {
      return [];
    }
    const mounted = new Set(visitedUids);
    if (scope.activeUid) {
      mounted.add(scope.activeUid);
    }
    return config.dashboards.filter((dashboard) => mounted.has(dashboard.uid));
  }, [config, visitedUids, scope.activeUid]);

  function patchScope(patch: Partial<GrafanaScopeState>) {
    setScope((prev) => ({ ...prev, ...patch }));
  }

  return (
    <>
      <header className="page-header">
        <h1>Observability</h1>
        <p>
          Grafana dashboards from{" "}
          {config ? (
            <a href={config.base_url} target="_blank" rel="noreferrer">
              {config.base_url}
            </a>
          ) : (
            "Grafana"
          )}
          {config?.embed === false
            ? ". Cluster Grafana requires login, so this tab opens dashboards in a new tab instead of embedding them."
            : ". Dashboard, time range, and variable filters are kept for this browser."}
        </p>
      </header>

      {error ? <div className="alert alert-error">{error}</div> : null}
      {loading ? <p className="loading">Loading…</p> : null}

      {config ? (
        <>
          {config.embed !== false ? (
            <div className="alert alert-info" style={{ fontSize: "0.85rem" }}>
              If an embed shows Grafana login, log in once at the Grafana URL, or use a
              public-dashboard token. Use Open in Grafana for the full editor.
            </div>
          ) : (
            <div className="alert alert-info" style={{ fontSize: "0.85rem" }}>
              Import the Observability v2 dashboards into Grafana and add a Postgres
              datasource with UID <code>postgres-mm</code> before these links have data.
              Prometheus panels use datasource UID <code>mimir</code> on this cluster.
            </div>
          )}

          <h2 style={{ fontSize: "1rem", marginBottom: "0.75rem" }}>Dashboards</h2>
          <div className="dashboard-tabs">
            {config.dashboards.map((dashboard) => (
              <button
                key={dashboard.uid}
                type="button"
                className={dashboard.uid === scope.activeUid ? "active" : undefined}
                onClick={() => patchScope({ activeUid: dashboard.uid })}
              >
                {dashboard.title}
              </button>
            ))}
          </div>

          <div className="observability-scope">
            <label>
              Time
              <select
                value={scope.from}
                onChange={(event) => patchScope({ from: event.target.value, to: "now" })}
              >
                {TIME_RANGES.some((row) => row.value === scope.from) ? null : (
                  <option value={scope.from}>{scope.from}</option>
                )}
                {TIME_RANGES.map((row) => (
                  <option key={row.value} value={row.value}>
                    {row.label}
                  </option>
                ))}
              </select>
            </label>
            <ScopeSelect
              label="Exchange"
              value={scope.exchange}
              options={exchanges}
              onChange={(exchange) => patchScope({ exchange })}
            />
            <ScopeSelect
              label="Symbol"
              value={scope.symbol}
              options={symbols}
              onChange={(symbol) => patchScope({ symbol })}
            />
            <ScopeSelect
              label="Actor"
              value={scope.actor_id}
              options={actors}
              onChange={(actor_id) => patchScope({ actor_id })}
            />
            <ScopeSelect
              label="Interval ms"
              value={scope.interval_ms}
              options={intervals}
              onChange={(interval_ms) => patchScope({ interval_ms })}
            />
            {showFrameScope ? (
              <>
                <label>
                  Ladder binning
                  <select
                    value={scope.ladder_binning}
                    onChange={(event) => patchScope({ ladder_binning: event.target.value })}
                  >
                    {LADDER_BINNING_OPTIONS.map((row) => (
                      <option key={row.value} value={row.value}>
                        {row.label}
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  Spread max
                  <select
                    value={scope.spread_max}
                    onChange={(event) => patchScope({ spread_max: event.target.value })}
                  >
                    {SPREAD_MAX_OPTIONS.map((value) => (
                      <option key={value} value={value}>
                        {value}
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  Spread buckets
                  <select
                    value={scope.spread_buckets}
                    onChange={(event) => patchScope({ spread_buckets: event.target.value })}
                  >
                    {SPREAD_BUCKET_OPTIONS.map((value) => (
                      <option key={value} value={value}>
                        {value}
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  Frame
                  <select
                    value={scope.target_frame_mode}
                    onChange={(event) => patchScope({ target_frame_mode: event.target.value })}
                  >
                    {TARGET_FRAME_MODES.map((row) => (
                      <option key={row.value} value={row.value}>
                        {row.label}
                      </option>
                    ))}
                  </select>
                </label>
              </>
            ) : null}
          </div>

          {activeDashboard ? (
            config.embed === false ? (
              <div className="card">
                <h3>{activeDashboard.title}</h3>
                <p>
                  <a
                    href={applyGrafanaScope(activeDashboard.open_url, scope)}
                    target="_blank"
                    rel="noreferrer"
                  >
                    Open in Grafana
                  </a>
                </p>
              </div>
            ) : (
              scopedDashboards.map((dashboard) => (
                <GrafanaPanel
                  key={dashboard.uid}
                  title={dashboard.title}
                  embedUrl={applyGrafanaScope(dashboard.embed_url, scope)}
                  openUrl={applyGrafanaScope(dashboard.open_url, scope)}
                  variant="full"
                  hidden={dashboard.uid !== scope.activeUid}
                />
              ))
            )
          ) : null}
        </>
      ) : null}
    </>
  );
}

function ScopeSelect({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: string;
  options: string[];
  onChange: (value: string) => void;
}) {
  const known = value === ALL_VALUE || options.includes(value);
  return (
    <label>
      {label}
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        <option value={ALL_VALUE}>All</option>
        {known ? null : <option value={value}>{value}</option>}
        {options.map((option) => (
          <option key={option} value={option}>
            {option}
          </option>
        ))}
      </select>
    </label>
  );
}
