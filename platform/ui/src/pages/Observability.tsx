import { useCallback, useEffect, useState } from "react";
import { ApiError, api, type GrafanaConfig } from "../api/client";
import { GrafanaPanel } from "../components/GrafanaPanel";

export function ObservabilityPage() {
  const [config, setConfig] = useState<GrafanaConfig | null>(null);
  const [activeUid, setActiveUid] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await api.get<GrafanaConfig>("/api/admin/platform/grafana");
      setConfig(data);
      setActiveUid((prev) => prev ?? data.dashboards[0]?.uid ?? null);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to load Grafana config");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const activeDashboard = config?.dashboards.find((d) => d.uid === activeUid);

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
            : "."}
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
                className={dashboard.uid === activeUid ? "active" : undefined}
                onClick={() => setActiveUid(dashboard.uid)}
              >
                {dashboard.title}
              </button>
            ))}
          </div>

          {activeDashboard ? (
            config.embed === false ? (
              <div className="card">
                <h3>{activeDashboard.title}</h3>
                <p>
                  <a href={activeDashboard.open_url} target="_blank" rel="noreferrer">
                    Open in Grafana
                  </a>
                </p>
              </div>
            ) : (
              <GrafanaPanel
                title={activeDashboard.title}
                embedUrl={activeDashboard.embed_url}
                openUrl={activeDashboard.open_url}
                variant="full"
              />
            )
          ) : null}
        </>
      ) : null}
    </>
  );
}
