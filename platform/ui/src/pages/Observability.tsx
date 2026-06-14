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
          Embedded Grafana panels from{" "}
          {config ? (
            <a href={config.base_url} target="_blank" rel="noreferrer">
              {config.base_url}
            </a>
          ) : (
            "Grafana"
          )}
          . Start Grafana with <code>docker compose up grafana</code> for local embeds.
        </p>
      </header>

      {error ? <div className="alert alert-error">{error}</div> : null}
      {loading ? <p className="loading">Loading…</p> : null}

      {config ? (
        <>
          <div className="alert alert-info" style={{ fontSize: "0.85rem" }}>
            If panels show a login screen, ensure Grafana allows embedding and anonymous
            viewer access (see docker-compose Grafana env vars).
          </div>

          <h2 style={{ fontSize: "1rem", marginBottom: "0.75rem" }}>Key metrics</h2>
          <div className="grafana-solo-grid">
            {config.solo_panels.map((panel) => (
              <GrafanaPanel
                key={`${panel.uid}-${panel.panel_id}`}
                title={panel.title}
                embedUrl={panel.embed_url}
                variant="solo"
              />
            ))}
          </div>

          <h2 style={{ fontSize: "1rem", margin: "1.5rem 0 0.75rem" }}>Dashboards</h2>
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
            <GrafanaPanel
              title={activeDashboard.title}
              embedUrl={activeDashboard.embed_url}
              openUrl={activeDashboard.open_url}
              variant="full"
            />
          ) : null}
        </>
      ) : null}
    </>
  );
}
