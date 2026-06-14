import { useCallback, useEffect, useState } from "react";
import { ApiError, api, type AdminStatus, type ServicesResponse } from "../api/client";
import { ServiceCard } from "../components/ServiceCard";

export function OverviewPage() {
  const [services, setServices] = useState<ServicesResponse | null>(null);
  const [status, setStatus] = useState<AdminStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [svc, st] = await Promise.all([
        api.get<ServicesResponse>("/api/admin/platform/services"),
        api.get<AdminStatus>("/api/admin/status"),
      ]);
      setServices(svc);
      setStatus(st);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to load overview");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
    const timer = window.setInterval(() => void load(), 15000);
    return () => window.clearInterval(timer);
  }, [load]);

  const upCount = services?.services.filter((s) => s.status === "up").length ?? 0;
  const total = services?.services.length ?? 0;

  return (
    <>
      <header className="page-header">
        <h1>Overview</h1>
        <p>Control plane status and decomposed runtime service health.</p>
      </header>

      {error ? <div className="alert alert-error">{error}</div> : null}
      {loading && !services ? <p className="loading">Loading…</p> : null}

      {status ? (
        <div className="card-grid" style={{ marginBottom: "1.5rem" }}>
          <div className="card">
            <h3>Control Plane</h3>
            <p>Bus: {status.bus}</p>
            <p>Registry: {status.registry_enabled ? "enabled" : "disabled"}</p>
            <p>Database: {status.database_url_configured ? "configured" : "not set"}</p>
          </div>
          <div className="card">
            <h3>Services</h3>
            <p>
              {upCount} / {total} reporting up
            </p>
            {services ? (
              <p style={{ fontSize: "0.8rem" }}>
                Last check: {new Date(services.checked_at_ms).toLocaleTimeString()}
              </p>
            ) : null}
            <button type="button" className="btn" onClick={() => void load()}>
              Refresh
            </button>
          </div>
          {status.grafana_url ? (
            <div className="card">
              <h3>Grafana</h3>
              <a href={status.grafana_url} target="_blank" rel="noreferrer">
                {status.grafana_url}
              </a>
            </div>
          ) : null}
        </div>
      ) : null}

      {services ? (
        <div className="card-grid">
          {services.services.map((service) => (
            <ServiceCard key={service.id} service={service} />
          ))}
        </div>
      ) : null}
    </>
  );
}
