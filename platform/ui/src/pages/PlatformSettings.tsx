import { FormEvent, useCallback, useEffect, useState } from "react";
import { JsonBlock } from "../components/JsonBlock";
import { ApiError, api, type PlatformSettings } from "../api/client";

export function PlatformSettingsPage() {
  const [settings, setSettings] = useState<PlatformSettings | null>(null);
  const [busJson, setBusJson] = useState("{}");
  const [persistenceJson, setPersistenceJson] = useState("{}");
  const [observabilityJson, setObservabilityJson] = useState("{}");
  const [schedulerJson, setSchedulerJson] = useState("{}");
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await api.get<PlatformSettings>("/api/admin/registry/platform-settings");
      setSettings(data);
      setBusJson(JSON.stringify(data.bus ?? {}, null, 2));
      setPersistenceJson(JSON.stringify(data.persistence ?? {}, null, 2));
      setObservabilityJson(JSON.stringify(data.observability ?? {}, null, 2));
      setSchedulerJson(JSON.stringify(data.scheduler ?? {}, null, 2));
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to load platform settings");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    setSaving(true);
    setError(null);
    try {
      const body = {
        bus: JSON.parse(busJson) as Record<string, unknown>,
        persistence: JSON.parse(persistenceJson) as Record<string, unknown>,
        observability: JSON.parse(observabilityJson) as Record<string, unknown>,
        scheduler: JSON.parse(schedulerJson) as Record<string, unknown>,
      };
      const saved = await api.put<PlatformSettings>("/api/admin/registry/platform-settings", body);
      setSettings(saved);
      setMessage("Platform settings saved");
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Save failed (check JSON)");
    } finally {
      setSaving(false);
    }
  }

  return (
    <>
      <header className="page-header">
        <h1>Platform settings</h1>
        <p>Shared bus, persistence, observability, and scheduler defaults for workers.</p>
      </header>
      {error ? <div className="alert alert-error">{error}</div> : null}
      {message ? <div className="alert alert-success">{message}</div> : null}
      {loading ? <p className="loading">Loading…</p> : null}
      {settings ? (
        <form onSubmit={(e) => void onSubmit(e)} className="card form-card">
          <h3>{settings.settings_id}</h3>
          <div className="form-field">
            <label htmlFor="bus-json">Bus</label>
            <textarea id="bus-json" rows={8} value={busJson} onChange={(e) => setBusJson(e.target.value)} />
          </div>
          <div className="form-field">
            <label htmlFor="persist-json">Persistence</label>
            <textarea
              id="persist-json"
              rows={4}
              value={persistenceJson}
              onChange={(e) => setPersistenceJson(e.target.value)}
            />
          </div>
          <div className="form-field">
            <label htmlFor="obs-json">Observability</label>
            <textarea
              id="obs-json"
              rows={6}
              value={observabilityJson}
              onChange={(e) => setObservabilityJson(e.target.value)}
            />
          </div>
          <div className="form-field">
            <label htmlFor="sched-json">Scheduler</label>
            <textarea
              id="sched-json"
              rows={4}
              value={schedulerJson}
              onChange={(e) => setSchedulerJson(e.target.value)}
            />
          </div>
          <div className="form-actions">
            <button type="submit" className="btn btn-primary" disabled={saving}>
              {saving ? "Saving…" : "Save settings"}
            </button>
            <button type="button" className="btn" onClick={() => void load()}>
              Reset
            </button>
          </div>
        </form>
      ) : null}
      {settings ? (
        <section className="card" style={{ marginTop: "1rem" }}>
          <h3>Current snapshot</h3>
          <JsonBlock value={settings} />
        </section>
      ) : null}
    </>
  );
}
