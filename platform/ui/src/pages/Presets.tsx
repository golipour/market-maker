import { FormEvent, useCallback, useEffect, useState } from "react";
import { JsonBlock } from "../components/JsonBlock";
import {
  ApiError,
  api,
  type Preset,
  type PresetUpsertPayload,
  type RegistryList,
} from "../api/client";

const EMPTY: PresetUpsertPayload = {
  preset_id: "",
  config: {},
  status: "active",
};

export function PresetsPage() {
  const [items, setItems] = useState<Preset[]>([]);
  const [form, setForm] = useState<PresetUpsertPayload>(EMPTY);
  const [configJson, setConfigJson] = useState("{}");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [showForm, setShowForm] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await api.get<RegistryList<Preset>>("/api/admin/registry/presets");
      setItems(data.items);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to load presets");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const selected = items.find((row) => row.preset_id === selectedId) ?? null;

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    setSaving(true);
    setError(null);
    try {
      const config = JSON.parse(configJson) as Record<string, unknown>;
      await api.post<Preset>("/api/admin/registry/presets", { ...form, config });
      setMessage(`Preset ${form.preset_id} saved`);
      setShowForm(false);
      setForm(EMPTY);
      setConfigJson("{}");
      await load();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Save failed (check JSON)");
    } finally {
      setSaving(false);
    }
  }

  return (
    <>
      <header className="page-header">
        <h1>Presets</h1>
        <p>Reusable strategy logic templates (MPC, risk, execution) without symbol binding.</p>
      </header>
      <div className="page-toolbar">
        <button type="button" className="btn btn-primary" onClick={() => setShowForm(true)}>
          New preset
        </button>
        <button type="button" className="btn" onClick={() => void load()}>
          Refresh
        </button>
      </div>
      {error ? <div className="alert alert-error">{error}</div> : null}
      {message ? <div className="alert alert-success">{message}</div> : null}
      {showForm ? (
        <form onSubmit={(e) => void onSubmit(e)} className="card form-card">
          <h3>New preset</h3>
          <div className="form-grid">
            <div className="form-field">
              <label htmlFor="preset-id">Preset ID</label>
              <input
                id="preset-id"
                required
                value={form.preset_id}
                onChange={(e) => setForm((p) => ({ ...p, preset_id: e.target.value }))}
              />
            </div>
          </div>
          <div className="form-field">
            <label htmlFor="preset-config">Config JSON</label>
            <textarea
              id="preset-config"
              rows={12}
              value={configJson}
              onChange={(e) => setConfigJson(e.target.value)}
            />
          </div>
          <div className="form-actions">
            <button type="submit" className="btn btn-primary" disabled={saving}>
              {saving ? "Saving…" : "Save"}
            </button>
            <button type="button" className="btn" onClick={() => setShowForm(false)}>
              Cancel
            </button>
          </div>
        </form>
      ) : null}
      {loading ? <p className="loading">Loading…</p> : null}
      <div className="table-wrap">
        <table className="data-table">
          <thead>
            <tr>
              <th>Preset</th>
              <th>Version</th>
              <th>Status</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {items.map((row) => (
              <tr key={row.preset_id}>
                <td>{row.preset_id}</td>
                <td>{row.version}</td>
                <td>{row.status}</td>
                <td>
                  <button type="button" className="btn" onClick={() => setSelectedId(row.preset_id)}>
                    View config
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {selected ? (
        <section className="card" style={{ marginTop: "1rem" }}>
          <h3>{selected.preset_id}</h3>
          <JsonBlock value={selected.config} />
        </section>
      ) : null}
    </>
  );
}
