import { FormEvent, useCallback, useEffect, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { ConfigEditor } from "../components/ConfigEditor";
import { EditModeNote } from "../components/EditModeNote";
import { JsonBlock } from "../components/JsonBlock";
import { ALL_FIELD_PATHS } from "../config/presetSchema";
import {
  ApiError,
  api,
  type Preset,
  type PresetUpsertPayload,
  type RegistryList,
} from "../api/client";
import {
  cloneConfig,
  deepMerge,
  mergeFormIntoConfig,
  stripHelperKeys,
  type JsonObject,
} from "../lib/presetConfig";

const STATUSES = ["active", "disabled"] as const;

const EMPTY: PresetUpsertPayload = {
  preset_id: "",
  config: {},
  status: "active",
};

export function PresetsPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const [items, setItems] = useState<Preset[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [form, setForm] = useState<PresetUpsertPayload>(EMPTY);
  const [config, setConfig] = useState<JsonObject>({});
  const [loadedConfig, setLoadedConfig] = useState<JsonObject>({});
  const [editingId, setEditingId] = useState<string | null>(null);
  const [showForm, setShowForm] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const openedFromQuery = useRef<string | null>(null);

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

  function bindPreset(row: Preset) {
    const cleaned = cloneConfig((row.config ?? {}) as JsonObject);
    setEditingId(row.preset_id);
    setForm({
      preset_id: row.preset_id,
      config: cleaned,
      version: row.version,
      status: row.status,
    });
    setLoadedConfig(cleaned);
    setConfig(cleaned);
    setSelectedId(row.preset_id);
    setShowForm(true);
    setMessage(null);
    setError(null);
  }

  useEffect(() => {
    const presetId = searchParams.get("preset_id")?.trim();
    if (!presetId || loading || openedFromQuery.current === presetId) {
      return;
    }
    const row = items.find((item) => item.preset_id === presetId);
    if (!row) {
      return;
    }
    openedFromQuery.current = presetId;
    bindPreset(row);
    setSearchParams({}, { replace: true });
  }, [items, loading, searchParams, setSearchParams]);

  const selected = items.find((row) => row.preset_id === selectedId) ?? null;

  function openCreate() {
    setEditingId(null);
    setForm(EMPTY);
    setLoadedConfig({});
    setConfig({});
    setShowForm(true);
    setMessage(null);
    setError(null);
  }

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    setSaving(true);
    setError(null);
    try {
      const fromForm = mergeFormIntoConfig(loadedConfig, config, ALL_FIELD_PATHS);
      const merged = deepMerge(fromForm, stripHelperKeys(config)) as JsonObject;
      const saved = await api.post<Preset>("/api/admin/registry/presets", {
        ...form,
        config: merged,
      });
      setMessage(`Preset ${saved.preset_id} saved`);
      setShowForm(false);
      setEditingId(null);
      setSelectedId(saved.preset_id);
      await load();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Save failed");
    } finally {
      setSaving(false);
    }
  }

  return (
    <>
      <header className="page-header">
        <h1>Presets</h1>
        <p>
          Reusable strategy logic templates (MPC, risk, execution) without symbol or account
          binding. Edit via form or raw JSON; save uses the same registry config object.
        </p>
        <EditModeNote mode="hybrid" />
      </header>
      <div className="page-toolbar">
        <button type="button" className="btn btn-primary" onClick={openCreate}>
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
          <h3>{editingId ? `Edit ${editingId}` : "New preset"}</h3>
          <div className="form-grid">
            <div className="form-field">
              <label htmlFor="preset-id">Preset ID</label>
              <input
                id="preset-id"
                required
                readOnly={Boolean(editingId)}
                value={form.preset_id}
                onChange={(e) => setForm((p) => ({ ...p, preset_id: e.target.value }))}
              />
            </div>
            <div className="form-field">
              <label htmlFor="preset-status">Status</label>
              <select
                id="preset-status"
                value={form.status ?? "active"}
                onChange={(e) => setForm((p) => ({ ...p, status: e.target.value }))}
              >
                {STATUSES.map((status) => (
                  <option key={status} value={status}>
                    {status}
                  </option>
                ))}
              </select>
            </div>
          </div>
          <ConfigEditor
            label="Preset config"
            mode="full"
            value={config}
            onChange={setConfig}
          />
          <div className="form-actions">
            <button type="submit" className="btn btn-primary" disabled={saving}>
              {saving ? "Saving…" : "Save"}
            </button>
            <button
              type="button"
              className="btn"
              onClick={() => {
                setShowForm(false);
                setEditingId(null);
              }}
            >
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
              <th>Config keys</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {items.map((row) => (
              <tr key={row.preset_id}>
                <td>{row.preset_id}</td>
                <td>{row.version}</td>
                <td>{row.status}</td>
                <td style={{ color: "var(--text-muted)", fontSize: "0.9rem" }}>
                  {Object.keys(row.config ?? {}).join(", ") || "—"}
                </td>
                <td style={{ whiteSpace: "nowrap" }}>
                  <button type="button" className="btn" onClick={() => setSelectedId(row.preset_id)}>
                    View
                  </button>{" "}
                  <button type="button" className="btn" onClick={() => bindPreset(row)}>
                    Edit
                  </button>
                </td>
              </tr>
            ))}
            {!loading && items.length === 0 ? (
              <tr>
                <td colSpan={5} style={{ color: "var(--text-muted)" }}>
                  No presets configured.
                </td>
              </tr>
            ) : null}
          </tbody>
        </table>
      </div>
      {selected && !showForm ? (
        <section className="card" style={{ marginTop: "1rem" }}>
          <h3>{selected.preset_id}</h3>
          <p style={{ color: "var(--text-muted)", marginBottom: "0.75rem" }}>
            Version {selected.version} · status {selected.status}. Merged with strategy{" "}
            <code>config_overrides</code> at runtime.
          </p>
          <h4>Config</h4>
          <JsonBlock value={selected.config} />
        </section>
      ) : null}
    </>
  );
}
