import { FormEvent, useCallback, useEffect, useState } from "react";
import { EditModeNote } from "../components/EditModeNote";
import { JsonBlock } from "../components/JsonBlock";
import {
  ApiError,
  api,
  type Exchange,
  type ExchangeUpsertPayload,
  type RegistryList,
} from "../api/client";

const STATUSES = ["active", "disabled"] as const;

const EMPTY: ExchangeUpsertPayload = {
  exchange: "",
  status: "active",
  capabilities: {},
  config: {},
};

export function ExchangesPage() {
  const [items, setItems] = useState<Exchange[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [form, setForm] = useState<ExchangeUpsertPayload>(EMPTY);
  const [capabilitiesJson, setCapabilitiesJson] = useState("{}");
  const [configJson, setConfigJson] = useState("{}");
  const [editingId, setEditingId] = useState<string | null>(null);
  const [showForm, setShowForm] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await api.get<RegistryList<Exchange>>("/api/admin/registry/exchanges");
      setItems(data.items);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to load exchanges");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const selected = items.find((row) => row.exchange === selectedId) ?? null;

  function openCreate() {
    setEditingId(null);
    setForm(EMPTY);
    setCapabilitiesJson("{}");
    setConfigJson("{}");
    setShowForm(true);
    setMessage(null);
    setError(null);
  }

  function openEdit(row: Exchange) {
    setEditingId(row.exchange);
    setForm({
      exchange: row.exchange,
      status: row.status,
      capabilities: row.capabilities ?? {},
      config: row.config ?? {},
    });
    setCapabilitiesJson(JSON.stringify(row.capabilities ?? {}, null, 2));
    setConfigJson(JSON.stringify(row.config ?? {}, null, 2));
    setShowForm(true);
    setSelectedId(row.exchange);
    setMessage(null);
    setError(null);
  }

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    setSaving(true);
    setError(null);
    try {
      const body: ExchangeUpsertPayload = {
        ...form,
        capabilities: JSON.parse(capabilitiesJson) as Record<string, unknown>,
        config: JSON.parse(configJson) as Record<string, unknown>,
      };
      const saved = await api.post<Exchange>("/api/admin/registry/exchanges", body);
      setMessage(`Exchange ${saved.exchange} saved`);
      setShowForm(false);
      setEditingId(null);
      setSelectedId(saved.exchange);
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
        <h1>Exchanges</h1>
        <p>
          Venue registry: default market-data, private-state, and fee settings merged into each
          strategy&apos;s runtime config. Status <code>active</code> means the venue is registered
          for use; <code>disabled</code> is an operator flag (the worker does not auto-block
          strategies on disabled venues yet).
        </p>
        <EditModeNote mode="hybrid" />
      </header>
      <div className="page-toolbar">
        <button type="button" className="btn btn-primary" onClick={openCreate}>
          New exchange
        </button>
        <button type="button" className="btn" onClick={() => void load()}>
          Refresh
        </button>
      </div>
      {error ? <div className="alert alert-error">{error}</div> : null}
      {message ? <div className="alert alert-success">{message}</div> : null}
      {showForm ? (
        <form onSubmit={(e) => void onSubmit(e)} className="card form-card">
          <h3>{editingId ? `Edit ${editingId}` : "New exchange"}</h3>
          <div className="form-grid">
            <div className="form-field">
              <label htmlFor="ex-id">Exchange ID</label>
              <input
                id="ex-id"
                required
                readOnly={Boolean(editingId)}
                value={form.exchange}
                onChange={(e) => setForm((p) => ({ ...p, exchange: e.target.value }))}
              />
            </div>
            <div className="form-field">
              <label htmlFor="ex-status">Status</label>
              <select
                id="ex-status"
                value={form.status ?? "active"}
                onChange={(e) => setForm((p) => ({ ...p, status: e.target.value }))}
              >
                {STATUSES.map((s) => (
                  <option key={s} value={s}>
                    {s}
                  </option>
                ))}
              </select>
            </div>
          </div>
          <div className="form-field">
            <label htmlFor="ex-capabilities">Capabilities JSON</label>
            <textarea
              id="ex-capabilities"
              rows={4}
              value={capabilitiesJson}
              onChange={(e) => setCapabilitiesJson(e.target.value)}
              placeholder='{"supports_websocket": true}'
            />
          </div>
          <div className="form-field">
            <label htmlFor="ex-config">Config JSON</label>
            <textarea
              id="ex-config"
              rows={14}
              value={configJson}
              onChange={(e) => setConfigJson(e.target.value)}
              placeholder='{"market_data": {"mode": "rest"}, "private": {...}, "fees": {...}}'
            />
          </div>
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
              <th>Exchange</th>
              <th>Status</th>
              <th>Config keys</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {items.map((row) => (
              <tr key={row.exchange}>
                <td>{row.exchange}</td>
                <td>{row.status}</td>
                <td style={{ color: "var(--text-muted)", fontSize: "0.9rem" }}>
                  {Object.keys(row.config ?? {}).join(", ") || "—"}
                </td>
                <td style={{ whiteSpace: "nowrap" }}>
                  <button type="button" className="btn" onClick={() => setSelectedId(row.exchange)}>
                    View
                  </button>{" "}
                  <button type="button" className="btn" onClick={() => openEdit(row)}>
                    Edit
                  </button>
                </td>
              </tr>
            ))}
            {!loading && items.length === 0 ? (
              <tr>
                <td colSpan={4} style={{ color: "var(--text-muted)" }}>
                  No exchanges configured.
                </td>
              </tr>
            ) : null}
          </tbody>
        </table>
      </div>
      {selected && !showForm ? (
        <section className="card" style={{ marginTop: "1rem" }}>
          <h3>{selected.exchange}</h3>
          <p style={{ color: "var(--text-muted)", marginBottom: "0.75rem" }}>
            Status: {selected.status}. This config is merged into worker runtime JSON for strategies
            on this venue (market transport, private polling, fees).
          </p>
          <h4>Config</h4>
          <JsonBlock value={selected.config ?? {}} />
          <h4 style={{ marginTop: "1rem" }}>Capabilities</h4>
          <JsonBlock value={selected.capabilities ?? {}} maxHeight="200px" />
        </section>
      ) : null}
    </>
  );
}
