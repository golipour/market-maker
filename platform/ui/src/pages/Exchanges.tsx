import { FormEvent, useCallback, useEffect, useState } from "react";
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
  const [form, setForm] = useState<ExchangeUpsertPayload>(EMPTY);
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

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    setSaving(true);
    setError(null);
    try {
      await api.post<Exchange>("/api/admin/registry/exchanges", form);
      setMessage(`Exchange ${form.exchange} saved`);
      setShowForm(false);
      setForm(EMPTY);
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
        <h1>Exchanges</h1>
        <p>Venue registry: capabilities and default market/private transport config.</p>
      </header>
      <div className="page-toolbar">
        <button type="button" className="btn btn-primary" onClick={() => setShowForm(true)}>
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
          <h3>New exchange</h3>
          <div className="form-grid">
            <div className="form-field">
              <label htmlFor="ex-id">Exchange ID</label>
              <input
                id="ex-id"
                required
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
              <th>Exchange</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {items.map((row) => (
              <tr key={row.exchange}>
                <td>{row.exchange}</td>
                <td>{row.status}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}
