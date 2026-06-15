import { FormEvent, useCallback, useEffect, useState } from "react";
import {
  ApiError,
  api,
  type Account,
  type AccountUpsertPayload,
  type Exchange,
  type RegistryList,
} from "../api/client";

const EMPTY: AccountUpsertPayload = {
  account_id: "",
  exchange: "",
  credential_ref: "",
  role: "trading",
  status: "active",
};

export function AccountsPage() {
  const [items, setItems] = useState<Account[]>([]);
  const [exchanges, setExchanges] = useState<Exchange[]>([]);
  const [form, setForm] = useState<AccountUpsertPayload>(EMPTY);
  const [showForm, setShowForm] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [accounts, exchangeRows] = await Promise.all([
        api.get<RegistryList<Account>>("/api/admin/registry/accounts"),
        api.get<RegistryList<Exchange>>("/api/admin/registry/exchanges"),
      ]);
      setItems(accounts.items);
      setExchanges(exchangeRows.items);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to load accounts");
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
      await api.post<Account>("/api/admin/registry/accounts", form);
      setMessage(`Account ${form.account_id} saved`);
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
        <h1>Accounts</h1>
        <p>Trading accounts linked to exchanges; credentials via env refs only.</p>
      </header>
      <div className="page-toolbar">
        <button type="button" className="btn btn-primary" onClick={() => setShowForm(true)}>
          New account
        </button>
        <button type="button" className="btn" onClick={() => void load()}>
          Refresh
        </button>
      </div>
      {error ? <div className="alert alert-error">{error}</div> : null}
      {message ? <div className="alert alert-success">{message}</div> : null}
      {showForm ? (
        <form onSubmit={(e) => void onSubmit(e)} className="card form-card">
          <h3>New account</h3>
          <div className="form-grid">
            <div className="form-field">
              <label htmlFor="acc-id">Account ID</label>
              <input
                id="acc-id"
                required
                value={form.account_id}
                onChange={(e) => setForm((p) => ({ ...p, account_id: e.target.value }))}
              />
            </div>
            <div className="form-field">
              <label htmlFor="acc-exchange">Exchange</label>
              <select
                id="acc-exchange"
                required
                value={form.exchange}
                onChange={(e) => setForm((p) => ({ ...p, exchange: e.target.value }))}
              >
                <option value="">Select…</option>
                {exchanges.map((row) => (
                  <option key={row.exchange} value={row.exchange}>
                    {row.exchange}
                  </option>
                ))}
              </select>
            </div>
            <div className="form-field">
              <label htmlFor="acc-cred">Credential ref</label>
              <input
                id="acc-cred"
                required
                placeholder="NOBITEX_API_TOKEN"
                value={form.credential_ref}
                onChange={(e) => setForm((p) => ({ ...p, credential_ref: e.target.value }))}
              />
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
              <th>Account</th>
              <th>Exchange</th>
              <th>Credential ref</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {items.map((row) => (
              <tr key={row.account_id}>
                <td>{row.account_id}</td>
                <td>{row.exchange}</td>
                <td>{row.credential_ref}</td>
                <td>{row.status}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}
