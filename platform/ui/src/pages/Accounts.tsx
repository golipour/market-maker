import { FormEvent, useCallback, useEffect, useState } from "react";
import { JsonBlock } from "../components/JsonBlock";
import {
  ApiError,
  api,
  type Account,
  type AccountUpsertPayload,
  type Exchange,
  type RegistryList,
} from "../api/client";

const STATUSES = ["active", "disabled"] as const;
const ROLES = ["trading"] as const;

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
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [form, setForm] = useState<AccountUpsertPayload>(EMPTY);
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

  const selected = items.find((row) => row.account_id === selectedId) ?? null;

  function openCreate() {
    setEditingId(null);
    setForm(EMPTY);
    setShowForm(true);
    setMessage(null);
    setError(null);
  }

  function openEdit(row: Account) {
    setEditingId(row.account_id);
    setForm({
      account_id: row.account_id,
      exchange: row.exchange,
      credential_ref: row.credential_ref,
      role: row.role,
      status: row.status,
    });
    setSelectedId(row.account_id);
    setShowForm(true);
    setMessage(null);
    setError(null);
  }

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    setSaving(true);
    setError(null);
    try {
      const saved = await api.post<Account>("/api/admin/registry/accounts", form);
      setMessage(`Account ${saved.account_id} saved`);
      setShowForm(false);
      setEditingId(null);
      setSelectedId(saved.account_id);
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
        <p>
          Trading accounts linked to exchanges. Secrets stay in <code>.env</code>;{" "}
          <code>credential_ref</code> is an env var name or JSON with{" "}
          <code>api_token_env</code> / <code>api_private_key_env</code> keys merged into exchange
          private config at runtime.
        </p>
      </header>
      <div className="page-toolbar">
        <button type="button" className="btn btn-primary" onClick={openCreate}>
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
          <h3>{editingId ? `Edit ${editingId}` : "New account"}</h3>
          <div className="form-grid">
            <div className="form-field">
              <label htmlFor="acc-id">Account ID</label>
              <input
                id="acc-id"
                required
                readOnly={Boolean(editingId)}
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
                placeholder='NOBITEX_API_TOKEN or {"api_token_env":"…","api_private_key_env":"…"}'
                value={form.credential_ref}
                onChange={(e) => setForm((p) => ({ ...p, credential_ref: e.target.value }))}
              />
            </div>
            <div className="form-field">
              <label htmlFor="acc-role">Role</label>
              <select
                id="acc-role"
                value={form.role ?? "trading"}
                onChange={(e) => setForm((p) => ({ ...p, role: e.target.value }))}
              >
                {ROLES.map((role) => (
                  <option key={role} value={role}>
                    {role}
                  </option>
                ))}
              </select>
            </div>
            <div className="form-field">
              <label htmlFor="acc-status">Status</label>
              <select
                id="acc-status"
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
              <th>Account</th>
              <th>Exchange</th>
              <th>Credential ref</th>
              <th>Role</th>
              <th>Status</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {items.map((row) => (
              <tr key={row.account_id}>
                <td>{row.account_id}</td>
                <td>{row.exchange}</td>
                <td style={{ fontSize: "0.9rem", maxWidth: "14rem", overflow: "hidden", textOverflow: "ellipsis" }}>
                  {row.credential_ref}
                </td>
                <td>{row.role}</td>
                <td>{row.status}</td>
                <td style={{ whiteSpace: "nowrap" }}>
                  <button type="button" className="btn" onClick={() => setSelectedId(row.account_id)}>
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
                <td colSpan={6} style={{ color: "var(--text-muted)" }}>
                  No accounts configured.
                </td>
              </tr>
            ) : null}
          </tbody>
        </table>
      </div>
      {selected && !showForm ? (
        <section className="card" style={{ marginTop: "1rem" }}>
          <h3>{selected.account_id}</h3>
          <p style={{ color: "var(--text-muted)", marginBottom: "0.75rem" }}>
            At resolve time, <code>credential_ref</code> is merged into{" "}
            <code>exchanges.{selected.exchange}.private</code> for strategies on this account.
          </p>
          <JsonBlock value={selected} />
        </section>
      ) : null}
    </>
  );
}
