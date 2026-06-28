import { FormEvent, useCallback, useEffect, useMemo, useState } from "react";
import { CredentialsEditor } from "../components/CredentialsEditor";
import { EditModeNote } from "../components/EditModeNote";
import { JsonBlock } from "../components/JsonBlock";
import {
  ApiError,
  api,
  type Account,
  type AccountUpsertPayload,
  type Exchange,
  type RegistryList,
} from "../api/client";
import {
  credentialsFormFromAccount,
  credentialsPayloadFromForm,
  credentialsSummary,
  defaultAuthModeForExchange,
  EMPTY_CREDENTIALS_FORM,
  type CredentialsFormState,
  type ExchangeAuthSpec,
} from "../lib/accountCredentials";

const STATUSES = ["active", "disabled"] as const;
const ROLES = ["trading"] as const;

const EMPTY: AccountUpsertPayload = {
  account_id: "",
  exchange: "",
  role: "trading",
  status: "active",
};

export function AccountsPage() {
  const [items, setItems] = useState<Account[]>([]);
  const [exchanges, setExchanges] = useState<Exchange[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [form, setForm] = useState<AccountUpsertPayload>(EMPTY);
  const [credentials, setCredentials] = useState<CredentialsFormState>(EMPTY_CREDENTIALS_FORM);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [showForm, setShowForm] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);

  const exchangeAuthSpecs = useMemo(() => {
    const map = new Map<string, ExchangeAuthSpec>();
    for (const row of exchanges) {
      if (row.auth_spec) {
        map.set(row.exchange, row.auth_spec);
      }
    }
    return map;
  }, [exchanges]);

  const selectedExchangeAuthSpec = form.exchange
    ? exchangeAuthSpecs.get(form.exchange) ?? null
    : null;

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
    setCredentials(EMPTY_CREDENTIALS_FORM);
    setShowForm(true);
    setMessage(null);
    setError(null);
  }

  function openEdit(row: Account) {
    setEditingId(row.account_id);
    setForm({
      account_id: row.account_id,
      exchange: row.exchange,
      role: row.role,
      status: row.status,
    });
    setCredentials(
      credentialsFormFromAccount(
        row.credentials,
        row.exchange,
        exchangeAuthSpecs.get(row.exchange),
      ),
    );
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
      const body: AccountUpsertPayload = {
        ...form,
        credentials: credentialsPayloadFromForm(credentials),
      };
      const saved = await api.post<Account>("/api/admin/registry/accounts", body);
      setMessage(`Account ${saved.account_id} saved`);
      setShowForm(false);
      setEditingId(null);
      setSelectedId(saved.account_id);
      await load();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : err instanceof Error ? err.message : "Save failed");
    } finally {
      setSaving(false);
    }
  }

  return (
    <>
      <header className="page-header">
        <h1>Accounts</h1>
        <p>
          Trading accounts linked to exchanges. API credentials are stored in the registry database
          and merged into <code>exchanges.*.private</code> at runtime.
        </p>
        <EditModeNote mode="form" />
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
                onChange={(e) => {
                  const exchange = e.target.value;
                  setForm((p) => ({ ...p, exchange }));
                  setCredentials((prev) => ({
                    ...prev,
                    auth_mode: defaultAuthModeForExchange(
                      exchange,
                      exchangeAuthSpecs.get(exchange),
                    ),
                  }));
                }}
              >
                <option value="">Select…</option>
                {exchanges.map((row) => (
                  <option key={row.exchange} value={row.exchange}>
                    {row.exchange}
                  </option>
                ))}
              </select>
            </div>
            {form.exchange ? (
              <CredentialsEditor
                exchange={form.exchange}
                authSpec={selectedExchangeAuthSpec}
                value={credentials}
                onChange={setCredentials}
              />
            ) : null}
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
            <button type="submit" className="btn btn-primary" disabled={saving || !form.exchange}>
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
              <th>Credentials</th>
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
                <td style={{ fontSize: "0.9rem" }}>
                  {row.credential_storage === "legacy_env"
                    ? `legacy env: ${row.credential_ref || "—"}`
                    : credentialsSummary(row.credentials)}
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
            Merged into <code>exchanges.{selected.exchange}.private</code> for strategies on this
            account.
          </p>
          <h4>Credentials</h4>
          <JsonBlock value={selected.credentials ?? {}} maxHeight="160px" />
          {selected.credential_storage === "legacy_env" ? (
            <>
              <h4 style={{ marginTop: "1rem" }}>Legacy env reference</h4>
              <JsonBlock value={selected.credential_ref} maxHeight="80px" />
            </>
          ) : null}
        </section>
      ) : null}
    </>
  );
}
