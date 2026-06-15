import { FormEvent, useCallback, useEffect, useState } from "react";
import { JsonBlock } from "./JsonBlock";
import {
  ApiError,
  api,
  type Account,
  type Allocation,
  type AllocationUpsertPayload,
  type RegistryList,
} from "../api/client";

const WALLET_STATUSES = ["active", "paused", "retired"] as const;

const EMPTY_WALLET: AllocationUpsertPayload = {
  allocation_id: "",
  account_id: "",
  quote_asset: "IRT",
  base_asset: "USDT",
  allocated_quote: 0,
  allocated_base: 0,
  halt_on_breach: true,
  status: "active",
};

type WalletPoolsPanelProps = {
  accounts: Account[];
  onWalletsChange?: (wallets: Allocation[]) => void;
};

export function WalletPoolsPanel({ accounts, onWalletsChange }: WalletPoolsPanelProps) {
  const [items, setItems] = useState<Allocation[]>([]);
  const [form, setForm] = useState<AllocationUpsertPayload>(EMPTY_WALLET);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [showForm, setShowForm] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (items.length > 0) {
      onWalletsChange?.(items);
    }
  }, [items, onWalletsChange]);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const rows = await api.get<RegistryList<Allocation>>("/api/admin/registry/allocations");
      setItems(rows.items);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to load capital pools");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const selected = items.find((row) => row.allocation_id === selectedId) ?? null;

  function openCreate() {
    setEditingId(null);
    setForm(EMPTY_WALLET);
    setShowForm(true);
    setMessage(null);
    setError(null);
  }

  function openEdit(row: Allocation) {
    setEditingId(row.allocation_id);
    setSelectedId(row.allocation_id);
    setForm({
      allocation_id: row.allocation_id,
      account_id: row.account_id,
      quote_asset: row.quote_asset,
      base_asset: row.base_asset,
      allocated_quote: row.allocated_quote,
      allocated_base: row.allocated_base,
      halt_on_breach: row.halt_on_breach,
      status: row.status,
    });
    setShowForm(true);
    setMessage(null);
    setError(null);
  }

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    setSaving(true);
    setError(null);
    setMessage(null);
    try {
      const body: AllocationUpsertPayload = {
        ...form,
        allocation_id: form.allocation_id?.trim() || undefined,
        allocated_quote: Number(form.allocated_quote) || 0,
        allocated_base: Number(form.allocated_base) || 0,
      };
      const saved = await api.post<Allocation>("/api/admin/registry/allocations", body);
      setMessage(`Capital pool ${saved.allocation_id} saved`);
      setShowForm(false);
      setEditingId(null);
      setSelectedId(saved.allocation_id);
      await load();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Save failed");
    } finally {
      setSaving(false);
    }
  }

  return (
    <section id="capital-pools" className="card" style={{ marginTop: "1.5rem" }}>
      <header className="page-header" style={{ marginBottom: "0.75rem" }}>
        <h2 style={{ margin: 0 }}>Capital pools</h2>
        <p style={{ margin: "0.35rem 0 0" }}>
          Initial base/quote budgets assigned to strategies using virtual sizing. Actor attribution
          is set when you link a pool on a strategy.
        </p>
      </header>
      <div className="page-toolbar">
        <button type="button" className="btn btn-primary" onClick={openCreate}>
          New pool
        </button>
        <button type="button" className="btn" onClick={() => void load()}>
          Refresh pools
        </button>
      </div>
      {error ? <div className="alert alert-error">{error}</div> : null}
      {message ? <div className="alert alert-success">{message}</div> : null}
      {showForm ? (
        <form onSubmit={(e) => void onSubmit(e)} className="card form-card" style={{ marginTop: "0.75rem" }}>
          <h3>{editingId ? `Edit ${editingId}` : "New capital pool"}</h3>
          <div className="form-grid">
            <div className="form-field">
              <label htmlFor="pool-id">Pool ID</label>
              <input
                id="pool-id"
                value={form.allocation_id ?? ""}
                onChange={(e) => setForm((prev) => ({ ...prev, allocation_id: e.target.value }))}
                placeholder="e.g. nobitex-usdt-pool"
              />
            </div>
            <div className="form-field">
              <label htmlFor="pool-account">Account</label>
              <select
                id="pool-account"
                required
                value={form.account_id}
                onChange={(e) => setForm((prev) => ({ ...prev, account_id: e.target.value }))}
              >
                <option value="">Select account…</option>
                {accounts.map((row) => (
                  <option key={row.account_id} value={row.account_id}>
                    {row.account_id} ({row.exchange})
                  </option>
                ))}
              </select>
            </div>
            <div className="form-field">
              <label htmlFor="pool-quote-asset">Quote asset</label>
              <input
                id="pool-quote-asset"
                value={form.quote_asset ?? ""}
                onChange={(e) => setForm((prev) => ({ ...prev, quote_asset: e.target.value }))}
              />
            </div>
            <div className="form-field">
              <label htmlFor="pool-base-asset">Base asset</label>
              <input
                id="pool-base-asset"
                value={form.base_asset ?? ""}
                onChange={(e) => setForm((prev) => ({ ...prev, base_asset: e.target.value }))}
              />
            </div>
            <div className="form-field">
              <label htmlFor="pool-quote">Initial quote</label>
              <input
                id="pool-quote"
                type="number"
                min={0}
                step="any"
                value={form.allocated_quote ?? 0}
                onChange={(e) =>
                  setForm((prev) => ({ ...prev, allocated_quote: Number(e.target.value) }))
                }
              />
            </div>
            <div className="form-field">
              <label htmlFor="pool-base">Initial base</label>
              <input
                id="pool-base"
                type="number"
                min={0}
                step="any"
                value={form.allocated_base ?? 0}
                onChange={(e) =>
                  setForm((prev) => ({ ...prev, allocated_base: Number(e.target.value) }))
                }
              />
            </div>
            <div className="form-field">
              <label htmlFor="pool-status">Status</label>
              <select
                id="pool-status"
                value={form.status ?? "active"}
                onChange={(e) => setForm((prev) => ({ ...prev, status: e.target.value }))}
              >
                {WALLET_STATUSES.map((status) => (
                  <option key={status} value={status}>
                    {status}
                  </option>
                ))}
              </select>
            </div>
            <div className="form-field">
              <label htmlFor="pool-halt">Halt on breach</label>
              <select
                id="pool-halt"
                value={form.halt_on_breach ? "yes" : "no"}
                onChange={(e) =>
                  setForm((prev) => ({ ...prev, halt_on_breach: e.target.value === "yes" }))
                }
              >
                <option value="yes">yes</option>
                <option value="no">no</option>
              </select>
            </div>
          </div>
          <div className="form-actions">
            <button type="submit" className="btn btn-primary" disabled={saving}>
              {saving ? "Saving…" : "Save pool"}
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
      {loading ? <p className="loading">Loading pools…</p> : null}
      <div className="table-wrap" style={{ marginTop: "0.75rem" }}>
        <table className="data-table">
          <thead>
            <tr>
              <th>Pool</th>
              <th>Account</th>
              <th>Initial quote</th>
              <th>Initial base</th>
              <th>Status</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {items.map((row) => (
              <tr key={row.allocation_id}>
                <td>{row.allocation_id}</td>
                <td>{row.account_id}</td>
                <td>
                  {row.allocated_quote} {row.quote_asset}
                </td>
                <td>
                  {row.allocated_base} {row.base_asset}
                </td>
                <td>{row.status}</td>
                <td style={{ whiteSpace: "nowrap" }}>
                  <button type="button" className="btn" onClick={() => setSelectedId(row.allocation_id)}>
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
                  No capital pools yet.
                </td>
              </tr>
            ) : null}
          </tbody>
        </table>
      </div>
      {selected && !showForm ? (
        <div style={{ marginTop: "0.75rem" }}>
          <JsonBlock value={selected} />
        </div>
      ) : null}
    </section>
  );
}
