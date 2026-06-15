import { FormEvent, useCallback, useEffect, useState } from "react";
import {
  ApiError,
  api,
  type Account,
  type Allocation,
  type AllocationUpsertPayload,
  type RegistryList,
  type Strategy,
} from "../api/client";

const ALLOCATION_STATUSES = ["active", "paused", "retired"] as const;

const EMPTY_FORM: AllocationUpsertPayload = {
  allocation_id: "",
  strategy_id: "",
  actor_id: "",
  account_id: "",
  quote_asset: "IRT",
  base_asset: "USDT",
  allocated_quote: 0,
  allocated_base: 0,
  halt_on_breach: true,
  status: "active",
};

export function AllocationsPage() {
  const [items, setItems] = useState<Allocation[]>([]);
  const [strategies, setStrategies] = useState<Strategy[]>([]);
  const [accounts, setAccounts] = useState<Account[]>([]);
  const [form, setForm] = useState<AllocationUpsertPayload>(EMPTY_FORM);
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
      const [allocations, strategyRows, accountRows] = await Promise.all([
        api.get<RegistryList<Allocation>>("/api/admin/registry/allocations"),
        api.get<RegistryList<Strategy>>("/api/admin/registry/strategies"),
        api.get<RegistryList<Account>>("/api/admin/registry/accounts"),
      ]);
      setItems(allocations.items);
      setStrategies(strategyRows.items);
      setAccounts(accountRows.items);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to load allocations");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  function openCreate() {
    setEditingId(null);
    setForm(EMPTY_FORM);
    setShowForm(true);
    setMessage(null);
    setError(null);
  }

  function openEdit(row: Allocation) {
    setEditingId(row.allocation_id);
    setForm({
      allocation_id: row.allocation_id,
      strategy_id: row.strategy_id,
      actor_id: row.actor_id,
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

  function onStrategyChange(strategyId: string) {
    const strategy = strategies.find((row) => row.strategy_id === strategyId);
    if (!strategy) {
      setForm((prev) => ({ ...prev, strategy_id: strategyId }));
      return;
    }
    setForm((prev) => ({
      ...prev,
      strategy_id: strategyId,
      actor_id: strategy.actor_id,
      account_id: strategy.account_id,
      allocation_id: prev.allocation_id || `${strategyId}-allocation`,
    }));
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
      setMessage(`Allocation ${saved.allocation_id} saved`);
      setShowForm(false);
      setEditingId(null);
      setForm(EMPTY_FORM);
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
        <h1>Allocations</h1>
        <p>Virtual wallet per strategy instance — initial base/quote the MPC rebalances within.</p>
      </header>

      <div className="page-toolbar">
        <button type="button" className="btn btn-primary" onClick={openCreate}>
          New allocation
        </button>
        <button type="button" className="btn" onClick={() => void load()}>
          Refresh
        </button>
      </div>

      {error ? <div className="alert alert-error">{error}</div> : null}
      {message ? <div className="alert alert-success">{message}</div> : null}

      {showForm ? (
        <form onSubmit={(e) => void onSubmit(e)} className="card form-card">
          <h3>{editingId ? `Edit ${editingId}` : "New allocation"}</h3>
          <div className="form-grid">
            <div className="form-field">
              <label htmlFor="alloc-id">Allocation ID</label>
              <input
                id="alloc-id"
                value={form.allocation_id ?? ""}
                onChange={(e) => setForm((prev) => ({ ...prev, allocation_id: e.target.value }))}
                placeholder="auto from strategy if empty"
              />
            </div>
            <div className="form-field">
              <label htmlFor="alloc-strategy">Strategy</label>
              <select
                id="alloc-strategy"
                required
                value={form.strategy_id}
                onChange={(e) => onStrategyChange(e.target.value)}
              >
                <option value="">Select strategy…</option>
                {strategies.map((row) => (
                  <option key={row.strategy_id} value={row.strategy_id}>
                    {row.strategy_id} ({row.symbol})
                  </option>
                ))}
              </select>
            </div>
            <div className="form-field">
              <label htmlFor="alloc-actor">Actor ID</label>
              <input
                id="alloc-actor"
                required
                value={form.actor_id}
                onChange={(e) => setForm((prev) => ({ ...prev, actor_id: e.target.value }))}
              />
            </div>
            <div className="form-field">
              <label htmlFor="alloc-account">Account</label>
              <select
                id="alloc-account"
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
              <label htmlFor="alloc-quote-asset">Quote asset</label>
              <input
                id="alloc-quote-asset"
                value={form.quote_asset ?? ""}
                onChange={(e) => setForm((prev) => ({ ...prev, quote_asset: e.target.value }))}
              />
            </div>
            <div className="form-field">
              <label htmlFor="alloc-base-asset">Base asset</label>
              <input
                id="alloc-base-asset"
                value={form.base_asset ?? ""}
                onChange={(e) => setForm((prev) => ({ ...prev, base_asset: e.target.value }))}
              />
            </div>
            <div className="form-field">
              <label htmlFor="alloc-quote">Initial quote balance</label>
              <input
                id="alloc-quote"
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
              <label htmlFor="alloc-base">Initial base balance</label>
              <input
                id="alloc-base"
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
              <label htmlFor="alloc-status">Status</label>
              <select
                id="alloc-status"
                value={form.status ?? "active"}
                onChange={(e) => setForm((prev) => ({ ...prev, status: e.target.value }))}
              >
                {ALLOCATION_STATUSES.map((status) => (
                  <option key={status} value={status}>
                    {status}
                  </option>
                ))}
              </select>
            </div>
            <div className="form-field">
              <label htmlFor="alloc-halt">Halt on breach</label>
              <select
                id="alloc-halt"
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
              {saving ? "Saving…" : "Save allocation"}
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
              <th>Allocation</th>
              <th>Strategy</th>
              <th>Actor</th>
              <th>Account</th>
              <th>Initial quote</th>
              <th>Initial base</th>
              <th>Halt on breach</th>
              <th>Status</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {items.map((row) => (
              <tr key={row.allocation_id}>
                <td>{row.allocation_id}</td>
                <td>{row.strategy_id}</td>
                <td>{row.actor_id}</td>
                <td>{row.account_id}</td>
                <td>
                  {row.allocated_quote} {row.quote_asset}
                </td>
                <td>
                  {row.allocated_base} {row.base_asset}
                </td>
                <td>{row.halt_on_breach ? "yes" : "no"}</td>
                <td>{row.status}</td>
                <td>
                  <button type="button" className="btn" onClick={() => openEdit(row)}>
                    Edit
                  </button>
                </td>
              </tr>
            ))}
            {!loading && items.length === 0 ? (
              <tr>
                <td colSpan={9} style={{ color: "var(--text-muted)" }}>
                  No allocations configured.
                </td>
              </tr>
            ) : null}
          </tbody>
        </table>
      </div>
    </>
  );
}
