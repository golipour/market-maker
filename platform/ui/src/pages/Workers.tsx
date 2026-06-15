import { FormEvent, useCallback, useEffect, useState } from "react";
import { EditModeNote } from "../components/EditModeNote";
import { JsonBlock } from "../components/JsonBlock";
import {
  ApiError,
  api,
  type RegistryList,
  type Strategy,
  type WorkerAssignment,
  type WorkerAssignmentUpsertPayload,
} from "../api/client";

const STATUSES = ["active", "released", "expired"] as const;

const EMPTY: WorkerAssignmentUpsertPayload = {
  assignment_id: "",
  worker_id: "",
  strategy_ids: [],
  lease_ttl_seconds: 300,
  status: "active",
};

function formatTime(value?: string | null): string {
  if (!value) {
    return "—";
  }
  const parsed = Date.parse(value);
  if (Number.isNaN(parsed)) {
    return value;
  }
  return new Date(parsed).toLocaleString();
}

function leaseRemainingSeconds(leaseExpiresAt?: string | null): string {
  if (!leaseExpiresAt) {
    return "—";
  }
  const expires = Date.parse(leaseExpiresAt);
  if (Number.isNaN(expires)) {
    return "—";
  }
  const remaining = Math.max(0, Math.round((expires - Date.now()) / 1000));
  return String(remaining);
}

export function WorkersPage() {
  const [items, setItems] = useState<WorkerAssignment[]>([]);
  const [strategies, setStrategies] = useState<Strategy[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [form, setForm] = useState<WorkerAssignmentUpsertPayload>(EMPTY);
  const [strategyPick, setStrategyPick] = useState("");
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
      const [assignments, strategyRows] = await Promise.all([
        api.get<RegistryList<WorkerAssignment>>("/api/admin/registry/worker-assignments"),
        api.get<RegistryList<Strategy>>("/api/admin/registry/strategies"),
      ]);
      setItems(assignments.items);
      setStrategies(strategyRows.items);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to load worker assignments");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
    const timer = window.setInterval(() => void load(), 10000);
    return () => window.clearInterval(timer);
  }, [load]);

  const selected = items.find((row) => row.assignment_id === selectedId) ?? null;

  function openCreate() {
    setEditingId(null);
    setForm(EMPTY);
    setStrategyPick("");
    setShowForm(true);
    setMessage(null);
    setError(null);
  }

  function openEdit(row: WorkerAssignment) {
    setEditingId(row.assignment_id);
    setForm({
      assignment_id: row.assignment_id,
      worker_id: row.worker_id,
      strategy_ids: [...row.strategy_ids],
      lease_ttl_seconds: 300,
      status: row.status,
    });
    setSelectedId(row.assignment_id);
    setStrategyPick("");
    setShowForm(true);
    setMessage(null);
    setError(null);
  }

  function addStrategy() {
    if (!strategyPick || form.strategy_ids.includes(strategyPick)) {
      return;
    }
    setForm((prev) => ({ ...prev, strategy_ids: [...prev.strategy_ids, strategyPick] }));
    setStrategyPick("");
  }

  function removeStrategy(strategyId: string) {
    setForm((prev) => ({
      ...prev,
      strategy_ids: prev.strategy_ids.filter((id) => id !== strategyId),
    }));
  }

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    setSaving(true);
    setError(null);
    try {
      const body: WorkerAssignmentUpsertPayload = {
        ...form,
        assignment_id: form.assignment_id?.trim() || undefined,
        lease_ttl_seconds: Number(form.lease_ttl_seconds) || 300,
      };
      const saved = await api.post<WorkerAssignment>(
        "/api/admin/registry/worker-assignments",
        body,
      );
      setMessage(`Assignment ${saved.assignment_id} saved`);
      setShowForm(false);
      setEditingId(null);
      setSelectedId(saved.assignment_id);
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
        <h1>Workers</h1>
        <p>Strategy worker lease assignments and heartbeats.</p>
        <EditModeNote mode="form" />
      </header>

      <div className="page-toolbar">
        <button type="button" className="btn btn-primary" onClick={openCreate}>
          New assignment
        </button>
        <button type="button" className="btn" onClick={() => void load()}>
          Refresh
        </button>
      </div>

      {error ? <div className="alert alert-error">{error}</div> : null}
      {message ? <div className="alert alert-success">{message}</div> : null}

      {showForm ? (
        <form onSubmit={(e) => void onSubmit(e)} className="card form-card">
          <h3>{editingId ? `Edit ${editingId}` : "New worker assignment"}</h3>
          <div className="form-grid">
            <div className="form-field">
              <label htmlFor="wa-id">Assignment ID</label>
              <input
                id="wa-id"
                readOnly={Boolean(editingId)}
                value={form.assignment_id ?? ""}
                onChange={(e) => setForm((p) => ({ ...p, assignment_id: e.target.value }))}
                placeholder="optional"
              />
            </div>
            <div className="form-field">
              <label htmlFor="wa-worker">Worker ID</label>
              <input
                id="wa-worker"
                required
                value={form.worker_id}
                onChange={(e) => setForm((p) => ({ ...p, worker_id: e.target.value }))}
              />
            </div>
            <div className="form-field">
              <label htmlFor="wa-ttl">Lease TTL (s)</label>
              <input
                id="wa-ttl"
                type="number"
                min={30}
                value={form.lease_ttl_seconds ?? 300}
                onChange={(e) =>
                  setForm((p) => ({ ...p, lease_ttl_seconds: Number(e.target.value) }))
                }
              />
            </div>
            <div className="form-field">
              <label htmlFor="wa-status">Status</label>
              <select
                id="wa-status"
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
            <div className="form-field" style={{ gridColumn: "1 / -1" }}>
              <label htmlFor="wa-strategy-pick">Strategies</label>
              <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
                <select
                  id="wa-strategy-pick"
                  value={strategyPick}
                  onChange={(e) => setStrategyPick(e.target.value)}
                >
                  <option value="">Add strategy…</option>
                  {strategies.map((row) => (
                    <option key={row.strategy_id} value={row.strategy_id}>
                      {row.strategy_id}
                    </option>
                  ))}
                </select>
                <button type="button" className="btn" onClick={addStrategy}>
                  Add
                </button>
              </div>
              <ul style={{ marginTop: "0.5rem", paddingLeft: "1.2rem" }}>
                {form.strategy_ids.map((strategyId) => (
                  <li key={strategyId}>
                    {strategyId}{" "}
                    <button
                      type="button"
                      className="btn"
                      style={{ marginLeft: "0.35rem", padding: "0.1rem 0.4rem" }}
                      onClick={() => removeStrategy(strategyId)}
                    >
                      remove
                    </button>
                  </li>
                ))}
                {form.strategy_ids.length === 0 ? (
                  <li style={{ color: "var(--text-muted)", listStyle: "none", marginLeft: "-1.2rem" }}>
                    No strategies selected
                  </li>
                ) : null}
              </ul>
            </div>
          </div>
          <div className="form-actions">
            <button type="submit" className="btn btn-primary" disabled={saving}>
              {saving ? "Saving…" : "Save assignment"}
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
              <th>Assignment</th>
              <th>Worker</th>
              <th>Strategies</th>
              <th>Lease remaining (s)</th>
              <th>Last heartbeat</th>
              <th>Status</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {items.map((row) => (
              <tr key={row.assignment_id}>
                <td>{row.assignment_id}</td>
                <td>{row.worker_id}</td>
                <td>{row.strategy_ids.join(", ") || "—"}</td>
                <td>{leaseRemainingSeconds(row.lease_expires_at)}</td>
                <td>{formatTime(row.heartbeat_at)}</td>
                <td>{row.status}</td>
                <td style={{ whiteSpace: "nowrap" }}>
                  <button type="button" className="btn" onClick={() => setSelectedId(row.assignment_id)}>
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
                <td colSpan={7} style={{ color: "var(--text-muted)" }}>
                  No worker assignments.
                </td>
              </tr>
            ) : null}
          </tbody>
        </table>
      </div>
      {selected && !showForm ? (
        <section className="card" style={{ marginTop: "1rem" }}>
          <h3>{selected.assignment_id}</h3>
          <p style={{ color: "var(--text-muted)", marginBottom: "0.75rem" }}>
            Worker <code>{selected.worker_id}</code> · lease expires in{" "}
            {leaseRemainingSeconds(selected.lease_expires_at)}s · last heartbeat{" "}
            {formatTime(selected.heartbeat_at)}
          </p>
          <JsonBlock value={selected} />
        </section>
      ) : null}
    </>
  );
}
