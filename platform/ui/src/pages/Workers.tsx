import { useCallback, useEffect, useState } from "react";
import { ApiError, api, type RegistryList, type WorkerAssignment } from "../api/client";

export function WorkersPage() {
  const [items, setItems] = useState<WorkerAssignment[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await api.get<RegistryList<WorkerAssignment>>(
        "/api/admin/registry/worker-assignments",
      );
      setItems(data.items);
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

  return (
    <>
      <header className="page-header">
        <h1>Workers</h1>
        <p>Strategy worker lease assignments and heartbeats.</p>
      </header>

      {error ? <div className="alert alert-error">{error}</div> : null}
      {loading ? <p className="loading">Loading…</p> : null}

      <div className="table-wrap">
        <table className="data-table">
          <thead>
            <tr>
              <th>Assignment</th>
              <th>Worker</th>
              <th>Strategies</th>
              <th>Lease TTL (s)</th>
              <th>Last heartbeat</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {items.map((row) => (
              <tr key={row.assignment_id}>
                <td>{row.assignment_id}</td>
                <td>{row.worker_id}</td>
                <td>{row.strategy_ids.join(", ") || "—"}</td>
                <td>{row.lease_ttl_seconds}</td>
                <td>
                  {row.heartbeat_at_ms
                    ? new Date(row.heartbeat_at_ms).toLocaleString()
                    : "—"}
                </td>
                <td>{row.status}</td>
              </tr>
            ))}
            {!loading && items.length === 0 ? (
              <tr>
                <td colSpan={6} style={{ color: "var(--text-muted)" }}>
                  No worker assignments.
                </td>
              </tr>
            ) : null}
          </tbody>
        </table>
      </div>
    </>
  );
}
