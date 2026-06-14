import { useCallback, useEffect, useState } from "react";
import { ApiError, api, type Allocation, type RegistryList } from "../api/client";

export function AllocationsPage() {
  const [items, setItems] = useState<Allocation[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await api.get<RegistryList<Allocation>>("/api/admin/registry/allocations");
      setItems(data.items);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to load allocations");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  return (
    <>
      <header className="page-header">
        <h1>Allocations</h1>
        <p>Portfolio quote/base caps per strategy and breach halt policy.</p>
      </header>

      {error ? <div className="alert alert-error">{error}</div> : null}
      {loading ? <p className="loading">Loading…</p> : null}

      <div className="table-wrap">
        <table className="data-table">
          <thead>
            <tr>
              <th>Allocation</th>
              <th>Strategy</th>
              <th>Actor</th>
              <th>Account</th>
              <th>Quote</th>
              <th>Base</th>
              <th>Halt on breach</th>
              <th>Status</th>
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
              </tr>
            ))}
            {!loading && items.length === 0 ? (
              <tr>
                <td colSpan={8} style={{ color: "var(--text-muted)" }}>
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
