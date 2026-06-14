import { useCallback, useEffect, useState } from "react";
import { ApiError, api, type RegistryList, type Strategy } from "../api/client";

export function StrategiesPage() {
  const [items, setItems] = useState<Strategy[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await api.get<RegistryList<Strategy>>("/api/admin/registry/strategies");
      setItems(data.items);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to load strategies");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  async function startStrategy(strategyId: string) {
    setMessage(null);
    setError(null);
    try {
      const result = await api.post<{ actor_id: string }>(
        `/api/admin/registry/strategies/${strategyId}/start`,
      );
      setMessage(`Start command accepted for actor ${result.actor_id}`);
      await load();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Start failed");
    }
  }

  return (
    <>
      <header className="page-header">
        <h1>Strategies</h1>
        <p>Registered strategies and runtime start controls.</p>
      </header>

      {error ? <div className="alert alert-error">{error}</div> : null}
      {message ? <div className="alert alert-success">{message}</div> : null}
      {loading ? <p className="loading">Loading…</p> : null}

      <div className="table-wrap">
        <table className="data-table">
          <thead>
            <tr>
              <th>Strategy</th>
              <th>Actor</th>
              <th>Symbol</th>
              <th>Account</th>
              <th>Preset</th>
              <th>Status</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {items.map((row) => (
              <tr key={row.strategy_id}>
                <td>{row.strategy_id}</td>
                <td>{row.actor_id}</td>
                <td>{row.symbol}</td>
                <td>{row.account_id}</td>
                <td>{row.preset_id}</td>
                <td>{row.status}</td>
                <td>
                  <button
                    type="button"
                    className="btn btn-primary"
                    onClick={() => void startStrategy(row.strategy_id)}
                  >
                    Start
                  </button>
                </td>
              </tr>
            ))}
            {!loading && items.length === 0 ? (
              <tr>
                <td colSpan={7} style={{ color: "var(--text-muted)" }}>
                  No strategies registered. Seed via the registry API or admin CLI.
                </td>
              </tr>
            ) : null}
          </tbody>
        </table>
      </div>
    </>
  );
}
