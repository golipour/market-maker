import { FormEvent, useCallback, useEffect, useState } from "react";
import {
  ApiError,
  api,
  type Exchange,
  type MarketFeed,
  type MarketFeedUpsertPayload,
  type RegistryList,
} from "../api/client";

const EMPTY: MarketFeedUpsertPayload = {
  feed_id: "",
  exchange: "",
  symbol: "USDT_IRT",
  interval_ms: 10000,
  mode: "rest",
  status: "active",
};

export function MarketFeedsPage() {
  const [items, setItems] = useState<MarketFeed[]>([]);
  const [exchanges, setExchanges] = useState<Exchange[]>([]);
  const [form, setForm] = useState<MarketFeedUpsertPayload>(EMPTY);
  const [showForm, setShowForm] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [feeds, exchangeRows] = await Promise.all([
        api.get<RegistryList<MarketFeed>>("/api/admin/registry/market-feeds"),
        api.get<RegistryList<Exchange>>("/api/admin/registry/exchanges"),
      ]);
      setItems(feeds.items);
      setExchanges(exchangeRows.items);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to load market feeds");
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
      const body: MarketFeedUpsertPayload = {
        ...form,
        feed_id: form.feed_id?.trim() || undefined,
        interval_ms: Number(form.interval_ms),
      };
      const saved = await api.post<MarketFeed>("/api/admin/registry/market-feeds", body);
      setMessage(`Feed ${saved.feed_id} saved`);
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
        <h1>Market feeds</h1>
        <p>
          Public market-data subscriptions published to{" "}
          <code>market:events:{"{exchange}"}:{"{symbol}"}:{"{interval_ms}"}</code>.
        </p>
      </header>
      <div className="page-toolbar">
        <button type="button" className="btn btn-primary" onClick={() => setShowForm(true)}>
          New feed
        </button>
        <button type="button" className="btn" onClick={() => void load()}>
          Refresh
        </button>
      </div>
      {error ? <div className="alert alert-error">{error}</div> : null}
      {message ? <div className="alert alert-success">{message}</div> : null}
      {showForm ? (
        <form onSubmit={(e) => void onSubmit(e)} className="card form-card">
          <h3>New market feed</h3>
          <div className="form-grid">
            <div className="form-field">
              <label htmlFor="feed-id">Feed ID</label>
              <input
                id="feed-id"
                value={form.feed_id ?? ""}
                onChange={(e) => setForm((p) => ({ ...p, feed_id: e.target.value }))}
                placeholder="auto-generated if empty"
              />
            </div>
            <div className="form-field">
              <label htmlFor="feed-exchange">Exchange</label>
              <select
                id="feed-exchange"
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
              <label htmlFor="feed-symbol">Symbol</label>
              <input
                id="feed-symbol"
                required
                value={form.symbol}
                onChange={(e) => setForm((p) => ({ ...p, symbol: e.target.value }))}
              />
            </div>
            <div className="form-field">
              <label htmlFor="feed-interval">Interval (ms)</label>
              <input
                id="feed-interval"
                type="number"
                required
                min={1}
                value={form.interval_ms}
                onChange={(e) => setForm((p) => ({ ...p, interval_ms: Number(e.target.value) }))}
              />
            </div>
            <div className="form-field">
              <label htmlFor="feed-mode">Mode</label>
              <input
                id="feed-mode"
                value={form.mode ?? "rest"}
                onChange={(e) => setForm((p) => ({ ...p, mode: e.target.value }))}
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
              <th>Feed</th>
              <th>Exchange</th>
              <th>Symbol</th>
              <th>Interval</th>
              <th>Mode</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {items.map((row) => (
              <tr key={row.feed_id}>
                <td>{row.feed_id}</td>
                <td>{row.exchange}</td>
                <td>{row.symbol}</td>
                <td>{row.interval_ms} ms</td>
                <td>{row.mode}</td>
                <td>{row.status}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}
