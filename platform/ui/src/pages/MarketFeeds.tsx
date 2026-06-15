import { FormEvent, useCallback, useEffect, useState } from "react";
import { EditModeNote } from "../components/EditModeNote";
import { JsonBlock } from "../components/JsonBlock";
import {
  ApiError,
  api,
  type Exchange,
  type MarketFeed,
  type MarketFeedUpsertPayload,
  type RegistryList,
} from "../api/client";

const STATUSES = ["active", "disabled"] as const;

const EMPTY: MarketFeedUpsertPayload = {
  feed_id: "",
  exchange: "",
  symbol: "USDT_IRT",
  interval_ms: 10000,
  mode: "rest",
  depth_limit: 100,
  trades_limit: 200,
  poll_interval_ms: 2000,
  status: "active",
};

export function MarketFeedsPage() {
  const [items, setItems] = useState<MarketFeed[]>([]);
  const [exchanges, setExchanges] = useState<Exchange[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [form, setForm] = useState<MarketFeedUpsertPayload>(EMPTY);
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

  const selected = items.find((row) => row.feed_id === selectedId) ?? null;

  function openCreate() {
    setEditingId(null);
    setForm(EMPTY);
    setShowForm(true);
    setMessage(null);
    setError(null);
  }

  function openEdit(row: MarketFeed) {
    setEditingId(row.feed_id);
    setForm({
      feed_id: row.feed_id,
      exchange: row.exchange,
      symbol: row.symbol,
      interval_ms: row.interval_ms,
      mode: row.mode,
      depth_limit: row.depth_limit,
      trades_limit: row.trades_limit,
      poll_interval_ms: row.poll_interval_ms,
      status: row.status,
    });
    setSelectedId(row.feed_id);
    setShowForm(true);
    setMessage(null);
    setError(null);
  }

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    setSaving(true);
    setError(null);
    try {
      const body: MarketFeedUpsertPayload = {
        ...form,
        feed_id: form.feed_id?.trim() || undefined,
        interval_ms: Number(form.interval_ms),
        depth_limit: Number(form.depth_limit),
        trades_limit: Number(form.trades_limit),
        poll_interval_ms: Number(form.poll_interval_ms),
      };
      const saved = await api.post<MarketFeed>("/api/admin/registry/market-feeds", body);
      setMessage(`Feed ${saved.feed_id} saved`);
      setShowForm(false);
      setEditingId(null);
      setSelectedId(saved.feed_id);
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
          Ingestor public market-data subscriptions. Publishes to{" "}
          <code>market:events:{"{exchange}"}:{"{symbol}"}:{"{interval_ms}"}</code>. These fields
          correspond to <code>exchanges.*.market_data</code> in the monolith JSON.
        </p>
        <EditModeNote mode="form" />
      </header>
      <div className="page-toolbar">
        <button type="button" className="btn btn-primary" onClick={openCreate}>
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
          <h3>{editingId ? `Edit ${editingId}` : "New market feed"}</h3>
          <div className="form-grid">
            <div className="form-field">
              <label htmlFor="feed-id">Feed ID</label>
              <input
                id="feed-id"
                readOnly={Boolean(editingId)}
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
              <label htmlFor="feed-interval">Frame interval (ms)</label>
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
            <div className="form-field">
              <label htmlFor="feed-depth">Depth limit</label>
              <input
                id="feed-depth"
                type="number"
                min={1}
                value={form.depth_limit ?? 100}
                onChange={(e) => setForm((p) => ({ ...p, depth_limit: Number(e.target.value) }))}
              />
            </div>
            <div className="form-field">
              <label htmlFor="feed-trades">Trades limit</label>
              <input
                id="feed-trades"
                type="number"
                min={1}
                value={form.trades_limit ?? 200}
                onChange={(e) => setForm((p) => ({ ...p, trades_limit: Number(e.target.value) }))}
              />
            </div>
            <div className="form-field">
              <label htmlFor="feed-poll">Poll interval (ms)</label>
              <input
                id="feed-poll"
                type="number"
                min={1}
                value={form.poll_interval_ms ?? 2000}
                onChange={(e) =>
                  setForm((p) => ({ ...p, poll_interval_ms: Number(e.target.value) }))
                }
              />
            </div>
            <div className="form-field">
              <label htmlFor="feed-status">Status</label>
              <select
                id="feed-status"
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
              <th>Feed</th>
              <th>Exchange</th>
              <th>Symbol</th>
              <th>Interval</th>
              <th>Poll</th>
              <th>Status</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {items.map((row) => (
              <tr key={row.feed_id}>
                <td>{row.feed_id}</td>
                <td>{row.exchange}</td>
                <td>{row.symbol}</td>
                <td>{row.interval_ms} ms</td>
                <td>{row.poll_interval_ms} ms</td>
                <td>{row.status}</td>
                <td style={{ whiteSpace: "nowrap" }}>
                  <button type="button" className="btn" onClick={() => setSelectedId(row.feed_id)}>
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
                  No market feeds configured.
                </td>
              </tr>
            ) : null}
          </tbody>
        </table>
      </div>
      {selected && !showForm ? (
        <section className="card" style={{ marginTop: "1rem" }}>
          <h3>{selected.feed_id}</h3>
          <p style={{ color: "var(--text-muted)", marginBottom: "0.75rem" }}>
            Topic{" "}
            <code>
              market:events:{selected.exchange}:{selected.symbol}:{selected.interval_ms}
            </code>
          </p>
          <JsonBlock value={selected} />
        </section>
      ) : null}
    </>
  );
}
