import { FormEvent, useCallback, useEffect, useState } from "react";
import { EditModeNote } from "../components/EditModeNote";
import { JsonBlock } from "../components/JsonBlock";
import {
  ApiError,
  api,
  type Exchange,
  type MarketFeed,
  type MarketFeedUpsertPayload,
  type MarketSource,
  type MarketSourceUpsertPayload,
  type RegistryList,
} from "../api/client";

const STATUSES = ["active", "disabled"] as const;

const EMPTY_SOURCE: MarketSourceUpsertPayload = {
  source_id: "",
  exchange: "",
  symbol: "USDT_IRT",
  status: "active",
};

const EMPTY_FEED: MarketFeedUpsertPayload = {
  feed_id: "",
  source_id: "",
  interval_ms: 10000,
  status: "active",
};

export function MarketFeedsPage() {
  const [sources, setSources] = useState<MarketSource[]>([]);
  const [feeds, setFeeds] = useState<MarketFeed[]>([]);
  const [exchanges, setExchanges] = useState<Exchange[]>([]);
  const [selectedSourceId, setSelectedSourceId] = useState<string | null>(null);
  const [selectedFeedId, setSelectedFeedId] = useState<string | null>(null);
  const [sourceForm, setSourceForm] = useState<MarketSourceUpsertPayload>(EMPTY_SOURCE);
  const [feedForm, setFeedForm] = useState<MarketFeedUpsertPayload>(EMPTY_FEED);
  const [editingSourceId, setEditingSourceId] = useState<string | null>(null);
  const [editingFeedId, setEditingFeedId] = useState<string | null>(null);
  const [showSourceForm, setShowSourceForm] = useState(false);
  const [showFeedForm, setShowFeedForm] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [exchangeError, setExchangeError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    setExchangeError(null);
    try {
      const [sourceRows, feedRows] = await Promise.all([
        api.get<RegistryList<MarketSource>>("/api/admin/registry/market-sources"),
        api.get<RegistryList<MarketFeed>>("/api/admin/registry/market-feeds"),
      ]);
      setSources(sourceRows.items);
      setFeeds(feedRows.items);
    } catch (err) {
      setError(
        err instanceof ApiError
          ? err.message
          : "Failed to load market data config — is the admin API running on :8080?",
      );
    }
    try {
      const exchangeRows = await api.get<RegistryList<Exchange>>("/api/admin/registry/exchanges");
      setExchanges(exchangeRows.items);
    } catch (err) {
      setExchangeError(
        err instanceof ApiError ? err.message : "Failed to load exchanges for the feed form",
      );
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const selectedSource = sources.find((row) => row.source_id === selectedSourceId) ?? null;
  const selectedFeed = feeds.find((row) => row.feed_id === selectedFeedId) ?? null;

  function openCreateSource() {
    setEditingSourceId(null);
    setSourceForm(EMPTY_SOURCE);
    setShowSourceForm(true);
    setMessage(null);
    setError(null);
  }

  function openEditSource(row: MarketSource) {
    setEditingSourceId(row.source_id);
    setSourceForm({
      source_id: row.source_id,
      exchange: row.exchange,
      symbol: row.symbol,
      status: row.status,
    });
    setSelectedSourceId(row.source_id);
    setShowSourceForm(true);
    setMessage(null);
    setError(null);
  }

  function openCreateFeed() {
    setEditingFeedId(null);
    setFeedForm({
      ...EMPTY_FEED,
      source_id: sources[0]?.source_id ?? "",
    });
    setShowFeedForm(true);
    setMessage(null);
    setError(null);
  }

  function openEditFeed(row: MarketFeed) {
    setEditingFeedId(row.feed_id);
    setFeedForm({
      feed_id: row.feed_id,
      source_id: row.source_id,
      interval_ms: row.interval_ms,
      status: row.status,
    });
    setSelectedFeedId(row.feed_id);
    setShowFeedForm(true);
    setMessage(null);
    setError(null);
  }

  async function onSubmitSource(event: FormEvent) {
    event.preventDefault();
    setSaving(true);
    setError(null);
    try {
      const body: MarketSourceUpsertPayload = {
        ...sourceForm,
        source_id: sourceForm.source_id?.trim() || undefined,
      };
      const saved = await api.post<MarketSource>("/api/admin/registry/market-sources", body);
      setMessage(`Market source ${saved.source_id} saved`);
      setShowSourceForm(false);
      setEditingSourceId(null);
      setSelectedSourceId(saved.source_id);
      await load();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Save failed");
    } finally {
      setSaving(false);
    }
  }

  async function onSubmitFeed(event: FormEvent) {
    event.preventDefault();
    setSaving(true);
    setError(null);
    try {
      const body: MarketFeedUpsertPayload = {
        ...feedForm,
        feed_id: feedForm.feed_id?.trim() || undefined,
        interval_ms: Number(feedForm.interval_ms),
      };
      const saved = await api.post<MarketFeed>("/api/admin/registry/market-feeds", body);
      setMessage(`Frame interval ${saved.feed_id} saved`);
      setShowFeedForm(false);
      setEditingFeedId(null);
      setSelectedFeedId(saved.feed_id);
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
        <h1>Market data</h1>
        <p>
          One <strong>market source</strong> identifies an exchange/symbol pair for the ingestor.
          Poll/depth/trades/mode come from the exchange&apos;s <code>config.market_data</code>{" "}
          (edit under <strong>Exchanges</strong>). Multiple <strong>frame intervals</strong> publish
          aggregated <code>market:events:…</code> topics from the same source without duplicate API
          polling.
        </p>
        <EditModeNote mode="form" />
      </header>
      <div className="page-toolbar">
        <button type="button" className="btn" onClick={() => void load()}>
          Refresh
        </button>
      </div>
      {error ? <div className="alert alert-error">{error}</div> : null}
      {exchangeError ? <div className="alert alert-error">{exchangeError}</div> : null}
      {message ? <div className="alert alert-success">{message}</div> : null}

      <section style={{ marginTop: "1rem" }}>
        <div className="page-toolbar">
          <h2 style={{ margin: 0, fontSize: "1.1rem" }}>Market sources</h2>
          <button type="button" className="btn btn-primary" onClick={openCreateSource}>
            New source
          </button>
        </div>
        {showSourceForm ? (
          <form onSubmit={(e) => void onSubmitSource(e)} className="card form-card">
            <h3>{editingSourceId ? `Edit ${editingSourceId}` : "New market source"}</h3>
            <p style={{ color: "var(--text-muted)", marginTop: 0 }}>
              Polling settings are inherited from the exchange&apos;s <code>market_data</code> config.
            </p>
            <div className="form-grid">
              <div className="form-field">
                <label htmlFor="source-id">Source ID</label>
                <input
                  id="source-id"
                  readOnly={Boolean(editingSourceId)}
                  value={sourceForm.source_id ?? ""}
                  onChange={(e) => setSourceForm((p) => ({ ...p, source_id: e.target.value }))}
                  placeholder="auto-generated if empty"
                />
              </div>
              <div className="form-field">
                <label htmlFor="source-exchange">Exchange</label>
                <select
                  id="source-exchange"
                  required
                  value={sourceForm.exchange}
                  onChange={(e) => setSourceForm((p) => ({ ...p, exchange: e.target.value }))}
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
                <label htmlFor="source-symbol">Symbol</label>
                <input
                  id="source-symbol"
                  required
                  value={sourceForm.symbol}
                  onChange={(e) => setSourceForm((p) => ({ ...p, symbol: e.target.value }))}
                />
              </div>
              <div className="form-field">
                <label htmlFor="source-status">Status</label>
                <select
                  id="source-status"
                  value={sourceForm.status ?? "active"}
                  onChange={(e) => setSourceForm((p) => ({ ...p, status: e.target.value }))}
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
                {saving ? "Saving…" : "Save source"}
              </button>
              <button
                type="button"
                className="btn"
                onClick={() => {
                  setShowSourceForm(false);
                  setEditingSourceId(null);
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
                <th>Source</th>
                <th>Exchange</th>
                <th>Symbol</th>
                <th>Poll (from exchange)</th>
                <th>Depth</th>
                <th>Status</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {sources.map((row) => (
                <tr key={row.source_id}>
                  <td>{row.source_id}</td>
                  <td>{row.exchange}</td>
                  <td>{row.symbol}</td>
                  <td>{row.poll_interval_ms} ms</td>
                  <td>{row.depth_limit}</td>
                  <td>{row.status}</td>
                  <td style={{ whiteSpace: "nowrap" }}>
                    <button
                      type="button"
                      className="btn"
                      onClick={() => setSelectedSourceId(row.source_id)}
                    >
                      View
                    </button>{" "}
                    <button type="button" className="btn" onClick={() => openEditSource(row)}>
                      Edit
                    </button>
                  </td>
                </tr>
              ))}
              {!loading && sources.length === 0 ? (
                <tr>
                  <td colSpan={7} style={{ color: "var(--text-muted)" }}>
                    No market sources configured.
                  </td>
                </tr>
              ) : null}
            </tbody>
          </table>
        </div>
        {selectedSource && !showSourceForm ? (
          <section className="card" style={{ marginTop: "1rem" }}>
            <h3>{selectedSource.source_id}</h3>
            <p style={{ color: "var(--text-muted)", marginBottom: "0.75rem" }}>
              Raw topics{" "}
              <code>
                market:raw:{selectedSource.exchange}:{selectedSource.symbol}:orderbook
              </code>
              . Poll {selectedSource.poll_interval_ms} ms, depth {selectedSource.depth_limit},{" "}
              trades {selectedSource.trades_limit} from exchange <code>market_data</code>.
            </p>
            <JsonBlock value={selectedSource} />
          </section>
        ) : null}
      </section>

      <section style={{ marginTop: "2rem" }}>
        <div className="page-toolbar">
          <h2 style={{ margin: 0, fontSize: "1.1rem" }}>Frame intervals</h2>
          <button
            type="button"
            className="btn btn-primary"
            onClick={openCreateFeed}
            disabled={sources.length === 0}
          >
            New interval
          </button>
        </div>
        {showFeedForm ? (
          <form onSubmit={(e) => void onSubmitFeed(e)} className="card form-card">
            <h3>{editingFeedId ? `Edit ${editingFeedId}` : "New frame interval"}</h3>
            <div className="form-grid">
              <div className="form-field">
                <label htmlFor="feed-id">Feed ID</label>
                <input
                  id="feed-id"
                  readOnly={Boolean(editingFeedId)}
                  value={feedForm.feed_id ?? ""}
                  onChange={(e) => setFeedForm((p) => ({ ...p, feed_id: e.target.value }))}
                  placeholder="auto-generated if empty"
                />
              </div>
              <div className="form-field">
                <label htmlFor="feed-source">Market source</label>
                <select
                  id="feed-source"
                  required
                  value={feedForm.source_id}
                  onChange={(e) => setFeedForm((p) => ({ ...p, source_id: e.target.value }))}
                >
                  <option value="">Select…</option>
                  {sources.map((row) => (
                    <option key={row.source_id} value={row.source_id}>
                      {row.source_id} ({row.exchange} / {row.symbol})
                    </option>
                  ))}
                </select>
              </div>
              <div className="form-field">
                <label htmlFor="feed-interval">Frame interval (ms)</label>
                <input
                  id="feed-interval"
                  type="number"
                  required
                  min={1}
                  value={feedForm.interval_ms}
                  onChange={(e) =>
                    setFeedForm((p) => ({ ...p, interval_ms: Number(e.target.value) }))
                  }
                />
              </div>
              <div className="form-field">
                <label htmlFor="feed-status">Status</label>
                <select
                  id="feed-status"
                  value={feedForm.status ?? "active"}
                  onChange={(e) => setFeedForm((p) => ({ ...p, status: e.target.value }))}
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
                {saving ? "Saving…" : "Save interval"}
              </button>
              <button
                type="button"
                className="btn"
                onClick={() => {
                  setShowFeedForm(false);
                  setEditingFeedId(null);
                }}
              >
                Cancel
              </button>
            </div>
          </form>
        ) : null}
        <div className="table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>Feed</th>
                <th>Source</th>
                <th>Exchange</th>
                <th>Symbol</th>
                <th>Interval</th>
                <th>Status</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {feeds.map((row) => (
                <tr key={row.feed_id}>
                  <td>{row.feed_id}</td>
                  <td>{row.source_id}</td>
                  <td>{row.exchange}</td>
                  <td>{row.symbol}</td>
                  <td>{row.interval_ms} ms</td>
                  <td>{row.status}</td>
                  <td style={{ whiteSpace: "nowrap" }}>
                    <button type="button" className="btn" onClick={() => setSelectedFeedId(row.feed_id)}>
                      View
                    </button>{" "}
                    <button type="button" className="btn" onClick={() => openEditFeed(row)}>
                      Edit
                    </button>
                  </td>
                </tr>
              ))}
              {!loading && feeds.length === 0 ? (
                <tr>
                  <td colSpan={7} style={{ color: "var(--text-muted)" }}>
                    No frame intervals configured.
                  </td>
                </tr>
              ) : null}
            </tbody>
          </table>
        </div>
        {selectedFeed && !showFeedForm ? (
          <section className="card" style={{ marginTop: "1rem" }}>
            <h3>{selectedFeed.feed_id}</h3>
            <p style={{ color: "var(--text-muted)", marginBottom: "0.75rem" }}>
              Topic{" "}
              <code>
                market:events:{selectedFeed.exchange}:{selectedFeed.symbol}:{selectedFeed.interval_ms}
              </code>
            </p>
            <JsonBlock value={selectedFeed} />
          </section>
        ) : null}
      </section>
    </>
  );
}
