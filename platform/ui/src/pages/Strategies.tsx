import { FormEvent, useCallback, useEffect, useState } from "react";
import { JsonBlock } from "../components/JsonBlock";
import {
  ApiError,
  api,
  type Account,
  type Preset,
  type RegistryList,
  type Strategy,
  type StrategyRuntimeConfig,
  type StrategyUpsertPayload,
} from "../api/client";

const STRATEGY_STATUSES = ["draft", "active", "paused", "retired"] as const;

const EMPTY_CREATE: StrategyUpsertPayload = {
  strategy_id: "",
  actor_id: "",
  account_id: "",
  symbol: "USDT_IRT",
  preset_id: "",
  quoting_exchange: "nobitex",
  liquidity_exchange: "nobitex",
  status: "draft",
  config_overrides: {},
};

export function StrategiesPage() {
  const [items, setItems] = useState<Strategy[]>([]);
  const [accounts, setAccounts] = useState<Account[]>([]);
  const [presets, setPresets] = useState<Preset[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [detail, setDetail] = useState<Strategy | null>(null);
  const [runtimeConfig, setRuntimeConfig] = useState<StrategyRuntimeConfig | null>(null);
  const [overridesJson, setOverridesJson] = useState("{}");
  const [detailStatus, setDetailStatus] = useState("draft");
  const [createForm, setCreateForm] = useState<StrategyUpsertPayload>(EMPTY_CREATE);
  const [showCreate, setShowCreate] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [detailLoading, setDetailLoading] = useState(false);
  const [saving, setSaving] = useState(false);

  const loadList = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [strategies, accountRows, presetRows] = await Promise.all([
        api.get<RegistryList<Strategy>>("/api/admin/registry/strategies"),
        api.get<RegistryList<Account>>("/api/admin/registry/accounts"),
        api.get<RegistryList<Preset>>("/api/admin/registry/presets"),
      ]);
      setItems(strategies.items);
      setAccounts(accountRows.items);
      setPresets(presetRows.items);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to load strategies");
    } finally {
      setLoading(false);
    }
  }, []);

  const loadDetail = useCallback(async (strategyId: string) => {
    setDetailLoading(true);
    setError(null);
    setRuntimeConfig(null);
    try {
      const [strategy, runtime] = await Promise.all([
        api.get<Strategy>(`/api/admin/registry/strategies/${strategyId}`),
        api.get<StrategyRuntimeConfig>(
          `/api/admin/registry/strategies/${strategyId}/runtime-config`,
        ),
      ]);
      setDetail(strategy);
      setDetailStatus(strategy.status);
      setOverridesJson(JSON.stringify(strategy.config_overrides ?? {}, null, 2));
      setRuntimeConfig(runtime);
    } catch (err) {
      setDetail(null);
      setError(err instanceof ApiError ? err.message : "Failed to load strategy detail");
    } finally {
      setDetailLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadList();
  }, [loadList]);

  useEffect(() => {
    if (selectedId) {
      void loadDetail(selectedId);
    } else {
      setDetail(null);
      setRuntimeConfig(null);
    }
  }, [selectedId, loadDetail]);

  function onAccountChange(accountId: string, target: "create" | "detail") {
    const account = accounts.find((row) => row.account_id === accountId);
    const exchange = account?.exchange ?? "nobitex";
    if (target === "create") {
      setCreateForm((prev) => ({
        ...prev,
        account_id: accountId,
        quoting_exchange: exchange,
        liquidity_exchange: exchange,
      }));
    }
  }

  async function startStrategy(strategyId: string) {
    setMessage(null);
    setError(null);
    try {
      const result = await api.post<{ actor_id: string }>(
        `/api/admin/registry/strategies/${strategyId}/start`,
      );
      setMessage(`Start command accepted for actor ${result.actor_id}`);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Start failed");
    }
  }

  async function onCreateSubmit(event: FormEvent) {
    event.preventDefault();
    setSaving(true);
    setError(null);
    setMessage(null);
    try {
      const body: StrategyUpsertPayload = { ...createForm };
      const strategyId = createForm.strategy_id;
      await api.post<Strategy>("/api/admin/registry/strategies", body);
      setMessage(`Strategy ${strategyId} saved`);
      setShowCreate(false);
      setCreateForm(EMPTY_CREATE);
      await loadList();
      setSelectedId(strategyId);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Create failed");
    } finally {
      setSaving(false);
    }
  }

  async function onPatchDetail(event: FormEvent) {
    event.preventDefault();
    if (!selectedId) {
      return;
    }
    setSaving(true);
    setError(null);
    setMessage(null);
    try {
      const config_overrides = JSON.parse(overridesJson) as Record<string, unknown>;
      await api.patch<Strategy>(`/api/admin/registry/strategies/${selectedId}`, {
        status: detailStatus,
        config_overrides,
      });
      setMessage(`Strategy ${selectedId} updated`);
      await loadList();
      await loadDetail(selectedId);
    } catch (err) {
      if (err instanceof SyntaxError) {
        setError("Config overrides must be valid JSON");
      } else {
        setError(err instanceof ApiError ? err.message : "Update failed");
      }
    } finally {
      setSaving(false);
    }
  }

  return (
    <>
      <header className="page-header">
        <h1>Strategies</h1>
        <p>Registry strategies, resolved runtime config, and lifecycle controls.</p>
      </header>

      <div className="page-toolbar">
        <button
          type="button"
          className="btn btn-primary"
          onClick={() => setShowCreate((open) => !open)}
        >
          {showCreate ? "Cancel" : "New strategy"}
        </button>
        <button type="button" className="btn" onClick={() => void loadList()}>
          Refresh
        </button>
      </div>

      {error ? <div className="alert alert-error">{error}</div> : null}
      {message ? <div className="alert alert-success">{message}</div> : null}

      {showCreate ? (
        <form onSubmit={(e) => void onCreateSubmit(e)} className="card form-card">
          <h3>New strategy</h3>
          <div className="form-grid">
            <div className="form-field">
              <label htmlFor="create-strategy-id">Strategy ID</label>
              <input
                id="create-strategy-id"
                required
                value={createForm.strategy_id}
                onChange={(e) =>
                  setCreateForm((prev) => ({ ...prev, strategy_id: e.target.value }))
                }
              />
            </div>
            <div className="form-field">
              <label htmlFor="create-actor-id">Actor ID</label>
              <input
                id="create-actor-id"
                required
                value={createForm.actor_id}
                onChange={(e) =>
                  setCreateForm((prev) => ({ ...prev, actor_id: e.target.value }))
                }
              />
            </div>
            <div className="form-field">
              <label htmlFor="create-account">Account</label>
              <select
                id="create-account"
                required
                value={createForm.account_id}
                onChange={(e) => onAccountChange(e.target.value, "create")}
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
              <label htmlFor="create-preset">Preset</label>
              <select
                id="create-preset"
                required
                value={createForm.preset_id}
                onChange={(e) =>
                  setCreateForm((prev) => ({ ...prev, preset_id: e.target.value }))
                }
              >
                <option value="">Select preset…</option>
                {presets.map((row) => (
                  <option key={row.preset_id} value={row.preset_id}>
                    {row.preset_id}
                  </option>
                ))}
              </select>
            </div>
            <div className="form-field">
              <label htmlFor="create-symbol">Symbol</label>
              <input
                id="create-symbol"
                required
                value={createForm.symbol}
                onChange={(e) =>
                  setCreateForm((prev) => ({ ...prev, symbol: e.target.value }))
                }
              />
            </div>
            <div className="form-field">
              <label htmlFor="create-quoting">Quoting exchange</label>
              <input
                id="create-quoting"
                required
                value={createForm.quoting_exchange}
                onChange={(e) =>
                  setCreateForm((prev) => ({ ...prev, quoting_exchange: e.target.value }))
                }
              />
            </div>
            <div className="form-field">
              <label htmlFor="create-liquidity">Liquidity exchange</label>
              <input
                id="create-liquidity"
                required
                value={createForm.liquidity_exchange}
                onChange={(e) =>
                  setCreateForm((prev) => ({ ...prev, liquidity_exchange: e.target.value }))
                }
              />
            </div>
            <div className="form-field">
              <label htmlFor="create-status">Status</label>
              <select
                id="create-status"
                value={createForm.status ?? "draft"}
                onChange={(e) =>
                  setCreateForm((prev) => ({ ...prev, status: e.target.value }))
                }
              >
                {STRATEGY_STATUSES.map((status) => (
                  <option key={status} value={status}>
                    {status}
                  </option>
                ))}
              </select>
            </div>
          </div>
          <button type="submit" className="btn btn-primary" disabled={saving}>
            {saving ? "Saving…" : "Create strategy"}
          </button>
        </form>
      ) : null}

      <div className="split-layout">
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
              </tr>
            </thead>
            <tbody>
              {items.map((row) => (
                <tr
                  key={row.strategy_id}
                  className={selectedId === row.strategy_id ? "row-selected" : "row-clickable"}
                  onClick={() => setSelectedId(row.strategy_id)}
                >
                  <td>{row.strategy_id}</td>
                  <td>{row.actor_id}</td>
                  <td>{row.symbol}</td>
                  <td>{row.account_id}</td>
                  <td>{row.preset_id}</td>
                  <td>{row.status}</td>
                </tr>
              ))}
              {!loading && items.length === 0 ? (
                <tr>
                  <td colSpan={6} style={{ color: "var(--text-muted)" }}>
                    No strategies registered. Seed via registry or create one above.
                  </td>
                </tr>
              ) : null}
            </tbody>
          </table>
        </div>

        <aside className="detail-panel">
          {!selectedId ? (
            <p className="loading">Select a strategy to view config.</p>
          ) : detailLoading ? (
            <p className="loading">Loading detail…</p>
          ) : detail ? (
            <>
              <div className="detail-panel-header">
                <h3>{detail.strategy_id}</h3>
                <button
                  type="button"
                  className="btn btn-primary"
                  onClick={() => void startStrategy(detail.strategy_id)}
                >
                  Start
                </button>
              </div>

              <dl className="detail-dl">
                <dt>Actor</dt>
                <dd>{detail.actor_id}</dd>
                <dt>Account</dt>
                <dd>{detail.account_id}</dd>
                <dt>Symbol</dt>
                <dd>{detail.symbol}</dd>
                <dt>Preset</dt>
                <dd>{detail.preset_id}</dd>
                <dt>Quoting / liquidity</dt>
                <dd>
                  {detail.quoting_exchange} / {detail.liquidity_exchange}
                </dd>
              </dl>

              <form onSubmit={(e) => void onPatchDetail(e)} className="detail-form">
                <div className="form-row">
                  <div className="form-field">
                    <label htmlFor="detail-status">Status</label>
                    <select
                      id="detail-status"
                      value={detailStatus}
                      onChange={(e) => setDetailStatus(e.target.value)}
                    >
                      {STRATEGY_STATUSES.map((status) => (
                        <option key={status} value={status}>
                          {status}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
                <div className="form-field">
                  <label htmlFor="detail-overrides">Config overrides (JSON)</label>
                  <textarea
                    id="detail-overrides"
                    className="json-textarea"
                    value={overridesJson}
                    onChange={(e) => setOverridesJson(e.target.value)}
                  />
                </div>
                <button type="submit" className="btn btn-primary" disabled={saving}>
                  {saving ? "Saving…" : "Save changes"}
                </button>
              </form>

              {runtimeConfig ? (
                <>
                  <h4 className="detail-section-title">Resolved runtime contract</h4>
                  <JsonBlock value={runtimeConfig.resolved_contract} maxHeight="280px" />
                  <h4 className="detail-section-title">Full runtime JSON</h4>
                  <JsonBlock value={runtimeConfig.runtime_config} />
                </>
              ) : null}
            </>
          ) : null}
        </aside>
      </div>

      {loading ? <p className="loading">Loading…</p> : null}
    </>
  );
}
