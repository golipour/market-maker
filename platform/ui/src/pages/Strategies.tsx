import { FormEvent, useCallback, useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { EditModeNote } from "../components/EditModeNote";
import { JsonBlock } from "../components/JsonBlock";
import { WalletPoolsPanel } from "../components/WalletPoolsPanel";
import {
  ApiError,
  api,
  type Account,
  type Allocation,
  type Preset,
  type RegistryList,
  type RuntimeOverview,
  type Strategy,
  type StrategyRuntimeConfig,
  type StrategyUpsertPayload,
} from "../api/client";
import {
  canStartTrading,
  canStopTrading,
  runtimeByStrategyId,
  tradingStatusClass,
  tradingStatusLabel,
} from "../lib/runtimeStatus";

const STRATEGY_STATUSES = ["draft", "active", "paused", "retired"] as const;
const SIZING_SOURCES = [
  { value: "physical", label: "Physical — full exchange account balance" },
  { value: "virtual", label: "Virtual — capital pool + attributed fills" },
] as const;

const EMPTY_FORM: StrategyUpsertPayload = {
  strategy_id: "",
  actor_id: "",
  account_id: "",
  symbol: "USDT_IRT",
  preset_id: "",
  quoting_exchange: "nobitex",
  liquidity_exchange: "nobitex",
  status: "draft",
  config_overrides: {},
  wallet_id: null,
  ledger_source: "physical",
};

export function StrategiesPage() {
  const [items, setItems] = useState<Strategy[]>([]);
  const [accounts, setAccounts] = useState<Account[]>([]);
  const [presets, setPresets] = useState<Preset[]>([]);
  const [wallets, setWallets] = useState<Allocation[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [detail, setDetail] = useState<Strategy | null>(null);
  const [runtimeConfig, setRuntimeConfig] = useState<StrategyRuntimeConfig | null>(null);
  const [form, setForm] = useState<StrategyUpsertPayload>(EMPTY_FORM);
  const [overridesJson, setOverridesJson] = useState("{}");
  const [editingId, setEditingId] = useState<string | null>(null);
  const [showForm, setShowForm] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [detailLoading, setDetailLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [runtime, setRuntime] = useState<RuntimeOverview | null>(null);
  const [runtimeLoading, setRuntimeLoading] = useState(false);
  const [controlBusy, setControlBusy] = useState<string | null>(null);

  const runtimeByStrategy = runtime ? runtimeByStrategyId(runtime.strategies) : new Map();

  const loadList = useCallback(async () => {
    setLoading(true);
    setError(null);
    const failures: string[] = [];
    try {
      const strategies = await api.get<RegistryList<Strategy>>("/api/admin/registry/strategies");
      setItems(strategies.items);
    } catch (err) {
      failures.push(
        err instanceof ApiError ? `strategies: ${err.message}` : "strategies: failed to load",
      );
    }
    try {
      const accountRows = await api.get<RegistryList<Account>>("/api/admin/registry/accounts");
      setAccounts(accountRows.items);
    } catch (err) {
      failures.push(
        err instanceof ApiError ? `accounts: ${err.message}` : "accounts: failed to load",
      );
    }
    try {
      const presetRows = await api.get<RegistryList<Preset>>("/api/admin/registry/presets");
      setPresets(presetRows.items);
    } catch (err) {
      failures.push(err instanceof ApiError ? `presets: ${err.message}` : "presets: failed to load");
    }
    try {
      const walletRows = await api.get<RegistryList<Allocation>>("/api/admin/registry/allocations");
      setWallets(walletRows.items);
    } catch (err) {
      failures.push(
        err instanceof ApiError ? `capital pools: ${err.message}` : "capital pools: failed to load",
      );
    }
    if (failures.length > 0) {
      setError(failures.join("; "));
    }
    setLoading(false);
  }, []);

  const loadRuntime = useCallback(async () => {
    setRuntimeLoading(true);
    try {
      const overview = await api.get<RuntimeOverview>("/api/admin/runtime/overview");
      setRuntime(overview);
    } catch {
      /* keep last snapshot; list endpoints may still work */
    } finally {
      setRuntimeLoading(false);
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
    void loadRuntime();
    const timer = window.setInterval(() => void loadRuntime(), 5000);
    return () => window.clearInterval(timer);
  }, [loadList, loadRuntime]);

  useEffect(() => {
    if (selectedId) {
      void loadDetail(selectedId);
    } else {
      setDetail(null);
      setRuntimeConfig(null);
    }
  }, [selectedId, loadDetail]);

  function onAccountChange(accountId: string) {
    const account = accounts.find((row) => row.account_id === accountId);
    const exchange = account?.exchange ?? form.quoting_exchange;
    setForm((prev) => ({
      ...prev,
      account_id: accountId,
      quoting_exchange: exchange,
      liquidity_exchange: exchange,
    }));
  }

  function openCreate() {
    setEditingId(null);
    setForm(EMPTY_FORM);
    setOverridesJson("{}");
    setShowForm(true);
    setMessage(null);
    setError(null);
  }

  function openEdit(row: Strategy) {
    setEditingId(row.strategy_id);
    setForm({
      strategy_id: row.strategy_id,
      actor_id: row.actor_id,
      account_id: row.account_id,
      symbol: row.symbol,
      preset_id: row.preset_id,
      quoting_exchange: row.quoting_exchange,
      liquidity_exchange: row.liquidity_exchange,
      status: row.status,
      config_overrides: row.config_overrides ?? {},
      wallet_id: row.wallet_id ?? null,
      ledger_source: row.ledger_source ?? "virtual",
    });
    setOverridesJson(JSON.stringify(row.config_overrides ?? {}, null, 2));
    setSelectedId(row.strategy_id);
    setShowForm(true);
    setMessage(null);
    setError(null);
  }

  async function startStrategy(strategyId: string) {
    setControlBusy(strategyId);
    setMessage(null);
    setError(null);
    try {
      const result = await api.post<{ actor_id: string }>(
        `/api/admin/registry/strategies/${strategyId}/start`,
      );
      setMessage(`Start accepted for actor ${result.actor_id} (reload config + resume)`);
      await loadRuntime();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Start failed");
    } finally {
      setControlBusy(null);
    }
  }

  async function stopStrategy(strategyId: string) {
    setControlBusy(strategyId);
    setMessage(null);
    setError(null);
    try {
      const result = await api.post<{ actor_id: string }>(
        `/api/admin/registry/strategies/${strategyId}/stop`,
      );
      setMessage(`Stop accepted for actor ${result.actor_id} (pause trading)`);
      await loadRuntime();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Stop failed");
    } finally {
      setControlBusy(null);
    }
  }

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    const ledgerSource = form.ledger_source === "physical" ? "physical" : "virtual";
    const walletId = ledgerSource === "virtual" ? form.wallet_id?.trim() || null : null;
    if (ledgerSource === "virtual" && !walletId) {
      const legacyPool = wallets.find((row) => row.strategy_id === form.strategy_id);
      if (!legacyPool) {
        setError(
          "Virtual sizing requires a capital pool. Create one below or pick an existing pool.",
        );
        return;
      }
    }
    setSaving(true);
    setError(null);
    setMessage(null);
    try {
      const config_overrides = JSON.parse(overridesJson) as Record<string, unknown>;
      const body: StrategyUpsertPayload = {
        ...form,
        config_overrides,
        ledger_source: ledgerSource,
        wallet_id: walletId,
      };
      const saved = await api.post<Strategy>("/api/admin/registry/strategies", body);
      setMessage(`Strategy ${saved.strategy_id} saved`);
      setShowForm(false);
      setEditingId(null);
      setSelectedId(saved.strategy_id);
      await loadList();
      await loadDetail(saved.strategy_id);
    } catch (err) {
      if (err instanceof SyntaxError) {
        setError("Config overrides must be valid JSON");
      } else {
        setError(err instanceof ApiError ? err.message : "Save failed");
      }
    } finally {
      setSaving(false);
    }
  }

  return (
    <>
      <header className="page-header">
        <h1>Strategies</h1>
        <p>
          Bot instances, sizing ledger (physical account vs virtual capital pool), and linked
          presets. <strong>Stop</strong> pauses trading. <strong>Start</strong> reloads the strategy
          config from the registry, then resumes. Workers boot paused — Start is required after a
          worker restart.
        </p>
        <EditModeNote mode="hybrid" />
      </header>

      <div className="page-toolbar">
        <button type="button" className="btn btn-primary" onClick={openCreate}>
          New strategy
        </button>
        <button type="button" className="btn" onClick={() => void loadList()}>
          Refresh
        </button>
        <button type="button" className="btn" onClick={() => void loadRuntime()} disabled={runtimeLoading}>
          {runtimeLoading ? "Refreshing live…" : "Refresh live status"}
        </button>
      </div>

      {runtime && !runtime.control.connected ? (
        <div className="alert" style={{ borderColor: "var(--warning)", color: "var(--warning)" }}>
          Worker runtime state not seen yet — start the worker and send a control command. Lease
          heartbeats still show worker process liveness.
        </div>
      ) : null}

      {error ? <div className="alert alert-error">{error}</div> : null}
      {message ? <div className="alert alert-success">{message}</div> : null}

      {showForm ? (
        <form onSubmit={(e) => void onSubmit(e)} className="card form-card">
          <h3>{editingId ? `Edit ${editingId}` : "New strategy"}</h3>
          <div className="form-grid">
            <div className="form-field">
              <label htmlFor="strategy-id">Strategy ID</label>
              <input
                id="strategy-id"
                required
                readOnly={Boolean(editingId)}
                value={form.strategy_id}
                onChange={(e) => setForm((prev) => ({ ...prev, strategy_id: e.target.value }))}
              />
            </div>
            <div className="form-field">
              <label htmlFor="strategy-actor-id">Actor ID</label>
              <input
                id="strategy-actor-id"
                required
                value={form.actor_id}
                onChange={(e) => setForm((prev) => ({ ...prev, actor_id: e.target.value }))}
              />
            </div>
            <div className="form-field">
              <label htmlFor="strategy-account">Account</label>
              <select
                id="strategy-account"
                required
                value={form.account_id}
                onChange={(e) => onAccountChange(e.target.value)}
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
              <label htmlFor="strategy-preset">Preset</label>
              <select
                id="strategy-preset"
                required
                value={form.preset_id}
                onChange={(e) => setForm((prev) => ({ ...prev, preset_id: e.target.value }))}
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
              <label htmlFor="strategy-symbol">Symbol</label>
              <input
                id="strategy-symbol"
                required
                value={form.symbol}
                onChange={(e) => setForm((prev) => ({ ...prev, symbol: e.target.value }))}
              />
            </div>
            <div className="form-field">
              <label htmlFor="strategy-quoting">Quoting exchange</label>
              <input
                id="strategy-quoting"
                required
                value={form.quoting_exchange}
                onChange={(e) =>
                  setForm((prev) => ({ ...prev, quoting_exchange: e.target.value }))
                }
              />
            </div>
            <div className="form-field">
              <label htmlFor="strategy-liquidity">Liquidity exchange</label>
              <input
                id="strategy-liquidity"
                required
                value={form.liquidity_exchange}
                onChange={(e) =>
                  setForm((prev) => ({ ...prev, liquidity_exchange: e.target.value }))
                }
              />
            </div>
            <div className="form-field">
              <label htmlFor="strategy-status">Status</label>
              <select
                id="strategy-status"
                value={form.status ?? "draft"}
                onChange={(e) => setForm((prev) => ({ ...prev, status: e.target.value }))}
              >
                {STRATEGY_STATUSES.map((status) => (
                  <option key={status} value={status}>
                    {status}
                  </option>
                ))}
              </select>
            </div>
            <div className="form-field form-field-full">
              <label htmlFor="strategy-sizing">Sizing for MPC / risk</label>
              <select
                id="strategy-sizing"
                value={form.ledger_source ?? "physical"}
                onChange={(e) => {
                  const ledger_source = e.target.value;
                  setForm((prev) => ({
                    ...prev,
                    ledger_source,
                    wallet_id: ledger_source === "physical" ? null : prev.wallet_id,
                  }));
                }}
              >
                {SIZING_SOURCES.map((source) => (
                  <option key={source.value} value={source.value}>
                    {source.label}
                  </option>
                ))}
              </select>
            </div>
            {form.ledger_source === "virtual" ? (
              <div className="form-field form-field-full">
                <label htmlFor="strategy-wallet">Capital pool</label>
                <select
                  id="strategy-wallet"
                  required
                  value={form.wallet_id ?? ""}
                  onChange={(e) =>
                    setForm((prev) => ({
                      ...prev,
                      wallet_id: e.target.value || null,
                    }))
                  }
                >
                  <option value="">Select pool…</option>
                  {wallets
                    .filter((row) => !form.account_id || row.account_id === form.account_id)
                    .map((row) => (
                      <option key={row.allocation_id} value={row.allocation_id}>
                        {row.allocation_id} ({row.allocated_base} {row.base_asset} /{" "}
                        {row.allocated_quote} {row.quote_asset})
                      </option>
                    ))}
                </select>
                <p style={{ color: "var(--text-muted)", fontSize: "0.85rem", marginTop: "0.35rem" }}>
                  Create pools in the section below. Virtual sizing uses pool balances plus fills
                  attributed to this strategy&apos;s actor.
                </p>
              </div>
            ) : null}
          </div>
          <div className="form-field">
            <label htmlFor="strategy-overrides">Config overrides (JSON)</label>
            <textarea
              id="strategy-overrides"
              className="json-textarea"
              rows={8}
              value={overridesJson}
              onChange={(e) => setOverridesJson(e.target.value)}
            />
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
                <th>Sizing</th>
                <th>Registry</th>
                <th>Trading</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {items.map((row) => {
                const live = runtimeByStrategy.get(row.strategy_id);
                const trading = live?.trading_status;
                const busy = controlBusy === row.strategy_id;
                return (
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
                  <td>
                    {row.ledger_source === "virtual"
                      ? `virtual${row.wallet_id ? ` (${row.wallet_id})` : ""}`
                      : "physical"}
                  </td>
                  <td>{row.status}</td>
                  <td>
                    {trading ? (
                      <span className={`status-badge ${tradingStatusClass(trading)}`}>
                        {tradingStatusLabel(trading)}
                      </span>
                    ) : (
                      <span style={{ color: "var(--text-muted)" }}>—</span>
                    )}
                  </td>
                  <td style={{ whiteSpace: "nowrap" }}>
                    <button
                      type="button"
                      className="btn btn-primary"
                      disabled={busy || !canStartTrading(trading)}
                      title={
                        canStartTrading(trading)
                          ? "Resume trading (worker must be running)"
                          : "Not resumable in current state"
                      }
                      onClick={(e) => {
                        e.stopPropagation();
                        void startStrategy(row.strategy_id);
                      }}
                    >
                      Start
                    </button>{" "}
                    <button
                      type="button"
                      className="btn"
                      disabled={busy || !canStopTrading(trading)}
                      title={
                        canStopTrading(trading)
                          ? "Pause trading"
                          : "Not running"
                      }
                      onClick={(e) => {
                        e.stopPropagation();
                        void stopStrategy(row.strategy_id);
                      }}
                    >
                      Stop
                    </button>{" "}
                    <button
                      type="button"
                      className="btn"
                      onClick={(e) => {
                        e.stopPropagation();
                        setSelectedId(row.strategy_id);
                      }}
                    >
                      View
                    </button>{" "}
                    <button
                      type="button"
                      className="btn"
                      onClick={(e) => {
                        e.stopPropagation();
                        openEdit(row);
                      }}
                    >
                      Edit
                    </button>
                  </td>
                </tr>
              );
              })}
              {!loading && items.length === 0 ? (
                <tr>
                  <td colSpan={9} style={{ color: "var(--text-muted)" }}>
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
              {(() => {
                const live = runtimeByStrategy.get(detail.strategy_id);
                const trading = live?.trading_status;
                const busy = controlBusy === detail.strategy_id;
                return (
              <div className="detail-panel-header">
                <h3>{detail.strategy_id}</h3>
                <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", alignItems: "center" }}>
                  {trading ? (
                    <span className={`status-badge ${tradingStatusClass(trading)}`}>
                      {tradingStatusLabel(trading)}
                    </span>
                  ) : null}
                  <button
                    type="button"
                    className="btn"
                    onClick={() => openEdit(detail)}
                  >
                    Edit
                  </button>
                  <button
                    type="button"
                    className="btn btn-primary"
                    disabled={busy || !canStartTrading(trading)}
                    onClick={() => void startStrategy(detail.strategy_id)}
                  >
                    Start
                  </button>
                  <button
                    type="button"
                    className="btn"
                    disabled={busy || !canStopTrading(trading)}
                    onClick={() => void stopStrategy(detail.strategy_id)}
                  >
                    Stop
                  </button>
                </div>
              </div>
                );
              })()}

              <dl className="detail-dl">
                <dt>Actor</dt>
                <dd>{detail.actor_id}</dd>
                <dt>Account</dt>
                <dd>{detail.account_id}</dd>
                <dt>Symbol</dt>
                <dd>{detail.symbol}</dd>
                <dt>Preset</dt>
                <dd>
                  {detail.preset_id}{" "}
                  <Link
                    to={`/presets?preset_id=${encodeURIComponent(detail.preset_id)}`}
                    className="btn"
                    style={{ marginLeft: "0.35rem", padding: "0.1rem 0.5rem", fontSize: "0.85rem" }}
                  >
                    Edit preset
                  </Link>
                </dd>
                <dt>Quoting / liquidity</dt>
                <dd>
                  {detail.quoting_exchange} / {detail.liquidity_exchange}
                </dd>
                <dt>Status</dt>
                <dd>{detail.status}</dd>
                <dt>Worker</dt>
                <dd>
                  {runtimeByStrategy.get(detail.strategy_id)?.worker_id ?? "—"}
                  {runtimeByStrategy.get(detail.strategy_id)?.worker_live
                    ? ` (${runtimeByStrategy.get(detail.strategy_id)?.worker_live})`
                    : ""}
                </dd>
                <dt>Trading</dt>
                <dd>
                  {runtimeByStrategy.get(detail.strategy_id)?.trading_status
                    ? tradingStatusLabel(
                        runtimeByStrategy.get(detail.strategy_id)!.trading_status,
                      )
                    : "—"}
                </dd>
                <dt>Wallet</dt>
                <dd>{detail.wallet_id ?? "—"}</dd>
                <dt>Sizing</dt>
                <dd>{detail.ledger_source === "virtual" ? "virtual (capital pool)" : "physical (exchange account)"}</dd>
              </dl>

              {Object.keys(detail.config_overrides ?? {}).length > 0 ? (
                <>
                  <h4 className="detail-section-title">Config overrides</h4>
                  <JsonBlock value={detail.config_overrides ?? {}} maxHeight="160px" />
                </>
              ) : null}

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

      <WalletPoolsPanel accounts={accounts} onWalletsChange={setWallets} />
    </>
  );
}
