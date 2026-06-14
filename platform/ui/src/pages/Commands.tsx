import { FormEvent, useState } from "react";
import { ApiError, api, type AdminCommandPayload, type CommandResult } from "../api/client";

const COMMANDS = [
  "pause",
  "resume",
  "kill_switch",
  "cancel_all",
  "dry_run",
  "observe_only",
  "reload_risk",
  "pause_local",
  "resume_local",
  "pause_global",
  "resume_global",
  "cancel_actor_orders",
];

export function CommandsPage() {
  const [command, setCommand] = useState("pause");
  const [actorId, setActorId] = useState("");
  const [exchange, setExchange] = useState("");
  const [reason, setReason] = useState("");
  const [enabled, setEnabled] = useState<string>("");
  const [payloadJson, setPayloadJson] = useState("{}");
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<CommandResult | null>(null);
  const [submitting, setSubmitting] = useState(false);

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    setSubmitting(true);
    setError(null);
    setResult(null);
    try {
      let payload: Record<string, unknown> = {};
      if (payloadJson.trim()) {
        payload = JSON.parse(payloadJson) as Record<string, unknown>;
      }
      const body: AdminCommandPayload = {
        command,
        actor_id: actorId || null,
        exchange: exchange || null,
        reason,
        payload,
      };
      if (enabled === "true") {
        body.enabled = true;
      } else if (enabled === "false") {
        body.enabled = false;
      }
      const response = await api.post<CommandResult>("/api/admin/commands", body);
      setResult(response);
    } catch (err) {
      if (err instanceof SyntaxError) {
        setError("Payload must be valid JSON");
      } else {
        setError(err instanceof ApiError ? err.message : "Command failed");
      }
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <>
      <header className="page-header">
        <h1>Commands</h1>
        <p>Publish admin commands to the runtime event bus.</p>
      </header>

      {error ? <div className="alert alert-error">{error}</div> : null}
      {result ? (
        <div className="alert alert-success">
          Command accepted — id {result.command_id}
        </div>
      ) : null}

      <form onSubmit={(e) => void onSubmit(e)} className="card" style={{ maxWidth: 640 }}>
        <div className="form-row">
          <div className="form-field">
            <label htmlFor="cmd">Command</label>
            <select id="cmd" value={command} onChange={(e) => setCommand(e.target.value)}>
              {COMMANDS.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>
          </div>
          <div className="form-field">
            <label htmlFor="actor">Actor ID</label>
            <input
              id="actor"
              value={actorId}
              onChange={(e) => setActorId(e.target.value)}
              placeholder="optional"
            />
          </div>
          <div className="form-field">
            <label htmlFor="exchange">Exchange</label>
            <input
              id="exchange"
              value={exchange}
              onChange={(e) => setExchange(e.target.value)}
              placeholder="optional"
            />
          </div>
          <div className="form-field">
            <label htmlFor="enabled">Enabled</label>
            <select id="enabled" value={enabled} onChange={(e) => setEnabled(e.target.value)}>
              <option value="">—</option>
              <option value="true">true</option>
              <option value="false">false</option>
            </select>
          </div>
        </div>
        <div className="form-field" style={{ marginBottom: "0.75rem" }}>
          <label htmlFor="reason">Reason</label>
          <input
            id="reason"
            value={reason}
            onChange={(e) => setReason(e.target.value)}
            placeholder="optional audit reason"
          />
        </div>
        <div className="form-field" style={{ marginBottom: "1rem" }}>
          <label htmlFor="payload">Payload (JSON)</label>
          <textarea
            id="payload"
            value={payloadJson}
            onChange={(e) => setPayloadJson(e.target.value)}
          />
        </div>
        <button type="submit" className="btn btn-primary" disabled={submitting}>
          {submitting ? "Sending…" : "Send command"}
        </button>
      </form>
    </>
  );
}
