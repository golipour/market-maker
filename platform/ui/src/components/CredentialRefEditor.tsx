import type { CredentialFormState, CredentialMode } from "../lib/credentialRef";

const MODES: { id: CredentialMode; label: string; hint: string }[] = [
  {
    id: "single_token",
    label: "Single API token env",
    hint: "One .env variable (maps to api_token_env).",
  },
  {
    id: "token_and_private_key",
    label: "API token + private key",
    hint: "Nobitex, Tabdeal, Wallex-style token + signing key.",
  },
  {
    id: "key_and_private_key",
    label: "API key + private key",
    hint: "Exchanges that use api_key_env instead of api_token_env.",
  },
  {
    id: "advanced_json",
    label: "Advanced JSON",
    hint: "Raw credential_ref object with api_token_env / api_key_env / api_private_key_env.",
  },
];

interface CredentialRefEditorProps {
  value: CredentialFormState;
  onChange: (next: CredentialFormState) => void;
}

export function CredentialRefEditor({ value, onChange }: CredentialRefEditorProps) {
  function setMode(mode: CredentialMode) {
    onChange({ ...value, mode });
  }

  return (
    <div className="form-field" style={{ gridColumn: "1 / -1" }}>
      <label htmlFor="acc-cred-mode">Credentials (.env variable names only)</label>
      <select
        id="acc-cred-mode"
        value={value.mode}
        onChange={(e) => setMode(e.target.value as CredentialMode)}
      >
        {MODES.map((mode) => (
          <option key={mode.id} value={mode.id}>
            {mode.label}
          </option>
        ))}
      </select>
      <p style={{ margin: "0.35rem 0 0.75rem", color: "var(--text-muted)", fontSize: "0.85rem" }}>
        {MODES.find((m) => m.id === value.mode)?.hint}
      </p>

      {value.mode === "single_token" ? (
        <input
          id="acc-cred-single"
          required
          placeholder="NOBITEX_API_TOKEN"
          value={value.singleEnv}
          onChange={(e) => onChange({ ...value, singleEnv: e.target.value })}
        />
      ) : null}

      {value.mode === "token_and_private_key" ? (
        <div className="form-grid">
          <div className="form-field">
            <label htmlFor="acc-cred-token">API token env</label>
            <input
              id="acc-cred-token"
              required
              placeholder="NOBITEX_API_TOKEN"
              value={value.tokenEnv}
              onChange={(e) => onChange({ ...value, tokenEnv: e.target.value })}
            />
          </div>
          <div className="form-field">
            <label htmlFor="acc-cred-private">API private key env</label>
            <input
              id="acc-cred-private"
              required
              placeholder="NOBITEX_API_PRIVATE_KEY"
              value={value.privateKeyEnv}
              onChange={(e) => onChange({ ...value, privateKeyEnv: e.target.value })}
            />
          </div>
        </div>
      ) : null}

      {value.mode === "key_and_private_key" ? (
        <div className="form-grid">
          <div className="form-field">
            <label htmlFor="acc-cred-key">API key env</label>
            <input
              id="acc-cred-key"
              required
              placeholder="EXCHANGE_API_KEY"
              value={value.keyEnv}
              onChange={(e) => onChange({ ...value, keyEnv: e.target.value })}
            />
          </div>
          <div className="form-field">
            <label htmlFor="acc-cred-private2">API private key env</label>
            <input
              id="acc-cred-private2"
              required
              placeholder="EXCHANGE_API_PRIVATE_KEY"
              value={value.privateKeyEnv}
              onChange={(e) => onChange({ ...value, privateKeyEnv: e.target.value })}
            />
          </div>
        </div>
      ) : null}

      {value.mode === "advanced_json" ? (
        <textarea
          id="acc-cred-json"
          rows={5}
          value={value.advancedJson}
          onChange={(e) => onChange({ ...value, advancedJson: e.target.value })}
          placeholder='{"api_token_env":"NOBITEX_API_TOKEN","api_private_key_env":"NOBITEX_API_PRIVATE_KEY"}'
        />
      ) : null}
    </div>
  );
}
