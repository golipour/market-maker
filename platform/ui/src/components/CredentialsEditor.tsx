import type { AuthMode, CredentialsFormState, ExchangeAuthSpec } from "../lib/accountCredentials";
import {
  AUTH_MODE_LABELS,
  supportedModesForExchange,
} from "../lib/accountCredentials";

interface CredentialsEditorProps {
  exchange: string;
  authSpec?: ExchangeAuthSpec | null;
  value: CredentialsFormState;
  onChange: (next: CredentialsFormState) => void;
}

export function CredentialsEditor({
  exchange,
  authSpec,
  value,
  onChange,
}: CredentialsEditorProps) {
  const modes = supportedModesForExchange(exchange, authSpec);
  const showSecret = value.auth_mode === "signed_key" || value.auth_mode === "signed_key_passphrase";
  const showPassphrase = value.auth_mode === "signed_key_passphrase";

  function setMode(auth_mode: AuthMode) {
    onChange({
      ...value,
      auth_mode,
      api_secret: auth_mode === "api_key" ? "" : value.api_secret,
      api_passphrase: auth_mode === "signed_key_passphrase" ? value.api_passphrase : "",
    });
  }

  return (
    <div className="form-field" style={{ gridColumn: "1 / -1" }}>
      <label htmlFor="acc-auth-mode">Credentials</label>
      <select
        id="acc-auth-mode"
        value={value.auth_mode}
        onChange={(e) => setMode(e.target.value as AuthMode)}
      >
        {modes.map((mode) => (
          <option key={mode} value={mode}>
            {AUTH_MODE_LABELS[mode]}
          </option>
        ))}
      </select>
      <p style={{ margin: "0.35rem 0 0.75rem", color: "var(--text-muted)", fontSize: "0.85rem" }}>
        Values are stored in the registry database for this account.
      </p>

      <div className="form-grid">
        <div className="form-field">
          <label htmlFor="acc-api-key">API key</label>
          <input
            id="acc-api-key"
            required
            autoComplete="off"
            placeholder="Public key or API key"
            value={value.api_key}
            onChange={(e) => onChange({ ...value, api_key: e.target.value })}
          />
        </div>

        {showSecret ? (
          <div className="form-field">
            <label htmlFor="acc-api-secret">API secret</label>
            <input
              id="acc-api-secret"
              required={!value.api_secret_set}
              type="password"
              autoComplete="new-password"
              placeholder={value.api_secret_set ? "Leave blank to keep existing" : "Private key or secret"}
              value={value.api_secret}
              onChange={(e) =>
                onChange({
                  ...value,
                  api_secret: e.target.value,
                  api_secret_set: e.target.value === "" ? value.api_secret_set : false,
                })
              }
            />
          </div>
        ) : null}

        {showPassphrase ? (
          <div className="form-field">
            <label htmlFor="acc-api-passphrase">API passphrase</label>
            <input
              id="acc-api-passphrase"
              required={!value.api_passphrase_set}
              type="password"
              autoComplete="new-password"
              placeholder={value.api_passphrase_set ? "Leave blank to keep existing" : "KuCoin passphrase"}
              value={value.api_passphrase}
              onChange={(e) =>
                onChange({
                  ...value,
                  api_passphrase: e.target.value,
                  api_passphrase_set: e.target.value === "" ? value.api_passphrase_set : false,
                })
              }
            />
          </div>
        ) : null}
      </div>
    </div>
  );
}
