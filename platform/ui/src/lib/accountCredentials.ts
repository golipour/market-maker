/** Unified account credentials — mirrors mm.registry.credentials + mm.exchanges.auth */

export type AuthMode = "api_key" | "signed_key" | "signed_key_passphrase";

const AUTH_MODES: readonly AuthMode[] = ["api_key", "signed_key", "signed_key_passphrase"];

export function isAuthMode(value: unknown): value is AuthMode {
  return typeof value === "string" && (AUTH_MODES as readonly string[]).includes(value);
}

export function asAuthMode(value: unknown, fallback: AuthMode = "api_key"): AuthMode {
  return isAuthMode(value) ? value : fallback;
}

export const AUTH_MODE_LABELS: Record<AuthMode, string> = {
  api_key: "API key",
  signed_key: "API key + secret",
  signed_key_passphrase: "API key + secret + passphrase",
};

export interface ExchangeAuthSpec {
  exchange: string;
  /** API may send a plain string; helpers normalize via asAuthMode. */
  default_mode: AuthMode | string;
  supported_modes: Array<AuthMode | string>;
  mode_labels: Partial<Record<AuthMode, string>> & Record<string, string>;
}

export interface AccountCredentials {
  auth_mode: AuthMode;
  api_key: string;
  api_secret?: string;
  api_passphrase?: string;
  api_secret_set?: boolean;
  api_passphrase_set?: boolean;
}

export interface CredentialsFormState {
  auth_mode: AuthMode;
  api_key: string;
  api_secret: string;
  api_passphrase: string;
  api_secret_set: boolean;
  api_passphrase_set: boolean;
}

export const EMPTY_CREDENTIALS_FORM: CredentialsFormState = {
  auth_mode: "api_key",
  api_key: "",
  api_secret: "",
  api_passphrase: "",
  api_secret_set: false,
  api_passphrase_set: false,
};

const MASK = "********";

export function defaultAuthModeForExchange(
  exchange: string,
  authSpec?: ExchangeAuthSpec | null,
): AuthMode {
  if (authSpec?.default_mode) {
    return asAuthMode(authSpec.default_mode, "signed_key");
  }
  const key = exchange.toLowerCase();
  if (key === "kucoin") {
    return "signed_key_passphrase";
  }
  if (key === "wallex") {
    return "api_key";
  }
  return "signed_key";
}

export function supportedModesForExchange(
  exchange: string,
  authSpec?: ExchangeAuthSpec | null,
): AuthMode[] {
  if (authSpec?.supported_modes?.length) {
    const modes = authSpec.supported_modes.filter(isAuthMode);
    if (modes.length > 0) {
      return modes;
    }
  }
  const key = exchange.toLowerCase();
  if (key === "wallex") {
    return ["api_key"];
  }
  if (key === "kucoin") {
    return ["signed_key_passphrase"];
  }
  if (key === "nobitex") {
    return ["signed_key", "api_key"];
  }
  return ["signed_key"];
}

export function credentialsFormFromAccount(
  credentials: AccountCredentials | { auth_mode?: string; api_key?: string; api_secret?: string; api_passphrase?: string; api_secret_set?: boolean; api_passphrase_set?: boolean } | null | undefined,
  exchange: string,
  authSpec?: ExchangeAuthSpec | null,
): CredentialsFormState {
  if (!credentials) {
    return {
      ...EMPTY_CREDENTIALS_FORM,
      auth_mode: defaultAuthModeForExchange(exchange, authSpec),
    };
  }
  return {
    auth_mode: asAuthMode(credentials.auth_mode, defaultAuthModeForExchange(exchange, authSpec)),
    api_key: credentials.api_key ?? "",
    api_secret: credentials.api_secret_set ? MASK : credentials.api_secret ?? "",
    api_passphrase: credentials.api_passphrase_set ? MASK : credentials.api_passphrase ?? "",
    api_secret_set: Boolean(credentials.api_secret_set),
    api_passphrase_set: Boolean(credentials.api_passphrase_set),
  };
}

export function credentialsPayloadFromForm(state: CredentialsFormState): AccountCredentials {
  const auth_mode = state.auth_mode;
  const api_key = state.api_key.trim();
  if (!api_key) {
    throw new Error("API key is required");
  }
  const payload: AccountCredentials = { auth_mode, api_key };
  if (auth_mode === "signed_key" || auth_mode === "signed_key_passphrase") {
    const secret = state.api_secret.trim();
    if (!secret && !state.api_secret_set) {
      throw new Error("API secret is required");
    }
    if (secret && secret !== MASK) {
      payload.api_secret = secret;
    }
  }
  if (auth_mode === "signed_key_passphrase") {
    const passphrase = state.api_passphrase.trim();
    if (!passphrase && !state.api_passphrase_set) {
      throw new Error("API passphrase is required");
    }
    if (passphrase && passphrase !== MASK) {
      payload.api_passphrase = passphrase;
    }
  }
  return payload;
}

export function credentialsSummary(credentials: AccountCredentials | null | undefined): string {
  if (!credentials) {
    return "—";
  }
  const mode = AUTH_MODE_LABELS[credentials.auth_mode] ?? credentials.auth_mode;
  const key = credentials.api_key ? `${credentials.api_key.slice(0, 6)}…` : "—";
  if (credentials.auth_mode === "api_key") {
    return `${mode}: ${key}`;
  }
  const parts = [mode, key];
  if (credentials.api_secret_set) {
    parts.push("secret ✓");
  }
  if (credentials.api_passphrase_set) {
    parts.push("passphrase ✓");
  }
  return parts.join(" · ");
}
