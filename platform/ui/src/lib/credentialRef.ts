/** Matches mm.registry.credentials.private_credentials_from_ref */

export type CredentialMode =
  | "single_token"
  | "token_and_private_key"
  | "key_and_private_key"
  | "advanced_json";

const ALLOWED_KEYS = new Set(["api_token_env", "api_key_env", "api_private_key_env"]);

export interface CredentialFormState {
  mode: CredentialMode;
  singleEnv: string;
  tokenEnv: string;
  privateKeyEnv: string;
  keyEnv: string;
  advancedJson: string;
}

export const EMPTY_CREDENTIAL_FORM: CredentialFormState = {
  mode: "single_token",
  singleEnv: "",
  tokenEnv: "",
  privateKeyEnv: "",
  keyEnv: "",
  advancedJson: "{}",
};

export function credentialFormFromRef(credentialRef: string): CredentialFormState {
  const ref = credentialRef.trim();
  if (!ref) {
    return { ...EMPTY_CREDENTIAL_FORM };
  }
  if (ref.startsWith("{")) {
    try {
      const payload = JSON.parse(ref) as Record<string, unknown>;
      const token = String(payload.api_token_env ?? "").trim();
      const key = String(payload.api_key_env ?? "").trim();
      const privateKey = String(payload.api_private_key_env ?? "").trim();
      if (token && privateKey && !key) {
        return {
          mode: "token_and_private_key",
          singleEnv: "",
          tokenEnv: token,
          privateKeyEnv: privateKey,
          keyEnv: "",
          advancedJson: ref,
        };
      }
      if (key && privateKey && !token) {
        return {
          mode: "key_and_private_key",
          singleEnv: "",
          tokenEnv: "",
          privateKeyEnv: privateKey,
          keyEnv: key,
          advancedJson: ref,
        };
      }
      return {
        ...EMPTY_CREDENTIAL_FORM,
        mode: "advanced_json",
        advancedJson: JSON.stringify(payload, null, 2),
      };
    } catch {
      return {
        ...EMPTY_CREDENTIAL_FORM,
        mode: "advanced_json",
        advancedJson: ref,
      };
    }
  }
  if (ref.startsWith("env:")) {
    return {
      ...EMPTY_CREDENTIAL_FORM,
      mode: "single_token",
      singleEnv: ref.slice(4).trim(),
    };
  }
  return {
    ...EMPTY_CREDENTIAL_FORM,
    mode: "single_token",
    singleEnv: ref,
  };
}

export function credentialRefFromForm(state: CredentialFormState): string {
  switch (state.mode) {
    case "single_token": {
      const name = state.singleEnv.trim();
      if (!name) {
        throw new Error("Env var name is required");
      }
      return name;
    }
    case "token_and_private_key": {
      const token = state.tokenEnv.trim();
      const privateKey = state.privateKeyEnv.trim();
      if (!token || !privateKey) {
        throw new Error("Both API token env and private key env are required");
      }
      return JSON.stringify({
        api_token_env: token,
        api_private_key_env: privateKey,
      });
    }
    case "key_and_private_key": {
      const key = state.keyEnv.trim();
      const privateKey = state.privateKeyEnv.trim();
      if (!key || !privateKey) {
        throw new Error("Both API key env and private key env are required");
      }
      return JSON.stringify({
        api_key_env: key,
        api_private_key_env: privateKey,
      });
    }
    case "advanced_json": {
      const payload = JSON.parse(state.advancedJson) as Record<string, unknown>;
      if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
        throw new Error("Credential JSON must be an object");
      }
      const out: Record<string, string> = {};
      for (const [key, value] of Object.entries(payload)) {
        if (ALLOWED_KEYS.has(key) && String(value).trim()) {
          out[key] = String(value).trim();
        }
      }
      if (Object.keys(out).length === 0) {
        throw new Error("Credential JSON needs at least one *_env key");
      }
      return JSON.stringify(out);
    }
    default:
      throw new Error("Unknown credential mode");
  }
}

export function credentialRefSummary(credentialRef: string): string {
  const form = credentialFormFromRef(credentialRef);
  switch (form.mode) {
    case "single_token":
      return form.singleEnv || "—";
    case "token_and_private_key":
      return `${form.tokenEnv} + ${form.privateKeyEnv}`;
    case "key_and_private_key":
      return `${form.keyEnv} + ${form.privateKeyEnv}`;
    case "advanced_json":
      return "custom JSON";
    default:
      return credentialRef;
  }
}

export function resolvedCredentialKeys(credentialRef: string): Record<string, string> {
  try {
    const serialized = credentialRefFromForm(credentialFormFromRef(credentialRef));
    if (serialized.startsWith("{")) {
      return JSON.parse(serialized) as Record<string, string>;
    }
    return { api_token_env: serialized };
  } catch {
    return {};
  }
}
