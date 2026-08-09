/** Pure helpers for preset / override config JSON round-trips. */

export type JsonObject = Record<string, unknown>;

const HELPER_KEY_RE = /^_/;

export function isPlainObject(value: unknown): value is JsonObject {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** Deep-clone and drop `_comment` / `_options` / other `_…` helper keys. */
export function stripHelperKeys(value: unknown): unknown {
  if (Array.isArray(value)) {
    return value.map(stripHelperKeys);
  }
  if (!isPlainObject(value)) {
    return value;
  }
  const out: JsonObject = {};
  for (const [key, child] of Object.entries(value)) {
    if (HELPER_KEY_RE.test(key)) {
      continue;
    }
    out[key] = stripHelperKeys(child);
  }
  return out;
}

export function getAt(root: unknown, path: string): unknown {
  if (!path) {
    return root;
  }
  const parts = path.split(".");
  let cur: unknown = root;
  for (const part of parts) {
    if (!isPlainObject(cur)) {
      return undefined;
    }
    cur = cur[part];
  }
  return cur;
}

export function setAt(root: JsonObject, path: string, value: unknown): JsonObject {
  const parts = path.split(".");
  if (parts.length === 0) {
    return root;
  }
  const clone: JsonObject = { ...root };
  let cursor: JsonObject = clone;
  for (let i = 0; i < parts.length - 1; i += 1) {
    const key = parts[i]!;
    const next = cursor[key];
    const child: JsonObject = isPlainObject(next) ? { ...next } : {};
    cursor[key] = child;
    cursor = child;
  }
  cursor[parts[parts.length - 1]!] = value;
  return clone;
}

/** Remove a path; prune empty parent objects. */
export function deleteAt(root: JsonObject, path: string): JsonObject {
  const parts = path.split(".");
  if (parts.length === 0) {
    return root;
  }

  function walk(node: JsonObject, index: number): JsonObject | undefined {
    const key = parts[index]!;
    if (index === parts.length - 1) {
      if (!(key in node)) {
        return node;
      }
      const next = { ...node };
      delete next[key];
      return Object.keys(next).length === 0 ? undefined : next;
    }
    const child = node[key];
    if (!isPlainObject(child)) {
      return node;
    }
    const updated = walk(child, index + 1);
    const next = { ...node };
    if (updated === undefined) {
      delete next[key];
    } else {
      next[key] = updated;
    }
    return Object.keys(next).length === 0 ? undefined : next;
  }

  return walk({ ...root }, 0) ?? {};
}

export function deepMerge(base: unknown, overlay: unknown): unknown {
  if (!isPlainObject(overlay)) {
    return overlay;
  }
  const baseObj = isPlainObject(base) ? base : {};
  const out: JsonObject = { ...baseObj };
  for (const [key, value] of Object.entries(overlay)) {
    if (value === undefined) {
      continue;
    }
    if (isPlainObject(value) && isPlainObject(baseObj[key])) {
      out[key] = deepMerge(baseObj[key], value);
    } else {
      out[key] = isPlainObject(value) ? deepMerge({}, value) : value;
    }
  }
  return out;
}

/**
 * Write known form paths into a previously loaded config blob so unknown keys
 * in the DB preset are preserved.
 */
export function mergeFormIntoConfig(
  loaded: JsonObject,
  formValue: JsonObject,
  knownPaths: readonly string[],
): JsonObject {
  const base = stripHelperKeys(loaded) as JsonObject;
  let out: JsonObject = isPlainObject(base) ? structuredClone(base) : {};
  for (const path of knownPaths) {
    const value = getAt(formValue, path);
    if (value === undefined) {
      continue;
    }
    out = setAt(out, path, value);
  }
  return out;
}

/** Deep-merge overlay onto base (for applying overrides onto a preset). */
export function mergeConfigs(base: JsonObject, overlay: JsonObject): JsonObject {
  return deepMerge(stripHelperKeys(base), stripHelperKeys(overlay)) as JsonObject;
}

/**
 * Serialize overrides: drop undefined leaf paths and empty objects.
 * `formValue` is already the sparse override object.
 */
export function serializeOverrides(formValue: JsonObject): JsonObject {
  function clean(value: unknown): unknown {
    if (Array.isArray(value)) {
      return value;
    }
    if (!isPlainObject(value)) {
      return value;
    }
    const out: JsonObject = {};
    for (const [key, child] of Object.entries(value)) {
      if (child === undefined || child === null || child === "") {
        continue;
      }
      const cleaned = clean(child);
      if (cleaned === undefined) {
        continue;
      }
      if (isPlainObject(cleaned) && Object.keys(cleaned).length === 0) {
        continue;
      }
      out[key] = cleaned;
    }
    return out;
  }
  return clean(stripHelperKeys(formValue)) as JsonObject;
}

export function cloneConfig(value: JsonObject): JsonObject {
  return structuredClone(stripHelperKeys(value) ?? {}) as JsonObject;
}

export function formatInherited(value: unknown): string {
  if (value === undefined) {
    return "";
  }
  if (typeof value === "boolean") {
    return value ? "true" : "false";
  }
  if (value === null) {
    return "null";
  }
  if (typeof value === "object") {
    return JSON.stringify(value);
  }
  return String(value);
}
