/**
 * Mirrors `mm_adapters.diagnostics.delta_codec.apply_patch` for WebSocket `frame_delta` messages.
 */
(function (global) {
  const REPLACE = "$replace";
  const APPEND = "$append";

  function clone(x) {
    if (x === undefined) return undefined;
    if (typeof structuredClone === "function") {
      try {
        return structuredClone(x);
      } catch {
        /* fall through */
      }
    }
    return JSON.parse(JSON.stringify(x));
  }

  function applyDiagnosticsPatch(base, patch) {
    if (patch === undefined || patch === null) {
      return clone(base);
    }
    if (typeof patch === "object" && patch !== null && !Array.isArray(patch)) {
      if (Object.prototype.hasOwnProperty.call(patch, REPLACE)) {
        return clone(patch[REPLACE]);
      }
      if (Object.prototype.hasOwnProperty.call(patch, APPEND)) {
        const b = Array.isArray(base) ? base : [];
        return clone(b).concat(clone(patch[APPEND]));
      }
      if (base !== null && typeof base === "object" && !Array.isArray(base)) {
        const out = clone(base);
        for (const [k, v] of Object.entries(patch)) {
          if (!(k in out)) {
            out[k] = clone(v);
          } else {
            out[k] = applyDiagnosticsPatch(out[k], v);
          }
        }
        return out;
      }
    }
    return clone(patch);
  }

  global.applyDiagnosticsPatch = applyDiagnosticsPatch;
})(typeof window !== "undefined" ? window : globalThis);
