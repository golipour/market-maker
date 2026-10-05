// Probe: does the overrides FORM surface the live bitwana-sample override keys?
import {
  PRESET_SECTIONS,
  applySchemaDefaults,
  fieldVisible,
} from "../src/config/presetSchema";
import { cloneConfig, deleteAt, getAt, isPlainObject, type JsonObject } from "../src/lib/presetConfig";

// Real preset values pulled from registry_strategy_presets (preset_id='sample').
const preset: JsonObject = {
  controller: {
    decision_timeout_ms: 3000,
    negative_objective_action: "floor_action",
    local_price_model: { kind: "conditional_mechanical_mixture" },
    optimizer: {
      kind: "monte_carlo_mpc_tensor_batched",
      horizon: 3,
      n_samples: 16,
      adam_epochs: 8,
    },
  },
};

// Real override pulled from registry_strategies (strategy_id='bitwana-sample').
const override: JsonObject = {
  controller: {
    optimizer: { horizon: 2, n_samples: 8, adam_epochs: 10 },
    decision_timeout_ms: 15000,
    negative_objective_action: "optimizer_action",
  },
};

function mergeForVisibility(base: JsonObject, overlay: JsonObject): JsonObject {
  function walk(a: unknown, b: unknown): unknown {
    if (!isPlainObject(b)) return b !== undefined ? b : a;
    const aObj = isPlainObject(a) ? a : {};
    const out: JsonObject = { ...aObj };
    for (const [k, c] of Object.entries(b)) out[k] = walk(aObj[k], c);
    return out;
  }
  return walk(base, overlay) as JsonObject;
}

// PresetConfigForm: mode="overrides"
const inheritedView = applySchemaDefaults(cloneConfig(preset));
const effective = mergeForVisibility(inheritedView, override);
const displayConfig = override; // value prop in overrides mode is `value`, not defaulted

const KEY_PATHS = [
  "controller.optimizer.horizon",
  "controller.optimizer.n_samples",
  "controller.optimizer.adam_epochs",
  "controller.decision_timeout_ms",
  "controller.negative_objective_action",
];

console.log("--- overrides form, REAL preset + REAL override ---");
for (const p of KEY_PATHS) {
  const section = PRESET_SECTIONS.find((s) => s.fields.some((f) => f.path === p));
  const field = section?.fields.find((f) => f.path === p);
  const visible = field ? fieldVisible(field, effective) : false;
  const inSection = !!section && section.fields.some((f) => f.path === p);
  const raw = getAt(displayConfig, p);
  const inherited = getAt(inheritedView, p);
  console.log(
    `  ${visible ? "VISIBLE " : "HIDDEN  "} ${p.padEnd(42)} raw=${JSON.stringify(raw)} inherited=${JSON.stringify(inherited)} type=${field?.type} section=${section?.id}`,
  );
}

console.log("\n--- section open/closed state (defaultOpen) ---");
PRESET_SECTIONS.forEach((s, i) => {
  const open = i < 3 || s.id.startsWith("optimizer");
  const vis = s.fields.filter((f) => fieldVisible(f, effective)).length;
  if (KEY_PATHS.some((p) => s.fields.some((f) => f.path === p))) {
    console.log(`  section ${s.id}: defaultOpen=${open}, visible ${vis}/${s.fields.length}`);
  }
});

console.log("\n--- can 'Clear section overrides' reach these? ---");
for (const s of PRESET_SECTIONS) {
  if (!KEY_PATHS.some((p) => s.fields.some((f) => f.path === p))) continue;
  let next = override;
  for (const f of s.fields) next = deleteAt(next, f.path);
  console.log(`  after clearing '${s.id}': ${JSON.stringify(next)}`);
}