/**
 * Pure helper tests for preset config round-trips.
 * Run: npx tsx scripts/test-preset-config.mts
 */
import assert from "node:assert/strict";
import {
  ALL_FIELD_PATHS,
  PRESET_SECTIONS,
  applySchemaDefaults,
  fieldVisible,
  FIELD_BY_PATH,
} from "../src/config/presetSchema.ts";
import {
  deleteAt,
  getAt,
  mergeFormIntoConfig,
  serializeOverrides,
  setAt,
  stripHelperKeys,
} from "../src/lib/presetConfig.ts";

function testStripHelperKeys() {
  const input = {
    interval_ms: 1000,
    _comment: "x",
    liquidity_sourcing: { mode: "off", _options: { mode: ["off"] } },
  };
  const out = stripHelperKeys(input) as Record<string, unknown>;
  assert.equal(out.interval_ms, 1000);
  assert.equal("_comment" in out, false);
  assert.deepEqual(out.liquidity_sourcing, { mode: "off" });
}

function testMergePreservesUnknown() {
  const loaded = {
    interval_ms: 5000,
    custom_extension: { enabled: true },
    controller: { dry_run: true, mystery: 1 },
  };
  const form = {
    interval_ms: 10_000,
    controller: { dry_run: false },
  };
  const merged = mergeFormIntoConfig(loaded, form, ALL_FIELD_PATHS);
  assert.equal(merged.interval_ms, 10_000);
  assert.deepEqual(merged.custom_extension, { enabled: true });
  assert.equal(getAt(merged, "controller.dry_run"), false);
  assert.equal(getAt(merged, "controller.mystery"), 1);
}

function testSerializeOverridesOmitsEmpty() {
  let overrides = {};
  overrides = setAt(overrides, "controller.dry_run", true);
  overrides = setAt(overrides, "liquidity_sourcing.mode", "live");
  overrides = setAt(overrides, "controller.optimizer.local.viability", {});
  const sparse = serializeOverrides(overrides);
  assert.equal(getAt(sparse, "controller.dry_run"), true);
  assert.equal(getAt(sparse, "liquidity_sourcing.mode"), "live");
  assert.equal(getAt(sparse, "controller.optimizer.local.viability"), undefined);
}

function testDeleteAtPrunes() {
  let obj = setAt({}, "controller.dry_run", true);
  obj = deleteAt(obj, "controller.dry_run");
  assert.deepEqual(obj, {});
}

function testKindVisibility() {
  const residualField = FIELD_BY_PATH.get("controller.local_price_model.reference_vwap_vol");
  assert.ok(residualField);
  assert.equal(
    fieldVisible(residualField, {
      controller: { local_price_model: { kind: "residual" } },
    }),
    false,
  );
  assert.equal(
    fieldVisible(residualField, {
      controller: { local_price_model: { kind: "conditional_mechanical_mixture" } },
    }),
    true,
  );
}

function testDefaultsFillMissing() {
  const withDefaults = applySchemaDefaults({});
  assert.equal(getAt(withDefaults, "interval_ms"), 10_000);
  assert.equal(getAt(withDefaults, "liquidity_sourcing.mode"), "off");
  assert.equal(getAt(withDefaults, "controller.optimizer.kind"), "monte_carlo_mpc_tensor_batched");
}

function testViabilityDualFieldsAreOnBothVenues() {
  const withDefaults = applySchemaDefaults({});
  for (const prefix of ["local", "global"] as const) {
    const base = `controller.optimizer.${prefix}.viability`;
    for (const key of [
      "lambda_max",
      "dual_ewma_decay",
      "shortfall_tolerance",
      "lambda_learning_rate",
      "recovery_fraction",
    ]) {
      const path = `${base}.${key}`;
      assert.ok(FIELD_BY_PATH.has(path), `missing ${path}`);
    }
    assert.equal(getAt(withDefaults, `${base}.lambda_max`), 100);
    assert.equal(getAt(withDefaults, `${base}.dual_ewma_decay`), 0.9);
    assert.equal(getAt(withDefaults, `${base}.recovery_fraction`), 0.0);
  }
}

/**
 * The overrides form reports how many schema fields are explicitly pinned.
 * This mirrors the counting used by PresetConfigForm so the badge, the
 * "Reset all to preset" banner, and the disabled state of
 * "Clear section overrides" are all covered.
 */
function countOverridden(value: Record<string, unknown>, paths: readonly string[]): number {
  return paths.reduce((total, p) => total + (getAt(value, p) !== undefined ? 1 : 0), 0);
}

function sectionFieldPaths(sectionId: string): string[] {
  const section = PRESET_SECTIONS.find((s) => s.id === sectionId);
  assert.ok(section, `missing section ${sectionId}`);
  return section!.fields.map((f) => f.path);
}

const OVERRIDE_SAMPLE = {
  controller: {
    optimizer: { horizon: 3, n_samples: 16, adam_epochs: 8 },
    decision_timeout_ms: 15000,
  },
};

function testCountOverriddenMatchesExplicitOverrides() {
  const optimizerPaths = sectionFieldPaths("optimizer");
  const controllerPaths = sectionFieldPaths("controller");
  assert.equal(countOverridden(OVERRIDE_SAMPLE, optimizerPaths), 3);
  assert.equal(countOverridden(OVERRIDE_SAMPLE, controllerPaths), 1);
  // A field the operator never touched must not count as overridden.
  assert.equal(countOverridden({}, optimizerPaths), 0);
  assert.equal(countOverridden({ controller: { optimizer: {} } }, optimizerPaths), 0);
}

function testClearSectionOverridesRemovesOnlyThatSection() {
  const optimizerPaths = sectionFieldPaths("optimizer");
  let next = OVERRIDE_SAMPLE as Record<string, unknown>;
  for (const p of optimizerPaths) next = deleteAt(next, p);
  // optimizer block is gone, the rest of the override survives untouched.
  assert.equal(getAt(next, "controller.optimizer.horizon"), undefined);
  assert.equal(getAt(next, "controller.optimizer.adam_epochs"), undefined);
  assert.equal(getAt(next, "controller.decision_timeout_ms"), 15000);
}

function testResetAllPreservesUnknownKeys() {
  // "Reset all to preset" must only strip schema-known paths. An unknown key
  // added by hand in the JSON tab is configuration we cannot see, so dropping
  // it silently would be data loss.
  const withUnknown = {
    controller: { optimizer: { horizon: 3 }, mystery_knob: 7 },
    custom_extension: { enabled: true },
  };
  let next = withUnknown as Record<string, unknown>;
  for (const section of PRESET_SECTIONS) {
    for (const field of section.fields) next = deleteAt(next, field.path);
  }
  assert.equal(getAt(next, "controller.optimizer.horizon"), undefined);
  assert.deepEqual(next.controller, { mystery_knob: 7 });
  assert.deepEqual(next.custom_extension, { enabled: true });
}

const tests = [
  testStripHelperKeys,
  testMergePreservesUnknown,
  testSerializeOverridesOmitsEmpty,
  testDeleteAtPrunes,
  testKindVisibility,
  testDefaultsFillMissing,
  testViabilityDualFieldsAreOnBothVenues,
  testCountOverriddenMatchesExplicitOverrides,
  testClearSectionOverridesRemovesOnlyThatSection,
  testResetAllPreservesUnknownKeys,
];

let failed = 0;
for (const test of tests) {
  try {
    test();
    console.log(`ok  ${test.name}`);
  } catch (err) {
    failed += 1;
    console.error(`FAIL ${test.name}`);
    console.error(err);
  }
}
if (failed > 0) {
  process.exit(1);
}
console.log(`\n${tests.length} passed`);
