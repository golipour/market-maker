/**
 * Pure helper tests for preset config round-trips.
 * Run: npx tsx scripts/test-preset-config.mts
 */
import assert from "node:assert/strict";
import {
  ALL_FIELD_PATHS,
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

const tests = [
  testStripHelperKeys,
  testMergePreservesUnknown,
  testSerializeOverridesOmitsEmpty,
  testDeleteAtPrunes,
  testKindVisibility,
  testDefaultsFillMissing,
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
