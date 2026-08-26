/**
 * Tests for Observability Grafana scope persistence.
 * Run: npx tsx scripts/test-observability-grafana.mts
 */
import assert from "node:assert/strict";
import {
  DEFAULT_GRAFANA_SCOPE,
  OBSERVABILITY_GRAFANA_STORAGE_KEY,
  applyGrafanaScope,
  loadGrafanaScope,
  parseGrafanaScope,
  saveGrafanaScope,
  usesFrameScope,
} from "../src/lib/observabilityGrafana.ts";

function memoryStorage(initial: Record<string, string> = {}): {
  getItem: (key: string) => string | null;
  setItem: (key: string, value: string) => void;
  data: Record<string, string>;
} {
  const data = { ...initial };
  return {
    data,
    getItem(key: string) {
      return Object.prototype.hasOwnProperty.call(data, key) ? data[key] : null;
    },
    setItem(key: string, value: string) {
      data[key] = value;
    },
  };
}

function testParseIgnoresJunkAndFillsDefaults() {
  const parsed = parseGrafanaScope({
    activeUid: "mm-obs-v2-orderbook-depth",
    exchange: "bitwana",
    extra: 1,
    interval_ms: "10000",
  });
  assert.equal(parsed.activeUid, "mm-obs-v2-orderbook-depth");
  assert.equal(parsed.exchange, "bitwana");
  assert.equal(parsed.interval_ms, "10000");
  assert.equal(parsed.symbol, DEFAULT_GRAFANA_SCOPE.symbol);
  assert.equal(parsed.from, DEFAULT_GRAFANA_SCOPE.from);
}

function testLoadAndSaveRoundTrip() {
  const storage = memoryStorage();
  const saved = {
    ...DEFAULT_GRAFANA_SCOPE,
    activeUid: "mm-obs-v2-market-data",
    exchange: "bitwana",
    symbol: "USDT_IRT",
    interval_ms: "10000",
    spread_max: "5000",
    target_frame_mode: "pinned",
  };
  saveGrafanaScope(saved, storage);
  assert.ok(storage.data[OBSERVABILITY_GRAFANA_STORAGE_KEY]);
  const loaded = loadGrafanaScope(storage);
  assert.deepEqual(loaded, saved);
}

function testLoadCorruptStorageReturnsDefaults() {
  const storage = memoryStorage({ [OBSERVABILITY_GRAFANA_STORAGE_KEY]: "{not-json" });
  assert.deepEqual(loadGrafanaScope(storage), DEFAULT_GRAFANA_SCOPE);
  assert.deepEqual(loadGrafanaScope(null), DEFAULT_GRAFANA_SCOPE);
}

function testApplyGrafanaScopeWritesTimeAndVars() {
  const scoped = applyGrafanaScope(
    "http://grafana.test:3000/d/mm-obs-v2-orderbook-depth?orgId=1&theme=dark&kiosk=1",
    {
      ...DEFAULT_GRAFANA_SCOPE,
      from: "now-6h",
      exchange: "bitwana",
      symbol: "USDT_IRT",
      interval_ms: "10000",
      actor_id: "bitwana-controller",
      spread_max: "5000",
      target_frame_mode: "pinned",
    },
  );
  const url = new URL(scoped);
  assert.equal(url.searchParams.get("from"), "now-6h");
  assert.equal(url.searchParams.get("to"), "now");
  assert.equal(url.searchParams.get("var-exchange"), "bitwana");
  assert.equal(url.searchParams.get("var-symbol"), "USDT_IRT");
  assert.equal(url.searchParams.get("var-interval_ms"), "10000");
  assert.equal(url.searchParams.get("var-actor_id"), "bitwana-controller");
  assert.equal(url.searchParams.get("var-spread_max"), "5000");
  assert.equal(url.searchParams.get("var-target_frame_mode"), "pinned");
  assert.equal(url.searchParams.getAll("var-exchange").length, 1);
}

function testApplyGrafanaScopeReplacesForcedDefaults() {
  const scoped = applyGrafanaScope(
    "http://grafana.test:3000/d/uid?from=now-1h&to=now&var-exchange=$__all&var-symbol=$__all",
    { ...DEFAULT_GRAFANA_SCOPE, exchange: "nobitex", from: "now-15m" },
  );
  const url = new URL(scoped);
  assert.equal(url.searchParams.get("from"), "now-15m");
  assert.equal(url.searchParams.get("var-exchange"), "nobitex");
  assert.equal(url.searchParams.getAll("var-exchange").length, 1);
}

function testFrameScopeDashboards() {
  assert.equal(usesFrameScope("mm-obs-v2-orderbook-depth"), true);
  assert.equal(usesFrameScope("mm-obs-v2-target-order-stems"), true);
  assert.equal(usesFrameScope("mm-obs-v2-market-data"), false);
}

const tests = [
  testParseIgnoresJunkAndFillsDefaults,
  testLoadAndSaveRoundTrip,
  testLoadCorruptStorageReturnsDefaults,
  testApplyGrafanaScopeWritesTimeAndVars,
  testApplyGrafanaScopeReplacesForcedDefaults,
  testFrameScopeDashboards,
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
