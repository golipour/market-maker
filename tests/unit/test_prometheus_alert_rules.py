from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def test_prometheus_loads_market_maker_alert_rules() -> None:
    config = yaml.safe_load((ROOT / "docker" / "prometheus.yml").read_text())

    assert "/etc/prometheus/rules/*.yml" in config["rule_files"]


def test_market_maker_alert_rules_cover_core_operational_failures() -> None:
    rules = yaml.safe_load(
        (ROOT / "docker" / "prometheus" / "market_maker_alerts.yml").read_text()
    )
    alert_names = {
        rule["alert"]
        for group in rules["groups"]
        for rule in group["rules"]
    }

    assert {
        "MarketMakerMetricsTargetDown",
        "MarketMakerNoRawOrderbookData",
        "MarketMakerFrameSyncNotReady",
        "MarketMakerExecutionSyncFailed",
        "MarketMakerOrderRouterFailure",
        "MarketMakerPendingOrderConfirmationStale",
        "MarketMakerPersistenceErrors",
        "MarketMakerKillSwitchEnabled",
    } <= alert_names
