from __future__ import annotations

from mm.registry.optimizer_config_migration import migrate_runtime_config_mapping


def test_migrate_inventory_zone_to_viability() -> None:
    payload = {
        "controller": {
            "optimizer": {
                "local": {
                    "w_inv": 0.01,
                    "inventory_zone": {
                        "comfort_low": -0.2,
                        "comfort_high": 0.2,
                    },
                    "cumulative_qty_floor_at_spread_max": 2.0,
                }
            }
        }
    }
    migrated, changed = migrate_runtime_config_mapping(payload)
    assert changed is True
    local = migrated["controller"]["optimizer"]["local"]
    assert "w_inv" not in local
    assert "inventory_zone" not in local
    assert local["viability"]["min_quote_qty"] == 2.0
    assert local["viability"]["lambda_init"] == 1.0
