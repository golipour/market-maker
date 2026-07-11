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
    assert local["viability"]["min_viable_quote"] == 50_000.0
    assert local["viability"]["lambda_init"] == 0.0


def test_migrate_sparse_optimizer_fills_canonical_defaults() -> None:
    payload = {
        "controller": {
            "optimizer": {
                "kind": "monte_carlo_mpc_tensor_batched",
                "horizon": 2,
            }
        }
    }
    migrated, changed = migrate_runtime_config_mapping(payload)
    assert changed is True
    optimizer = migrated["controller"]["optimizer"]
    assert optimizer["local"]["viability"]["lambda_init"] == 0.0
    assert optimizer["local"]["validity"]["min_score"] == 0.3
    assert optimizer["global"]["viability"]["availability_target"] == 0.99
    assert optimizer["adam_epochs"] == 40


def test_migrate_w_book_to_validity() -> None:
    payload = {
        "controller": {
            "optimizer": {
                "local": {
                    "w_book": 0.002,
                }
            }
        }
    }
    migrated, changed = migrate_runtime_config_mapping(payload)
    assert changed is True
    local = migrated["controller"]["optimizer"]["local"]
    assert "w_book" not in local
    assert local["validity"]["min_score"] == 0.3
    assert local["validity"]["penalty_scale"] == 100.0


def test_migrate_sparse_optimizer_preserves_existing_overrides() -> None:
    payload = {
        "controller": {
            "optimizer": {
                "kind": "monte_carlo_mpc_tensor_batched",
                "horizon": 4,
                "local": {
                    "n_components": 2,
                    "viability": {
                        "min_viable_quote": 75_000.0,
                        "margin_buffer": 5_000.0,
                        "availability_target": 0.99,
                        "shortfall_tolerance": 0.05,
                        "lambda_init": 1.0,
                        "lambda_learning_rate": 0.5,
                    },
                },
            }
        }
    }
    migrated, changed = migrate_runtime_config_mapping(payload)
    assert changed is True
    optimizer = migrated["controller"]["optimizer"]
    assert optimizer["horizon"] == 4
    assert optimizer["local"]["n_components"] == 2
    assert optimizer["local"]["viability"]["min_viable_quote"] == 75_000.0


def test_migrate_legacy_viability_keys_to_viable_quote_formulation() -> None:
    payload = {
        "controller": {
            "optimizer": {
                "local": {
                    "cumulative_qty_floor_at_spread_max": 1.0,
                    "viability": {
                        "min_free_quote": 1.0,
                        "margin_sharpness": 0.1,
                        "availability_target": 0.9,
                        "shortfall_tolerance": 0.05,
                        "lambda_init": 1.0,
                        "lambda_learning_rate": 1.0,
                    },
                }
            }
        }
    }
    migrated, changed = migrate_runtime_config_mapping(payload)
    assert changed is True
    viability = migrated["controller"]["optimizer"]["local"]["viability"]
    assert "min_free_quote" not in viability
    assert "margin_sharpness" not in viability
    assert viability["min_viable_quote"] == 50_000.0
    assert viability["margin_buffer"] == 5_000.0
    assert viability["availability_target"] == 0.99


def test_migrate_oversized_viability_min_viable_quote_resyncs_to_canonical() -> None:
    payload = {
        "controller": {
            "optimizer": {
                "local": {
                    "cumulative_qty_floor_at_spread_max": 1.0,
                    "viability": {
                        "min_viable_quote": 5_000_000.0,
                        "margin_buffer": 5_000.0,
                        "availability_target": 0.99,
                        "shortfall_tolerance": 0.05,
                        "lambda_init": 1.0,
                        "lambda_learning_rate": 1.0,
                    },
                }
            }
        }
    }
    migrated, changed = migrate_runtime_config_mapping(payload)
    assert changed is True
    assert (
        migrated["controller"]["optimizer"]["local"]["viability"]["min_viable_quote"]
        == 50_000.0
    )


def test_migrate_warm_start_bool_to_object() -> None:
    payload = {
        "controller": {
            "optimizer": {
                "kind": "monte_carlo_mpc_tensor_batched",
                "warm_start": True,
                "local": {},
            }
        }
    }
    migrated, changed = migrate_runtime_config_mapping(payload)
    assert changed is True
    ws = migrated["controller"]["optimizer"]["warm_start"]
    assert ws["enabled"] is True
    assert ws["alpha"] == 1.0
    assert ws["preserve_optimizer_state"] is True


def test_migrate_missing_warm_start_merges_canonical_object() -> None:
    payload = {
        "controller": {
            "optimizer": {
                "kind": "monte_carlo_mpc_tensor_batched",
                "local": {},
            }
        }
    }
    migrated, changed = migrate_runtime_config_mapping(payload)
    assert changed is True
    ws = migrated["controller"]["optimizer"]["warm_start"]
    assert isinstance(ws, dict)
    assert "enabled" in ws
    assert "alpha" in ws
