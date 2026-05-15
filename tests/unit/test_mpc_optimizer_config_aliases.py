from __future__ import annotations

from mm.strategies.mpc_engine import normalize_mpc_optimizer_config


def test_normalize_target_inventory_level_and_max_total_qty() -> None:
    cfg = {
        "target_inventory_level": 123.5,
        "max_total_qty": 0.42,
        "spread_target_lots": 9.0,
    }
    normalize_mpc_optimizer_config(cfg)
    assert cfg == {
        "target_inv": 123.5,
        "total_q_target": 0.42,
        "spread_depth": 9.0,
    }


def test_normalize_preserves_legacy_keys_when_new_absent() -> None:
    cfg = {"target_inv": 1.0, "total_q_target": 2.0}
    normalize_mpc_optimizer_config(cfg)
    assert cfg == {"target_inv": 1.0, "total_q_target": 2.0}


def test_normalize_project_max_total_qty() -> None:
    cfg = {"project_max_total_qty": True}
    normalize_mpc_optimizer_config(cfg)
    assert cfg == {"project_total_q_target": True}


def test_project_max_total_qty_overrides_legacy() -> None:
    cfg = {"project_total_q_target": False, "project_max_total_qty": True}
    normalize_mpc_optimizer_config(cfg)
    assert cfg == {"project_total_q_target": True}


def test_new_names_override_legacy_in_dict_order() -> None:
    cfg = {"target_inv": 1.0, "target_inventory_level": 99.0}
    normalize_mpc_optimizer_config(cfg)
    assert cfg == {"target_inv": 99.0}
