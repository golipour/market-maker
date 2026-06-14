from __future__ import annotations

from pathlib import Path

import pytest

from mm.tools.run_local_stack import (
    DEFAULT_CONFIGS,
    StackService,
    build_command,
    default_services,
    parse_args,
)


def test_default_configs_exist() -> None:
    for path in DEFAULT_CONFIGS.values():
        assert path.is_file(), path


def test_default_services_order_ingestor_first() -> None:
    services = default_services(DEFAULT_CONFIGS)
    assert services[0].name == "ingestor"
    assert services[0].start_first is True
    assert {svc.name for svc in services} == {"ingestor", "persistence", "worker", "consumer"}


def test_build_command_uses_current_interpreter() -> None:
    import sys

    service = StackService("ingestor", "mm.ingestor", Path("cfg.json"))
    cmd = build_command(service)
    assert cmd[0] == sys.executable
    assert cmd[1:5] == ["-m", "mm.ingestor", "--config", "cfg.json"]


def test_parse_args_with_admin_and_skip() -> None:
    args = parse_args(["--with-admin", "--skip", "worker", "persistence"])
    assert args.with_admin is True
    assert set(args.skip) == {"worker", "persistence"}
