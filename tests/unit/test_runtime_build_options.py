from __future__ import annotations

import pytest

from mm.runtime.build_options import RuntimeBuildOptions
from mm.runtime.config import (
    BusRuntimeConfig,
    ControllerRuntimeConfig,
    MarketDataSourceConfig,
    MarketMakerRuntimeConfig,
    PersistenceRuntimeConfig,
)
from mm.runtime.config import BusRuntimeConfig, ControllerRuntimeConfig, MarketDataSourceConfig, MarketMakerRuntimeConfig, PersistenceRuntimeConfig
from mm.runtime.engine import build_runtime_components, resolve_build_options


def _minimal_config(**kwargs) -> MarketMakerRuntimeConfig:
    local = kwargs.pop("local", MarketDataSourceConfig(role="local", exchange="nobitex", symbol="BTC_IRT"))
    controller = kwargs.pop("controller", ControllerRuntimeConfig(enabled=True, actor_id="test-actor"))
    return MarketMakerRuntimeConfig(local=local, controller=controller, **kwargs)


def test_runtime_build_options_for_mode_presets() -> None:
    monolith = RuntimeBuildOptions.for_mode("monolith")
    assert monolith.enable_public_ingestion is True
    assert monolith.enable_strategy is True
    assert monolith.enable_persistence is True

    consumer = RuntimeBuildOptions.for_mode("consumer_only")
    assert consumer.enable_public_ingestion is False
    assert consumer.enable_strategy is True

    ingestor = RuntimeBuildOptions.for_mode("ingestor_only")
    assert ingestor.enable_public_ingestion is True
    assert ingestor.enable_strategy is False
    assert ingestor.enable_private_state is False
    assert ingestor.enable_persistence is False

    persistence = RuntimeBuildOptions.for_mode("persistence_only")
    assert persistence.enable_public_ingestion is False
    assert persistence.enable_strategy is False
    assert persistence.enable_persistence is True


@pytest.mark.asyncio
async def test_resolve_build_options_mixed_sources_stays_monolith() -> None:
    config = MarketMakerRuntimeConfig(
        local=MarketDataSourceConfig(
            role="local",
            exchange="nobitex",
            symbol="BTC_IRT",
            market_data_source="bus",
        ),
        global_=MarketDataSourceConfig(
            role="global",
            exchange="wallex",
            symbol="BTC_IRT",
            market_data_source="local",
        ),
        bus=BusRuntimeConfig(kind="redis_streams"),
    )
    options = resolve_build_options(config)
    assert options.mode == "monolith"
    assert options.enable_public_ingestion is True


@pytest.mark.asyncio
async def test_build_runtime_components_monolith_still_wires_strategy(monkeypatch) -> None:
    monkeypatch.setattr("mm.runtime.engine.load_mm_env", lambda: False)
    config = _minimal_config()

    runtime = await build_runtime_components(config)

    assert runtime.build_options.mode == "monolith"
    assert runtime.frame_assembler is not None
    assert runtime.strategy_runner is not None
    assert runtime.engines


@pytest.mark.asyncio
async def test_build_runtime_components_can_disable_public_ingestion(monkeypatch) -> None:
    monkeypatch.setattr("mm.runtime.engine.load_mm_env", lambda: False)
    options = RuntimeBuildOptions(enable_public_ingestion=False)
    config = _minimal_config()

    runtime = await build_runtime_components(config, build_options=options)

    assert runtime.engines == {}
    assert runtime.public_market_selections == {}
    assert runtime.frame_assembler is not None


@pytest.mark.asyncio
async def test_build_runtime_components_market_data_source_bus_skips_local_engine(monkeypatch) -> None:
    monkeypatch.setattr("mm.runtime.engine.load_mm_env", lambda: False)
    config = _minimal_config(
        local=MarketDataSourceConfig(
            role="local",
            exchange="nobitex",
            symbol="BTC_IRT",
            market_data_source="bus",
        ),
        bus=BusRuntimeConfig(kind="redis_streams"),
    )

    runtime = await build_runtime_components(config)

    assert runtime.engines == {}


@pytest.mark.asyncio
async def test_build_runtime_components_rejects_bus_source_on_memory_bus() -> None:
    config = _minimal_config(
        local=MarketDataSourceConfig(
            role="local",
            exchange="nobitex",
            symbol="BTC_IRT",
            market_data_source="bus",
        ),
    )

    with pytest.raises(ValueError, match="redis_streams"):
        await build_runtime_components(config)


@pytest.mark.asyncio
async def test_build_runtime_components_can_disable_persistence(monkeypatch) -> None:
    monkeypatch.setenv("MM_DATABASE_URL", "postgresql://example")
    monkeypatch.setattr("mm.runtime.engine.load_mm_env", lambda: False)
    config = _minimal_config(persistence=PersistenceRuntimeConfig(enabled=True))
    options = RuntimeBuildOptions(enable_persistence=False)

    runtime = await build_runtime_components(config, build_options=options)

    assert runtime.persistence_daemons == ()


@pytest.mark.asyncio
async def test_build_runtime_components_can_disable_observability(monkeypatch) -> None:
    monkeypatch.setattr("mm.runtime.engine.load_mm_env", lambda: False)
    options = RuntimeBuildOptions(enable_observability=False)
    config = _minimal_config()

    runtime = await build_runtime_components(config, build_options=options)

    assert runtime.observability_daemons == ()


@pytest.mark.asyncio
async def test_build_runtime_components_ingestor_only_mode(monkeypatch) -> None:
    monkeypatch.setattr("mm.runtime.engine.load_mm_env", lambda: False)
    config = _minimal_config()
    options = RuntimeBuildOptions.for_mode("ingestor_only")

    runtime = await build_runtime_components(config, build_options=options)

    assert runtime.engines
    assert runtime.frame_assembler is None
    assert runtime.strategy_runner is None
    assert runtime.private_reconcilers == {}
    assert runtime.persistence_daemons == ()
