from __future__ import annotations

import asyncio
import contextlib
import io
import math
import time
from dataclasses import dataclass, field
from typing import Any

from mm.bus.base import AbstractEventBus
from mm.core.interfaces.actor import AbstractOrderManager
from mm.core.models.execution import Action, Order
from mm.core.models.instrument import SymbolBinding
from mm.core.models.market import MarketEvent, RawOrderbook
from mm.core.models.private import PrivateStateSnapshot
from mm.core.models.strategy import StrategyInputFrame
from mm.runtime.config import ControllerRuntimeConfig


@dataclass(slots=True)
class MpcControllerRunner:
    """Bridge StrategyInputFrame into the legacy MPC functionality.

    The bridge keeps the new runtime float-first: mid/prices are quote-per-base,
    quantities are base units, and order commands stay float until the exchange
    command adapter boundary.
    """

    config: ControllerRuntimeConfig
    binding: SymbolBinding
    frame_queue: asyncio.Queue[StrategyInputFrame]
    order_managers: dict[str, AbstractOrderManager]
    bus: AbstractEventBus | None = None
    _controller: Any = field(init=False, repr=False)
    _market_flow_model: Any = field(init=False, repr=False)
    _price_model: Any = field(init=False, repr=False)
    _previous_frame: StrategyInputFrame | None = field(default=None, init=False, repr=False)
    _task: asyncio.Task | None = field(default=None, init=False, repr=False)
    _stopped: asyncio.Event = field(default_factory=asyncio.Event, init=False, repr=False)

    def __post_init__(self) -> None:
        self._controller = _build_controller(self.config)
        self._market_flow_model = getattr(self._controller, "market_flow_model", None)
        self._price_model = getattr(self._controller, "price_model", None)

    async def start(self) -> None:
        self._stopped.clear()
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._stopped.set()
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None

    async def _run(self) -> None:
        try:
            while not self._stopped.is_set():
                frame = await self.frame_queue.get()
                await self.on_frame(frame)
        except asyncio.CancelledError:
            raise

    async def on_frame(self, frame: StrategyInputFrame) -> Action:
        t0 = time.monotonic_ns()
        try:
            state = _state_from_frame(frame, self.binding, self.config.max_spread)
            _update_market_flow_model(self._market_flow_model, frame)
            _update_float_price_model(self._price_model, self._previous_frame, frame)
            optimizer_stdout = io.StringIO()
            with contextlib.redirect_stdout(optimizer_stdout):
                sigmoid_action = self._controller.optimize(state)
            orders = _orders_from_sigmoid_action(
                actor_id=self.config.actor_id,
                event=frame.primary,
                sigmoid_action=sigmoid_action,
                min_spread=self.config.min_spread,
                max_spread=self.config.max_spread,
                qty_step=self.config.qty_step,
                min_order_qty=self.config.min_order_qty,
                max_orders_per_side=self.config.max_orders_per_side,
            )
            action = Action(actor_id=self.config.actor_id, orders_to_create=tuple(orders))
            await self._publish_decision(
                frame,
                state,
                action,
                latency_ms=(time.monotonic_ns() - t0) / 1_000_000,
                optimizer_log=optimizer_stdout.getvalue().strip(),
            )
            if not self.config.observe_only:
                manager = self.order_managers.get(frame.primary.exchange.lower().strip())
                if manager is not None:
                    await manager.sync([action])
            self._previous_frame = frame
            return action
        except Exception as exc:
            await self._publish(
                self.config.normalized_diagnostics_topic,
                {
                    "type": "ControllerError",
                    "actor_id": self.config.actor_id,
                    "exchange": frame.primary.exchange,
                    "symbol": frame.primary.symbol,
                    "t_end_ms": frame.primary.t_end_ms,
                    "error": type(exc).__name__,
                    "message": str(exc),
                },
            )
            return Action(actor_id=self.config.actor_id)

    async def _publish_decision(
        self,
        frame: StrategyInputFrame,
        state: Any,
        action: Action,
        *,
        latency_ms: float,
        optimizer_log: str,
    ) -> None:
        price_params = self._price_model.get_params() if hasattr(self._price_model, "get_params") else {}
        flow_params = self._market_flow_model.get_params() if hasattr(self._market_flow_model, "get_params") else {}
        payload = {
            "type": "ControllerDecision",
            "actor_id": self.config.actor_id,
            "exchange": frame.primary.exchange,
            "symbol": frame.primary.symbol,
            "t_end_ms": frame.primary.t_end_ms,
            "observe_only": self.config.observe_only,
            "hedge_status": frame.hedge_status,
            "has_hedge": frame.has_hedge,
            "mid": state.mid,
            "inventory": state.inventory,
            "cash": state.cash,
            "buy_volume": state.buy_vol,
            "sell_volume": state.sell_vol,
            "order_count": len(action.orders_to_create),
            "orders": [_order_summary(order) for order in action.orders_to_create],
            "price_model": {
                "kind": (self.config.price_model or {}).get("kind", "residual"),
                "params": price_params,
            },
            "market_flow_model": {
                "kind": (self.config.market_flow_model or {}).get("kind", "hurdle_lognormal"),
                "params": flow_params,
            },
            "optimizer": {
                "kind": (self.config.optimizer or {}).get("kind", "monte_carlo_mpc_tensor_batched"),
                "last_expected_delta_wealth": getattr(self._controller, "_last_E_dW", None),
                "last_components": getattr(self._controller, "_last_J_components", None),
                "log": optimizer_log,
            },
            "latency_ms": latency_ms,
            "unit_policy": "prices=quote_per_base, quantities=base, cash/notional=quote",
        }
        await self._publish(self.config.normalized_diagnostics_topic, payload)
        await self._publish(self.config.normalized_action_topic, action)

    async def _publish(self, topic: str, event: Any) -> None:
        if self.bus is not None:
            await self.bus.publish(topic, event)


def _build_controller(config: ControllerRuntimeConfig) -> Any:
    from mm.core.models.real_execution import RealExecutionModel
    from mm.mpc.market_order_flow_torch import (
        TorchDiscreteGammaModel,
        TorchHurdleGammaModel,
        TorchHurdleLogNormalModel,
        TorchHurdleWeibullModel,
        TorchWeibullModel,
    )
    from mm.mpc.mpc_tensor_batched import MonteCarloMPCTensorBatched
    from mm.mpc.price_torch import TorchResidualPriceModel

    flow_cfg = dict(config.market_flow_model or {})
    flow_kind = str(flow_cfg.pop("kind", "hurdle_lognormal")).lower()
    if flow_kind == "gamma":
        market_flow_model = TorchDiscreteGammaModel(**flow_cfg)
    elif flow_kind == "weibull":
        market_flow_model = TorchWeibullModel(**flow_cfg)
    elif flow_kind == "hurdle_gamma":
        market_flow_model = TorchHurdleGammaModel(**flow_cfg)
    elif flow_kind == "hurdle_weibull":
        market_flow_model = TorchHurdleWeibullModel(**flow_cfg)
    elif flow_kind == "hurdle_lognormal":
        market_flow_model = TorchHurdleLogNormalModel(**flow_cfg)
    else:
        raise ValueError(f"unsupported market_flow_model kind: {flow_kind}")

    price_cfg = dict(config.price_model or {})
    price_kind = str(price_cfg.pop("kind", "residual")).lower()
    if price_kind != "residual":
        raise ValueError(f"unsupported price_model kind: {price_kind}")
    price_model = TorchResidualPriceModel(market_flow_model=market_flow_model, **price_cfg)

    execution_cfg = dict(config.execution_model or {})
    execution_kind = str(execution_cfg.pop("kind", "real")).lower()
    if execution_kind == "observed_step":
        execution_kind = "real"
    if execution_kind != "real":
        raise ValueError(f"unsupported execution_model kind: {execution_kind}")
    execution_model = RealExecutionModel(**execution_cfg)

    optimizer_cfg = dict(config.optimizer or {})
    optimizer_kind = str(optimizer_cfg.pop("kind", "monte_carlo_mpc_tensor_batched")).lower()
    if optimizer_kind != "monte_carlo_mpc_tensor_batched":
        raise ValueError(f"unsupported optimizer kind: {optimizer_kind}")
    optimizer_defaults = {
        "n_samples": 8,
        "horizon": 2,
        "adam_epochs": 10,
        "spread_max": config.max_spread,
        "n_grid": max(2, int(round(config.max_spread))),
        "ob_levels": max(2, int(round(config.max_spread))),
        "total_q_min": config.min_order_qty,
        "total_q_target": None,
        "project_total_q_target": False,
        "device": "cpu",
    }
    optimizer_defaults.update(optimizer_cfg)
    return MonteCarloMPCTensorBatched(
        price_model=price_model,
        market_flow_model=market_flow_model,
        execution_model=execution_model,
        **optimizer_defaults,
    )


def _state_from_frame(frame: StrategyInputFrame, binding: SymbolBinding, spread_max: float):
    from mm.mpc.mpc_state import OrderbookRelative, State

    primary_mid = _mid(frame.primary.orderbook)
    bids = tuple(
        (spread, level.qty)
        for level in frame.primary.orderbook.bids
        if (spread := primary_mid - level.price) >= 0 and spread <= spread_max and level.qty > 0
    )
    asks = tuple(
        (spread, level.qty)
        for level in frame.primary.orderbook.asks
        if (spread := level.price - primary_mid) >= 0 and spread <= spread_max and level.qty > 0
    )
    primary_inventory, primary_cash = _inventory_cash(frame.primary_private, binding.primary.base_asset, binding.primary.quote_asset)
    hedge_buy_ref = hedge_sell_ref = None
    hedge_inventory = 0.0
    hedge_cash = 0.0
    if frame.hedge is not None:
        hedge_buy_ref = frame.hedge.orderbook.asks[0].price if frame.hedge.orderbook.asks else _mid(frame.hedge.orderbook)
        hedge_sell_ref = frame.hedge.orderbook.bids[0].price if frame.hedge.orderbook.bids else _mid(frame.hedge.orderbook)
    if frame.hedge_private is not None and binding.hedge is not None:
        hedge_inventory, hedge_cash = _inventory_cash(
            frame.hedge_private,
            binding.hedge.base_asset,
            binding.hedge.quote_asset,
        )
    return State(
        mid=primary_mid,
        ob_rel=OrderbookRelative(bids=bids, asks=asks),
        buy_vol=frame.primary.buy_volume,
        sell_vol=frame.primary.sell_volume,
        inventory=int(round(primary_inventory)),
        cash=primary_cash,
        buy_ref=frame.primary.orderbook.asks[0].price if frame.primary.orderbook.asks else primary_mid,
        sell_ref=frame.primary.orderbook.bids[0].price if frame.primary.orderbook.bids else primary_mid,
        hedge_buy_ref=hedge_buy_ref,
        hedge_sell_ref=hedge_sell_ref,
        hedge_inventory=int(round(hedge_inventory)),
        hedge_cash=hedge_cash,
    )


def _update_market_flow_model(model: Any, frame: StrategyInputFrame) -> None:
    if not hasattr(model, "update"):
        return
    # The old market-flow estimators only read t_end/buy/sell volume from Step.
    curr_step = (
        frame.primary.t_end_ms,
        None,
        frame.primary.buy_volume,
        frame.primary.sell_volume,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        None,
        None,
        None,
        None,
        0.0,
        0.0,
    )
    model.update(None, curr_step)


def _update_float_price_model(model: Any, previous: StrategyInputFrame | None, current: StrategyInputFrame) -> None:
    """Update the residual price sampler from float-domain mid changes.

    The old residual updater depends on trade_stats tick/lot Orderbook objects.
    This bridge keeps the same residual model parameters but learns from the new
    float mid transition so controller runs do not need the old stepper objects.
    """

    if model is None or previous is None:
        return
    try:
        prev_mid = _mid(previous.primary.orderbook)
        curr_mid = _mid(current.primary.orderbook)
    except ValueError:
        return
    delta = float(curr_mid - prev_mid)
    gamma = float(getattr(model, "_gamma", 0.95))
    gamma = max(1e-6, min(1.0, gamma))
    alpha = 1.0 - gamma
    sigma_min = float(getattr(model, "_sigma_min", 1e-6))
    mu_buy = gamma * float(getattr(model, "_mu_buy", 0.0)) + alpha * delta
    mu_sell = gamma * float(getattr(model, "_mu_sell", 0.0)) + alpha * delta
    sigma2_buy = gamma * float(getattr(model, "_sigma2_buy", 1.0)) + alpha * (delta - mu_buy) ** 2
    sigma2_sell = gamma * float(getattr(model, "_sigma2_sell", 1.0)) + alpha * (delta - mu_sell) ** 2
    model._mu_buy = float(mu_buy)
    model._mu_sell = float(mu_sell)
    model._sigma2_buy = float(max(sigma_min * sigma_min, sigma2_buy))
    model._sigma2_sell = float(max(sigma_min * sigma_min, sigma2_sell))
    model._last_residual = delta
    sigma_buy = math.sqrt(model._sigma2_buy)
    sigma_sell = math.sqrt(model._sigma2_sell)
    if hasattr(model, "mu_buy_hist"):
        model.mu_buy_hist.append(float(mu_buy))
    if hasattr(model, "mu_sell_hist"):
        model.mu_sell_hist.append(float(mu_sell))
    if hasattr(model, "sigma_buy_hist"):
        model.sigma_buy_hist.append(float(sigma_buy))
    if hasattr(model, "sigma_sell_hist"):
        model.sigma_sell_hist.append(float(sigma_sell))
    if hasattr(model, "residual_mu_hist"):
        model.residual_mu_hist.append(float(0.5 * (mu_buy + mu_sell)))
    if hasattr(model, "residual_sigma_hist"):
        model.residual_sigma_hist.append(float(math.sqrt(0.5 * (sigma_buy * sigma_buy + sigma_sell * sigma_sell))))


def _orders_from_sigmoid_action(
    *,
    actor_id: str,
    event: MarketEvent,
    sigmoid_action: Any,
    min_spread: float,
    max_spread: float,
    qty_step: float,
    min_order_qty: float,
    max_orders_per_side: int,
) -> list[Order]:
    mid = _mid(event.orderbook)
    qty_step = max(0.0, float(qty_step))
    levels = [float(i) for i in range(max(1, int(round(min_spread))), max(1, int(round(max_spread))) + 1)]
    orders: list[Order] = []
    orders.extend(
        _side_orders(
            event=event,
            actor_id=actor_id,
            side="buy",
            mid=mid,
            spreads=levels,
            depth_fn=sigmoid_action.depth_bid,
            qty_step=qty_step,
            min_order_qty=min_order_qty,
            max_orders=max_orders_per_side,
        )
    )
    orders.extend(
        _side_orders(
            event=event,
            actor_id=actor_id,
            side="sell",
            mid=mid,
            spreads=levels,
            depth_fn=sigmoid_action.depth_ask,
            qty_step=qty_step,
            min_order_qty=min_order_qty,
            max_orders=max_orders_per_side,
        )
    )
    return orders


def _side_orders(
    *,
    event: MarketEvent,
    actor_id: str,
    side: str,
    mid: float,
    spreads: list[float],
    depth_fn: Any,
    qty_step: float,
    min_order_qty: float,
    max_orders: int,
) -> list[Order]:
    out: list[Order] = []
    prev_cumulative = float(depth_fn(0.0))
    for idx, spread in enumerate(spreads):
        cumulative = float(depth_fn(spread))
        qty = max(0.0, cumulative - prev_cumulative)
        prev_cumulative = cumulative
        qty = _quantize_qty(qty, qty_step)
        if qty < min_order_qty or qty <= 0.0:
            continue
        price = mid - spread if side == "buy" else mid + spread
        if price <= 0 or not math.isfinite(price):
            continue
        out.append(
            Order(
                exchange=event.exchange,
                symbol=event.symbol,
                side=side,
                price=float(price),
                qty=float(qty),
                client_order_id=f"{actor_id}:{side}:{event.t_end_ms}:{idx}",
            )
        )
        if len(out) >= max(0, int(max_orders)):
            break
    return out


def _quantize_qty(qty: float, qty_step: float) -> float:
    if qty_step <= 0:
        return float(qty)
    return math.floor(float(qty) / qty_step) * qty_step


def _mid(book: RawOrderbook) -> float:
    if book.bids and book.asks:
        return 0.5 * (book.bids[0].price + book.asks[0].price)
    if book.bids:
        return book.bids[0].price
    if book.asks:
        return book.asks[0].price
    raise ValueError("cannot build MPC state from empty orderbook")


def _inventory_cash(snapshot: PrivateStateSnapshot | None, base_asset: str, quote_asset: str) -> tuple[float, float]:
    if snapshot is None:
        return 0.0, 0.0
    base = snapshot.balances.get(base_asset)
    quote = snapshot.balances.get(quote_asset)
    inventory = (base.free + base.locked) if base is not None else 0.0
    cash = (quote.free + quote.locked) if quote is not None else 0.0
    return float(inventory), float(cash)


def _order_summary(order: Order) -> dict[str, Any]:
    return {
        "exchange": order.exchange,
        "symbol": order.symbol,
        "side": order.side,
        "price": order.price,
        "qty": order.qty,
        "notional": order.price * order.qty,
        "client_order_id": order.client_order_id,
    }
