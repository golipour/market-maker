"""Multi-strategy worker entrypoint.

Run: ``python -m mm.worker --config mm/configs/market_maker.worker.example.json``
"""

from __future__ import annotations

import argparse
import asyncio
import json
import signal

from mm.worker.worker import build_worker_components
from mm.worker.worker_config import load_worker_config


async def run(args: argparse.Namespace) -> None:
    config = load_worker_config(args.config)
    worker = await build_worker_components(config)
    contract = config.as_contract()
    contract["type"] = "WorkerConfigCheck"
    contract["assignment_id"] = worker.assignment.assignment_id
    contract["loaded_strategy_ids"] = [item.strategy_id for item in worker.strategies]
    contract["shared_exchanges"] = list(worker.account_pool.order_managers.keys())
    print(json.dumps(contract, ensure_ascii=False), flush=True)

    stop_event = asyncio.Event()

    def _request_stop(*_args: object) -> None:
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _request_stop)
        except NotImplementedError:
            signal.signal(sig, lambda *_: stop_event.set())

    await worker.start()
    try:
        if args.timeout_s > 0:
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=args.timeout_s)
            except asyncio.TimeoutError:
                print(
                    json.dumps(
                        {
                            "type": "Timeout",
                            "timeout_s": args.timeout_s,
                            "message": f"worker stopped after timeout_s={args.timeout_s}",
                        }
                    ),
                    flush=True,
                )
        else:
            await stop_event.wait()
    finally:
        await worker.stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-strategy worker.")
    parser.add_argument("--config", required=True, help="Path to worker JSON config.")
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=0.0,
        help="Stop after this many seconds; 0 runs until SIGINT/SIGTERM.",
    )
    return parser.parse_args()


def main() -> None:
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
