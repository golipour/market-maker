"""Standalone persistence subscriber entrypoint.

Run: ``python -m mm.persistence --config mm/configs/market_maker.persistence.example.json``
"""

from __future__ import annotations

import argparse
import asyncio
import json
import signal

from mm.persistence.service import build_persistence_service
from mm.persistence.service_config import load_persistence_service_config


async def run(args: argparse.Namespace) -> None:
    config = load_persistence_service_config(args.config)
    service = await build_persistence_service(config)
    contract = config.as_contract()
    contract["type"] = "PersistenceConfigCheck"
    contract["writers"] = [type(writer).__name__ for writer in service.writers]
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

    await service.start()
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
                            "message": f"persistence service stopped after timeout_s={args.timeout_s}",
                        }
                    ),
                    flush=True,
                )
        else:
            await stop_event.wait()
    finally:
        await service.stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run standalone persistence subscriber.")
    parser.add_argument("--config", required=True, help="Path to persistence JSON config.")
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
