"""Shared public market-data ingestor entrypoint.

Run: ``python -m mm.ingestor --config mm/configs/market_maker.ingestor.example.json``
"""

from __future__ import annotations

import argparse
import asyncio
import json
import signal

from mm.runtime.ingestor import IngestorComponents, build_ingestor_components
from mm.runtime.ingestor_config import load_ingestor_config_from_registry


async def run(args: argparse.Namespace) -> None:
    config = load_ingestor_config_from_registry(args.config)
    ingestor = await build_ingestor_components(config)
    contract = config.as_contract()
    contract["type"] = "IngestorConfigCheck"
    contract["selected_public_market_transports"] = {
        key: selection.transport
        for key, selection in ingestor.public_market_selections.items()
    }
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

    await ingestor.start()
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
                            "message": f"ingestor stopped after timeout_s={args.timeout_s}",
                        }
                    ),
                    flush=True,
                )
        else:
            await stop_event.wait()
    finally:
        await ingestor.stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run shared public market-data ingestor.")
    parser.add_argument("--config", required=True, help="Path to ingestor JSON config.")
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
