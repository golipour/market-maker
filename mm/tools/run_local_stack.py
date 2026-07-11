"""Run the decomposed local stack in one command.

Example::

    python3 -m mm.tools.run_local_stack
    python3 -m mm.tools.run_local_stack --no-with-admin
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from mm.env_loader import load_mm_env
from mm.registry.store import registry_database_url
from mm.tools.seed_local_registry import ensure_local_worker_registry

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIGS = {
    "ingestor": ROOT / "mm/configs/market_maker.ingestor.example.json",
    "consumer": ROOT / "mm/configs/market_maker.runtime.consumer.example.json",
    "persistence": ROOT / "mm/configs/market_maker.persistence.example.json",
    "worker": ROOT / "mm/configs/market_maker.worker.example.json",
}


@dataclass(frozen=True, slots=True)
class StackService:
    name: str
    module: str
    config: Path
    start_first: bool = False


def default_services(configs: dict[str, Path]) -> list[StackService]:
    return [
        StackService("ingestor", "mm.ingestor", configs["ingestor"], start_first=True),
        StackService("persistence", "mm.persistence", configs["persistence"]),
        StackService("worker", "mm.worker", configs["worker"]),
        StackService("consumer", "mm.tools.print_events", configs["consumer"]),
    ]


def build_command(service: StackService) -> list[str]:
    return [sys.executable, "-m", service.module, "--config", str(service.config)]


def _prefix_stream(stream: subprocess.PIPE, prefix: str) -> None:
    assert stream is not None
    for line in stream:
        print(f"[{prefix}] {line.rstrip()}", flush=True)


def _start_service(service: StackService) -> subprocess.Popen[str]:
    command = build_command(service)
    print(f"[stack] starting {service.name}: {' '.join(command)}", flush=True)
    proc = subprocess.Popen(
        command,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        start_new_session=True,
    )
    thread = threading.Thread(
        target=_prefix_stream,
        args=(proc.stdout, service.name),
        daemon=True,
        name=f"log-{service.name}",
    )
    thread.start()
    return proc


def _terminate_process(proc: subprocess.Popen[str], *, sig: signal.Signals) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, sig)
    except ProcessLookupError:
        return
    except PermissionError:
        # macOS / sandbox: may not signal the whole group; fall back to the child.
        try:
            proc.send_signal(sig)
        except ProcessLookupError:
            return


def _terminate_all(procs: list[subprocess.Popen[str]]) -> None:
    for proc in procs:
        _terminate_process(proc, sig=signal.SIGTERM)
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if all(proc.poll() is not None for proc in procs):
            return
        time.sleep(0.1)
    for proc in procs:
        _terminate_process(proc, sig=signal.SIGKILL)


def _maybe_seed_worker_registry(services: list[StackService]) -> None:
    if not any(svc.name == "worker" for svc in services):
        return
    if not registry_database_url():
        print(
            "[stack] warning: MM_DATABASE_URL unset; worker may fail without registry seed",
            flush=True,
        )
        return
    if ensure_local_worker_registry():
        print("[stack] seeded local registry for worker-1", flush=True)


def _redis_url_from_env() -> str:
    return os.environ.get("MM_BUS_REDIS_URL", "redis://localhost:6379/0").strip()


def _check_redis_available(redis_url: str, *, timeout_s: float = 60.0) -> bool:
    try:
        import redis
        from redis.exceptions import BusyLoadingError
    except ImportError:
        return True
    deadline = time.monotonic() + max(1.0, float(timeout_s))
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        client = redis.from_url(redis_url, socket_connect_timeout=2.0, socket_timeout=2.0)
        try:
            if client.ping():
                return True
        except BusyLoadingError as exc:
            last_error = exc
            print(
                "[stack] waiting for Redis to finish loading dataset…",
                flush=True,
            )
        except Exception as exc:
            last_error = exc
        finally:
            client.close()
        time.sleep(1.0)
    if last_error is not None:
        print(
            f"[stack] warning: Redis not ready at {redis_url} "
            f"({type(last_error).__name__}: {last_error})",
            flush=True,
        )
    else:
        print(f"[stack] warning: Redis not ready at {redis_url}", flush=True)
    print(
        "[stack] start infrastructure first: docker compose up -d redis db",
        flush=True,
    )
    return False


def run_stack(
    services: list[StackService],
    *,
    with_admin: bool,
    ingestor_warmup_s: float,
) -> int:
    load_mm_env()
    if not _check_redis_available(_redis_url_from_env()):
        return 1
    _maybe_seed_worker_registry(services)
    procs: list[tuple[str, subprocess.Popen[str]]] = []
    stop_requested = False

    def _handle_stop(signum: int, _frame: object) -> None:
        nonlocal stop_requested
        if stop_requested:
            _terminate_all([proc for _, proc in procs])
            raise SystemExit(128 + signum)
        stop_requested = True
        print("[stack] shutting down…", flush=True)
        _terminate_all([proc for _, proc in procs])

    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    try:
        first = [svc for svc in services if svc.start_first]
        rest = [svc for svc in services if not svc.start_first]
        for service in first:
            procs.append((service.name, _start_service(service)))
        if first and ingestor_warmup_s > 0:
            time.sleep(ingestor_warmup_s)
        for service in rest:
            procs.append((service.name, _start_service(service)))
        if with_admin:
            admin_cmd = [sys.executable, "-m", "mm.api.admin"]
            print(f"[stack] starting admin: {' '.join(admin_cmd)}", flush=True)
            admin = subprocess.Popen(
                admin_cmd,
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
            threading.Thread(
                target=_prefix_stream,
                args=(admin.stdout, "admin"),
                daemon=True,
                name="log-admin",
            ).start()
            procs.append(("admin", admin))

        print("[stack] all services started (Ctrl+C to stop)", flush=True)
        while True:
            for name, proc in procs:
                code = proc.poll()
                if code is not None:
                    print(f"[stack] {name} exited with code {code}", flush=True)
                    _terminate_all([p for _, p in procs if p is not proc])
                    return code if code is not None else 1
            time.sleep(0.5)
    finally:
        _terminate_all([proc for _, proc in procs])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ingestor, consumer, persistence, worker (and optionally admin) locally.",
    )
    parser.add_argument(
        "--with-admin",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Start the control-plane admin API on :8080 (default: on). Use --no-with-admin to skip.",
    )
    parser.add_argument(
        "--ingestor-warmup-s",
        type=float,
        default=2.0,
        help="Seconds to wait after ingestor starts before launching other services.",
    )
    for name in DEFAULT_CONFIGS:
        parser.add_argument(
            f"--{name.replace('_', '-')}-config",
            default=str(DEFAULT_CONFIGS[name]),
            help=f"Path to {name} JSON config.",
        )
    parser.add_argument(
        "--skip",
        action="append",
        nargs="+",
        choices=tuple(DEFAULT_CONFIGS.keys()),
        default=[],
        help="Skip one or more stack services (repeatable).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    configs = {
        name: Path(getattr(args, f"{name}_config"))
        for name in DEFAULT_CONFIGS
    }
    missing = [name for name, path in configs.items() if not path.is_file()]
    if missing:
        raise SystemExit(f"Config file(s) not found: {', '.join(missing)}")

    skip = {name for batch in args.skip for name in batch}
    services = [svc for svc in default_services(configs) if svc.name not in skip]
    if not services:
        raise SystemExit("No services selected to run.")

    code = run_stack(
        services,
        with_admin=args.with_admin,
        ingestor_warmup_s=args.ingestor_warmup_s,
    )
    raise SystemExit(code)


if __name__ == "__main__":
    main()
