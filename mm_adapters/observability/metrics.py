from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any


def _enabled() -> bool:
    return bool(str(os.environ.get("MM_METRICS_PORT", "")).strip())


def _port() -> int:
    v = str(os.environ.get("MM_METRICS_PORT", "")).strip()
    try:
        return int(v)
    except Exception:
        return 8000


def _sanitize_label_value(v: Any) -> str:
    s = "" if v is None else str(v)
    return s.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


@dataclass(frozen=True, slots=True)
class _SeriesKey:
    name: str
    labels: tuple[tuple[str, str], ...]


class MetricsRegistry:
    """
    Very small Prometheus text-format registry.

    Enough for local dev and for scraping/remote_write pipelines.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._help: dict[str, str] = {}
        self._type: dict[str, str] = {}
        self._values: dict[_SeriesKey, float] = {}

    def gauge(self, name: str, value: float, *, labels: dict[str, Any] | None = None, help: str = "") -> None:
        labels = labels or {}
        key = _SeriesKey(name=name, labels=tuple(sorted((k, _sanitize_label_value(v)) for k, v in labels.items())))
        with self._lock:
            if help:
                self._help[name] = help
            self._type[name] = "gauge"
            self._values[key] = float(value)

    def counter_inc(self, name: str, inc: float = 1.0, *, labels: dict[str, Any] | None = None, help: str = "") -> None:
        labels = labels or {}
        key = _SeriesKey(name=name, labels=tuple(sorted((k, _sanitize_label_value(v)) for k, v in labels.items())))
        with self._lock:
            if help:
                self._help[name] = help
            self._type[name] = "counter"
            self._values[key] = float(self._values.get(key, 0.0) + float(inc))

    def render(self) -> str:
        with self._lock:
            lines: list[str] = []
            # HELP/TYPE
            for name in sorted(set(self._type.keys()) | set(self._help.keys())):
                h = self._help.get(name)
                if h:
                    lines.append(f"# HELP {name} {h}")
                t = self._type.get(name)
                if t:
                    lines.append(f"# TYPE {name} {t}")
            # Samples
            for key, value in sorted(self._values.items(), key=lambda kv: (kv[0].name, kv[0].labels)):
                if key.labels:
                    lbl = ",".join([f'{k}="{v}"' for k, v in key.labels])
                    lines.append(f"{key.name}{{{lbl}}} {value}")
                else:
                    lines.append(f"{key.name} {value}")
            return "\n".join(lines) + "\n"


REGISTRY = MetricsRegistry()


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        try:
            if self.path not in ("/metrics", "/metrics/"):
                self.send_response(404)
                self.end_headers()
                return
            body = REGISTRY.render().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            # Scraper closed the socket early (timeout / cancel); normal under Prometheus/Grafana.
            return

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        # Silence default http.server logs; rely on structured logging instead.
        return


_server_thread: threading.Thread | None = None


def start_server(host: str = "0.0.0.0", port: int | None = None) -> None:
    """
    Start a /metrics server in a daemon thread.
    No-op unless MM_METRICS_PORT is set.
    """
    global _server_thread
    if not _enabled():
        return
    if _server_thread is not None and _server_thread.is_alive():
        return
    port = int(port if port is not None else _port())

    def _run():
        httpd = HTTPServer((host, port), _Handler)
        httpd.serve_forever(poll_interval=0.5)

    _server_thread = threading.Thread(target=_run, daemon=True, name="mm-metrics")
    _server_thread.start()

