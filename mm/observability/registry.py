from __future__ import annotations

import math
import os
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any


DEFAULT_HISTOGRAM_BUCKETS_MS = (
    1.0,
    2.5,
    5.0,
    10.0,
    25.0,
    50.0,
    100.0,
    250.0,
    500.0,
    1000.0,
    2500.0,
    5000.0,
    float("inf"),
)


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


def _normalize_histogram_buckets(buckets: tuple[float, ...] | None) -> tuple[float, ...]:
    b = DEFAULT_HISTOGRAM_BUCKETS_MS if buckets is None else buckets
    if not b:
        return (float("inf"),)
    if b[-1] != float("inf") and not math.isinf(b[-1]):
        return tuple(b) + (float("inf"),)
    return tuple(b)


def _le_label(upper: float) -> str:
    if math.isinf(upper) and upper > 0:
        return "+Inf"
    if upper == int(upper):
        return str(int(upper))
    return str(upper)


@dataclass(frozen=True, slots=True)
class _SeriesKey:
    name: str
    labels: tuple[tuple[str, str], ...]


class _HistogramAgg:
    __slots__ = ("edges", "cumulative_counts", "sum_v", "count_v")

    def __init__(self, edges: tuple[float, ...]) -> None:
        self.edges = edges
        self.cumulative_counts = [0.0] * len(edges)
        self.sum_v = 0.0
        self.count_v = 0.0

    def observe(self, value: float) -> None:
        self.count_v += 1.0
        self.sum_v += float(value)
        for i, upper in enumerate(self.edges):
            if value <= upper:
                self.cumulative_counts[i] += 1.0


class MetricsRegistry:
    """Small Prometheus text-format registry used by the async runtime."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._help: dict[str, str] = {}
        self._type: dict[str, str] = {}
        self._values: dict[_SeriesKey, float] = {}
        self._histogram_data: dict[_SeriesKey, _HistogramAgg] = {}
        self._histogram_help: dict[str, str] = {}

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

    def histogram_observe(
        self,
        name: str,
        value: float,
        *,
        labels: dict[str, Any] | None = None,
        help: str = "",
        buckets: tuple[float, ...] | None = None,
    ) -> None:
        labels = labels or {}
        edges = _normalize_histogram_buckets(buckets)
        key = _SeriesKey(name=name, labels=tuple(sorted((k, _sanitize_label_value(v)) for k, v in labels.items())))
        with self._lock:
            if help:
                self._histogram_help[name] = help
            agg = self._histogram_data.get(key)
            if agg is None:
                agg = _HistogramAgg(edges)
                self._histogram_data[key] = agg
            agg.observe(float(value))

    def render(self) -> str:
        with self._lock:
            lines: list[str] = []
            for name in sorted(set(self._type.keys()) | set(self._help.keys())):
                h = self._help.get(name)
                if h:
                    lines.append(f"# HELP {name} {h}")
                t = self._type.get(name)
                if t:
                    lines.append(f"# TYPE {name} {t}")
            for key, value in sorted(self._values.items(), key=lambda kv: (kv[0].name, kv[0].labels)):
                if key.labels:
                    lbl = ",".join([f'{k}="{v}"' for k, v in key.labels])
                    lines.append(f"{key.name}{{{lbl}}} {value}")
                else:
                    lines.append(f"{key.name} {value}")

            hist_metric_names = sorted({k.name for k in self._histogram_data.keys()})
            for name in hist_metric_names:
                h = self._histogram_help.get(name)
                if h:
                    lines.append(f"# HELP {name} {h}")
                lines.append(f"# TYPE {name} histogram")
            for key in sorted(self._histogram_data.keys(), key=lambda k: (k.name, k.labels)):
                agg = self._histogram_data[key]
                lbl_base = ",".join([f'{k}="{v}"' for k, v in key.labels])
                for i, upper in enumerate(agg.edges):
                    le = _le_label(upper)
                    if lbl_base:
                        lines.append(f'{key.name}_bucket{{{lbl_base},le="{le}"}} {agg.cumulative_counts[i]}')
                    else:
                        lines.append(f'{key.name}_bucket{{le="{le}"}} {agg.cumulative_counts[i]}')
                if lbl_base:
                    lines.append(f'{key.name}_sum{{{lbl_base}}} {agg.sum_v}')
                    lines.append(f'{key.name}_count{{{lbl_base}}} {agg.count_v}')
                else:
                    lines.append(f"{key.name}_sum {agg.sum_v}")
                    lines.append(f"{key.name}_count {agg.count_v}")

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
            return

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        return


_server_thread: threading.Thread | None = None


def start_server(host: str = "0.0.0.0", port: int | None = None) -> None:
    """Start a daemon-thread `/metrics` server when metrics are enabled."""

    global _server_thread
    if not _enabled():
        return
    if _server_thread is not None and _server_thread.is_alive():
        return
    port = int(port if port is not None else _port())
    httpd = HTTPServer((host, port), _Handler)

    def _run() -> None:
        httpd.serve_forever(poll_interval=0.5)

    _server_thread = threading.Thread(target=_run, daemon=True, name="mm-metrics")
    _server_thread.start()
