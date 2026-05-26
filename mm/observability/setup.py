from __future__ import annotations

import os
from pathlib import Path

from mm.env_loader import repo_root
from mm.observability.logs import log_event
from mm.observability import schema as obs_schema
from mm.runtime.config import ObservabilityRuntimeConfig


def _resolve_log_path_for_process(path: str) -> str:
    """Resolve relative log paths against the git repo root.

    Docker Compose mounts ``./logs`` at the repository root for Alloy. If the
    runtime is started from another working directory, a relative ``log_path``
    would write outside that mount and Loki would stay empty.
    """

    p = Path(path.strip())
    if p.is_absolute():
        return str(p)
    return str((repo_root() / p).resolve())


def configure_observability(config: ObservabilityRuntimeConfig) -> None:
    """Apply observability-related environment and optional OTLP / metrics HTTP server."""

    if config.json_logs:
        os.environ.setdefault("MM_JSON_LOG", "1")
    if config.log_path:
        try:
            os.environ.setdefault("MM_LOG_PATH", _resolve_log_path_for_process(str(config.log_path)))
        except Exception as exc:
            log_event(
                obs_schema.EVENT_OBSERVABILITY_SETUP_FAILED,
                {
                    "component": "logs",
                    "error": type(exc).__name__,
                    "message": str(exc),
                },
            )
    if config.metrics_enabled:
        os.environ.setdefault("MM_METRICS_PORT", str(config.metrics_port))
        try:
            from mm.observability.registry import start_server

            status = start_server(port=config.metrics_port)
            if status.get("status") in {"started", "already_running"}:
                log_event(
                    obs_schema.EVENT_OBSERVABILITY_METRICS_READY,
                    {
                        "component": "metrics",
                        "status": status.get("status"),
                        "host": status.get("host"),
                        "port": status.get("port"),
                        "endpoint": status.get("endpoint"),
                    },
                )
        except Exception as exc:
            details = {
                "component": "metrics",
                "host": "0.0.0.0",
                "port": config.metrics_port,
                "error": type(exc).__name__,
                "message": str(exc),
            }
            if isinstance(exc, PermissionError):
                details["hint"] = "socket creation or bind was denied; run outside a restricted sandbox or disable metrics"
            log_event(
                obs_schema.EVENT_OBSERVABILITY_SETUP_FAILED,
                details,
            )
    if config.run_id:
        os.environ.setdefault("MM_RUN_ID", config.run_id)
    if config.service_name:
        os.environ.setdefault("MM_SERVICE", config.service_name)
    if not config.otel_enabled:
        return
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except Exception as exc:
        log_event(
            obs_schema.EVENT_OBSERVABILITY_SETUP_FAILED,
            {
                "component": "otel_import",
                "error": type(exc).__name__,
                "message": str(exc),
            },
        )
        return
    endpoint = config.otlp_endpoint or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    try:
        resource = Resource.create({"service.name": config.service_name, "mm.run_id": config.run_id})
        provider = TracerProvider(resource=resource)
        exporter_kwargs = {"endpoint": endpoint.rstrip("/") + "/v1/traces"} if endpoint else {}
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(**exporter_kwargs)))
        trace.set_tracer_provider(provider)
    except Exception as exc:
        log_event(
            obs_schema.EVENT_OBSERVABILITY_SETUP_FAILED,
            {
                "component": "otel",
                "error": type(exc).__name__,
                "message": str(exc),
            },
        )
