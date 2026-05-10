from __future__ import annotations

import os

from mm.runtime.config import ObservabilityRuntimeConfig


def configure_observability(config: ObservabilityRuntimeConfig) -> None:
    """Configure optional logs/metrics/traces for the async runtime."""

    if config.json_logs:
        os.environ.setdefault("MM_JSON_LOG", "1")
    if config.log_path:
        os.environ.setdefault("MM_LOG_PATH", config.log_path)
    if config.metrics_enabled:
        os.environ.setdefault("MM_METRICS_PORT", str(config.metrics_port))
        try:
            from mm.observability_metrics import start_server

            start_server(port=config.metrics_port)
        except Exception:
            pass
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
    except Exception:
        return
    endpoint = config.otlp_endpoint or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    resource = Resource.create({"service.name": config.service_name, "mm.run_id": config.run_id})
    provider = TracerProvider(resource=resource)
    exporter_kwargs = {"endpoint": endpoint.rstrip("/") + "/v1/traces"} if endpoint else {}
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(**exporter_kwargs)))
    trace.set_tracer_provider(provider)
