import type { ServiceProbe } from "../api/client";

interface ServiceCardProps {
  service: ServiceProbe;
}

export function ServiceCard({ service }: ServiceCardProps) {
  const up = service.status === "up";
  return (
    <div className="card">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <h3>{service.name}</h3>
        <span className={`status-badge ${up ? "status-up" : "status-down"}`}>
          {service.status}
        </span>
      </div>
      <p>ID: {service.id}</p>
      <p>
        Metrics:{" "}
        {service.metrics_ok ? (
          <span style={{ color: "var(--success)" }}>ok</span>
        ) : (
          <span style={{ color: "var(--danger)" }}>fail</span>
        )}
      </p>
      {service.health_url ? (
        <p>
          Health:{" "}
          {service.health_ok ? (
            <span style={{ color: "var(--success)" }}>ok</span>
          ) : (
            <span style={{ color: "var(--danger)" }}>fail</span>
          )}
        </p>
      ) : null}
      <p style={{ fontSize: "0.8rem" }}>
        {service.detail === "not running"
          ? "Service not running on this host (start it separately)."
          : service.detail}
      </p>
      <a href={service.metrics_url} target="_blank" rel="noreferrer">
        Metrics endpoint
      </a>
    </div>
  );
}
