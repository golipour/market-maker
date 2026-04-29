# Observability Stack

LGTM-based observability platform deployed on Kubernetes using Helm charts.

**Namespace:** `observability`

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│  Kubernetes Cluster (3 nodes)                                            │
│                                                                          │
│  ┌──────────────┐    ┌────────────────────┐    ┌──────────────────────┐  │
│  │  Application  │    │  Alloy Agent       │    │  Alloy Gateway       │  │
│  │    Pods       │───▶│  (DaemonSet)       │───▶│  (Deployment)        │  │
│  └──────────────┘    │                    │    │                      │  │
│                      │ • Pod log collect  │    │ • Receives OTLP      │  │
│                      │ • Kubelet/cAdvisor │    │ • Batches signals    │  │
│                      │ • Node metrics     │    │ • Routes traces →    │  │
│                      │   (built-in unix)  │    │   Tempo              │  │
│                      │ • OTLP forwarding  │    │ • Routes metrics →   │  │
│                      │   to gateway       │    │   Mimir              │  │
│                      └──────┬─────────────┘    │ • Scrapes KSM        │  │
│                             │                  └──┬──────────┬────────┘  │
│                      Logs   │        Traces       │  Metrics │           │
│                             ▼                     ▼          ▼           │
│                      ┌────────────┐    ┌───────────┐  ┌──────────────┐  │
│                      │   Loki     │    │   Tempo   │  │    Mimir     │  │
│                      │ (Single    │    │  (Single  │  │ (Micro-      │  │
│                      │  Binary)   │    │   Binary) │  │  services)   │  │
│                      └─────┬──────┘    └─────┬─────┘  └──────┬───────┘  │
│                            │                 │               │           │
│                            └────────┬────────┘               │           │
│                                     ▼                        ▼           │
│                              ┌─────────────┐                            │
│                              │    MinIO    │                            │
│                              │ (Standalone)│                            │
│                              │  S3 store   │                            │
│                              └─────────────┘                            │
│                                                                          │
│  ┌──────────────────┐    ┌──────────────────────────────────────┐       │
│  │  kube-state-      │    │  Grafana (obs-grafana)               │       │
│  │  metrics          │    │  Datasources: Loki, Tempo, Mimir    │       │
│  │  (Deployment)     │    │  Dashboards: dotdc/grafana-dashboards│       │
│  └──────────────────┘    └──────────────────────────────────────┘       │
└──────────────────────────────────────────────────────────────────────────┘
```

## Components

| Component | Chart | Version | Mode | Replicas |
|-----------|-------|---------|------|----------|
| Loki | `grafana/loki` | 6.55.0 | SingleBinary | 1 |
| Mimir | `grafana/mimir-distributed` | 4.5.0 | Microservices (minimal) | 1 per component |
| Tempo | `grafana/tempo` | 1.24.4 | SingleBinary | 1 |
| Grafana | `grafana/grafana` | 10.5.15 | Deployment | 1 |
| Alloy Agent | `grafana/alloy` | 1.7.0 | DaemonSet | 1 per node (3) |
| Alloy Gateway | `grafana/alloy` | 1.7.0 | Deployment | 1 |
| MinIO | `minio/minio` | 5.4.0 | Standalone | 1 |
| kube-state-metrics | `prometheus-community/kube-state-metrics` | latest | Deployment | 1 |

## Data Flow

### Logs
`Application Pods` → `Alloy Agent` (Kubernetes discovery + scraping) → `Loki Gateway` (nginx) → `Loki` → `MinIO` (S3)

### Traces
`Application Pods` → `Alloy Agent` (OTLP) → `Alloy Gateway` (OTLP + batch) → `Tempo` (OTLP gRPC :4317) → `MinIO` (S3)

### Metrics (OTLP path)
`Application Pods` → `Alloy Agent` (OTLP) → `Alloy Gateway` (OTLP + batch) → `Mimir` (OTLP HTTP via nginx `/otlp`) → `MinIO` (S3)

### Metrics (Prometheus scrape path)

| Job | Scraper | Target | Notes |
|-----|---------|--------|-------|
| `kubelet` | Alloy Agent | API server proxy `/metrics` per local node | Bearer token auth, HTTPS |
| `cadvisor` | Alloy Agent | API server proxy `/metrics/cadvisor` per local node | Container resource metrics |
| `node-exporter` | Alloy Agent | Built-in `prometheus.exporter.unix` | Replaces separate node-exporter DaemonSet |
| `kube-state-metrics` | Alloy Gateway | `kube-state-metrics.observability.svc:8080` | Cluster-wide, KSM v2 schema |

All scrape jobs write to Mimir via `prometheus.remote_write` at `http://mimir-nginx/api/v1/push`.

## Grafana Dashboards

All dashboards are provisioned from [Grafana.com](https://grafana.com/grafana/dashboards/) using `gnetId` references — no custom JSON definitions.

| Dashboard | gnetId | Revision | Source |
|-----------|--------|----------|--------|
| Node Exporter Full | 1860 | 37 | [rfmoz](https://grafana.com/grafana/dashboards/1860) |
| Kubernetes / Views / Global | 15757 | 43 | [dotdc](https://grafana.com/grafana/dashboards/15757) |
| Kubernetes / Views / Namespaces | 15758 | 46 | [dotdc](https://grafana.com/grafana/dashboards/15758) |
| Kubernetes / Views / Pods | 15760 | 39 | [dotdc](https://grafana.com/grafana/dashboards/15760) |

The dotdc dashboards are modern, actively maintained, and compatible with kube-state-metrics v2 out of the box.

## Grafana Datasources

| Name | Type | URL | Default |
|------|------|-----|---------|
| Mimir | `prometheus` | `http://mimir-nginx/prometheus` | Yes |
| Loki | `loki` | `http://loki-gateway` | No |
| Tempo | `tempo` | `http://tempo` | No |

## Object Storage (MinIO)

A shared standalone MinIO instance serves as the S3-compatible object store for all backends.

| Bucket | Used By |
|--------|---------|
| `loki-chunks` | Loki — log chunk data |
| `loki-ruler` | Loki — ruler configs |
| `loki-admin` | Loki — admin data |
| `mimir-tsdb` | Mimir — TSDB blocks |
| `mimir-ruler` | Mimir — ruler configs |
| `mimir-alertmanager` | Mimir — alertmanager state |
| `tempo-traces` | Tempo — trace data (reserved) |

**Endpoint:** `minio:9000` (ClusterIP)  
**Storage:** 10Gi PVC via `nfs-client` storage class

## Alloy Pipeline

### Agent (DaemonSet)
- **Log collection:** `discovery.kubernetes` (pod role) → `loki.source.kubernetes` → `loki.write` to `loki-gateway`
- **OTLP forwarding:** `otelcol.receiver.otlp` (gRPC + HTTP) → `otelcol.exporter.otlp` to `alloy-gateway:4317`
- **Kubelet metrics:** API server proxy → `prometheus.scrape` (HTTPS, bearer token, filtered to local node) → `prometheus.remote_write` to Mimir
- **cAdvisor metrics:** API server proxy → `prometheus.scrape` (same auth, filtered to local node) → `prometheus.remote_write` to Mimir
- **Node metrics:** `prometheus.exporter.unix` (built-in) → `prometheus.relabel` (sets `job=node-exporter`, `node=HOSTNAME`) → `prometheus.remote_write` to Mimir

### Gateway (Deployment)
- **Receives:** OTLP (gRPC + HTTP) from agents
- **Processes:** `otelcol.processor.batch` for efficient batching
- **Routes:**
  - Traces → `otelcol.exporter.otlp` to `tempo:4317`
  - Metrics → `otelcol.exporter.otlphttp` to `mimir-nginx/otlp`
- **Scrapes:** `kube-state-metrics.observability.svc:8080` → `prometheus.remote_write` to Mimir

## Design Decisions

### Shared MinIO over per-chart MinIO subcharts
Both Loki and Mimir Helm charts bundle their own MinIO subchart. Running separate MinIO instances wastes resources and creates ServiceAccount naming conflicts. A single standalone MinIO instance is simpler to manage.

### Alloy Agent + Gateway split
The agent runs as a DaemonSet for node-level log collection, local metrics scraping, and OTLP ingestion. The gateway centralizes signal routing, batching, and cluster-wide scraping. This decouples collection from routing and allows independent scaling.

### Built-in node metrics (no separate node-exporter)
Alloy's `prometheus.exporter.unix` provides node-level metrics (CPU, memory, disk, network) without deploying a separate node-exporter DaemonSet, reducing cluster resource usage.

### kube-state-metrics v2 + modern dashboards
kube-state-metrics v2 uses `resource` and `unit` labels (e.g., `kube_pod_container_resource_requests{resource="memory",unit="byte"}`) instead of v1 suffixed metric names. The dotdc community dashboards are built for this schema.

### Mimir microservices with minimal replicas
Mimir runs in microservices mode with 1 replica per component. Zone-aware replication and non-essential components (alertmanager, ruler, query-scheduler, overrides-exporter, rollout-operator) are disabled for dev.

### Loki SingleBinary mode
Loki runs as a single binary with `auth_enabled: false` and `replication_factor: 1` for dev simplicity.

### CoreDNS override
Both Loki and Mimir nginx configs require `global.dnsService: coredns` since the cluster uses CoreDNS.

### Grafana `fullnameOverride`
Set to `obs-grafana` to avoid ClusterRole naming conflicts with existing Grafana installations.

## Repository Structure

```
base/
  alloy-agent.yaml              # Alloy agent (DaemonSet) Helm values
  alloy-gateway.yaml            # Alloy gateway (Deployment) Helm values
  grafana.yaml                  # Grafana Helm values + datasources + dashboards
  kube-state-metrics-values.yml # kube-state-metrics Helm values
  loki-values.yml               # Loki Helm values (SingleBinary + S3)
  mimir-values.yml              # Mimir Helm values (microservices + S3)
  minio-values.yml              # Standalone MinIO Helm values
  tempo-values.yml              # Tempo Helm values
environments/
  dev.yml                       # Dev environment overrides (placeholder)
  prod.yml                      # Prod environment overrides (placeholder)
```

## Deployment

### Prerequisites
- Kubernetes >= 1.26
- Storage class: `nfs-client` (or equivalent)
- CoreDNS as cluster DNS provider

### Add Helm repos

```bash
helm repo add grafana https://grafana.github.io/helm-charts
helm repo add minio https://charts.min.io
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update
```

### Install / Upgrade

```bash
kubectl create namespace observability --dry-run=client -o yaml | kubectl apply -f -

# Storage
helm upgrade --install minio minio/minio \
  --version 5.4.0 -n observability -f base/minio-values.yml

# Backends
helm upgrade --install loki grafana/loki \
  -n observability -f base/loki-values.yml
helm upgrade --install mimir grafana/mimir-distributed \
  --version 4.5.0 -n observability -f base/mimir-values.yml
helm upgrade --install tempo grafana/tempo \
  -n observability -f base/tempo-values.yml

# Metrics source
helm upgrade --install kube-state-metrics prometheus-community/kube-state-metrics \
  -n observability -f base/kube-state-metrics-values.yml

# Collection & routing
helm upgrade --install alloy-agent grafana/alloy \
  -n observability -f base/alloy-agent.yaml
helm upgrade --install alloy-gateway grafana/alloy \
  -n observability -f base/alloy-gateway.yaml

# Visualization
helm upgrade --install grafana grafana/grafana \
  -n observability -f base/grafana.yaml
```

### Verify

```bash
# All pods healthy
kubectl get pods -n observability

# Scrape targets up
kubectl exec -n observability deploy/mimir-querier -- \
  wget -qO- 'http://mimir-nginx/prometheus/api/v1/query?query=up'
```
