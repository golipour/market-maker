import type { RuntimeOverviewStrategy, RuntimeOverviewWorker, TradingStatus, WorkerLiveStatus } from "../api/client";

export function workerLiveLabel(live: WorkerLiveStatus): string {
  switch (live) {
    case "alive":
      return "Alive";
    case "stale":
      return "Stale";
    case "offline":
      return "Offline";
  }
}

export function workerLiveClass(live: WorkerLiveStatus): string {
  switch (live) {
    case "alive":
      return "status-up";
    case "stale":
      return "status-warning";
    case "offline":
      return "status-down";
  }
}

export function tradingStatusLabel(status: TradingStatus): string {
  switch (status) {
    case "running":
      return "Running";
    case "paused":
      return "Paused";
    case "breach":
      return "Breach";
    case "kill_switch":
      return "Kill switch";
    case "dry_run":
      return "Dry run";
    case "worker_offline":
      return "Worker offline";
    case "unassigned":
      return "No worker";
    case "registry_inactive":
      return "Inactive";
  }
}

export function tradingStatusClass(status: TradingStatus): string {
  switch (status) {
    case "running":
      return "status-up";
    case "paused":
      return "status-warning";
    case "breach":
    case "kill_switch":
      return "status-down";
    case "dry_run":
      return "status-muted";
    case "worker_offline":
    case "unassigned":
    case "registry_inactive":
      return "status-muted";
  }
}

export function runtimeByStrategyId(
  strategies: RuntimeOverviewStrategy[],
): Map<string, RuntimeOverviewStrategy> {
  return new Map(strategies.map((row) => [row.strategy_id, row]));
}

export function runtimeByWorkerId(
  workers: RuntimeOverviewWorker[],
): Map<string, RuntimeOverviewWorker> {
  const map = new Map<string, RuntimeOverviewWorker>();
  for (const row of workers) {
    const previous = map.get(row.worker_id);
    if (!previous || liveRank(row.live) > liveRank(previous.live)) {
      map.set(row.worker_id, row);
    }
  }
  return map;
}

function liveRank(live: WorkerLiveStatus): number {
  return live === "alive" ? 3 : live === "stale" ? 2 : 1;
}

export function canStartTrading(status: TradingStatus | undefined): boolean {
  return status === "paused" || status === "breach" || status === "worker_offline";
}

export function canStopTrading(status: TradingStatus | undefined): boolean {
  return status === "running";
}
