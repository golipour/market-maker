interface GrafanaPanelProps {
  title: string;
  embedUrl: string;
  openUrl?: string;
  variant?: "solo" | "full";
}

export function GrafanaPanel({
  title,
  embedUrl,
  openUrl,
  variant = "solo",
}: GrafanaPanelProps) {
  return (
    <div className={`grafana-panel grafana-panel--${variant}`}>
      <div className="grafana-panel-header">
        <span>{title}</span>
        {openUrl ? (
          <a href={openUrl} target="_blank" rel="noreferrer">
            Open in Grafana
          </a>
        ) : null}
      </div>
      <iframe
        title={title}
        src={embedUrl}
        loading="lazy"
        referrerPolicy="no-referrer"
        sandbox="allow-scripts allow-same-origin allow-popups allow-forms allow-popups-to-escape-sandbox"
      />
    </div>
  );
}
