import { ReactNode, useState } from "react";

type Props = {
  id: string;
  title: string;
  help?: string;
  defaultOpen?: boolean;
  /** Optional count rendered as a badge, e.g. the number of overridden fields. */
  badge?: number;
  children: ReactNode;
};

export function Section({ id, title, help, defaultOpen = true, badge, children }: Props) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <section className="advanced-section" data-section={id}>
      <button
        type="button"
        className="advanced-section-header"
        aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
      >
        <span className="advanced-section-chevron">{open ? "▾" : "▸"}</span>
        <span>{title}</span>
        {badge !== undefined ? (
          <span className="advanced-section-badge">{badge}</span>
        ) : null}
        {help ? (
          <span className="field-info" title={help} onClick={(e) => e.stopPropagation()}>
            ⓘ
          </span>
        ) : null}
      </button>
      {open ? <div className="advanced-section-body">{children}</div> : null}
    </section>
  );
}
