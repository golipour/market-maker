export type RegistryEditMode = "form" | "json" | "hybrid";

const COPY: Record<RegistryEditMode, string> = {
  form: "Structured form — scalar registry fields. Secrets are .env names only, never values.",
  json: "JSON editor — nested runtime config blobs (presets, platform settings).",
  hybrid:
    "Hybrid — form for identity/status; advanced Form|JSON editor for nested preset/override config.",
};

interface EditModeNoteProps {
  mode: RegistryEditMode;
}

export function EditModeNote({ mode }: EditModeNoteProps) {
  return (
    <p className="edit-mode-note" style={{ color: "var(--text-muted)", fontSize: "0.85rem", marginTop: "0.5rem" }}>
      {COPY[mode]}
    </p>
  );
}
