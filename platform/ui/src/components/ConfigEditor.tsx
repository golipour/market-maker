import { useEffect, useState } from "react";
import { PresetConfigForm, type PresetConfigFormMode } from "./PresetConfigForm";
import {
  cloneConfig,
  isPlainObject,
  type JsonObject,
} from "../lib/presetConfig";

type Props = {
  value: JsonObject;
  onChange: (next: JsonObject) => void;
  mode: PresetConfigFormMode;
  inherited?: JsonObject | null;
  /** Optional label above the editor. */
  label?: string;
};

type ViewMode = "form" | "json";

export function ConfigEditor({ value, onChange, mode, inherited = null, label }: Props) {
  const [view, setView] = useState<ViewMode>("form");
  const [jsonText, setJsonText] = useState(() => JSON.stringify(value ?? {}, null, 2));
  const [jsonError, setJsonError] = useState<string | null>(null);

  useEffect(() => {
    if (view === "form") {
      setJsonText(JSON.stringify(value ?? {}, null, 2));
      setJsonError(null);
    }
  }, [value, view]);

  function switchToForm() {
    try {
      const parsed = JSON.parse(jsonText) as unknown;
      if (!isPlainObject(parsed)) {
        setJsonError("Config must be a JSON object");
        return;
      }
      onChange(cloneConfig(parsed));
      setJsonError(null);
      setView("form");
    } catch (err) {
      setJsonError(err instanceof Error ? err.message : "Invalid JSON");
    }
  }

  function switchToJson() {
    setJsonText(JSON.stringify(value ?? {}, null, 2));
    setJsonError(null);
    setView("json");
  }

  return (
    <div className="config-editor">
      <div className="config-editor-toolbar">
        {label ? <span className="config-editor-label">{label}</span> : null}
        <div className="view-tabs" role="tablist">
          <button
            type="button"
            role="tab"
            className={view === "form" ? "view-tab active" : "view-tab"}
            aria-selected={view === "form"}
            onClick={() => {
              if (view === "json") {
                switchToForm();
              } else {
                setView("form");
              }
            }}
          >
            Form
          </button>
          <button
            type="button"
            role="tab"
            className={view === "json" ? "view-tab active" : "view-tab"}
            aria-selected={view === "json"}
            onClick={switchToJson}
          >
            JSON
          </button>
        </div>
      </div>
      {jsonError ? <div className="alert alert-error">{jsonError}</div> : null}
      {view === "form" ? (
        <PresetConfigForm
          value={value}
          onChange={onChange}
          mode={mode}
          inherited={inherited}
        />
      ) : (
        <div className="form-field">
          <textarea
            className="config-json-textarea"
            rows={18}
            value={jsonText}
            onChange={(e) => {
              setJsonText(e.target.value);
              setJsonError(null);
            }}
            onBlur={() => {
              try {
                const parsed = JSON.parse(jsonText) as unknown;
                if (!isPlainObject(parsed)) {
                  setJsonError("Config must be a JSON object");
                  return;
                }
                onChange(cloneConfig(parsed));
                setJsonError(null);
              } catch (err) {
                setJsonError(err instanceof Error ? err.message : "Invalid JSON");
              }
            }}
          />
        </div>
      )}
    </div>
  );
}
