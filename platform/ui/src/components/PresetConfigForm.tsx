import {
  ALL_FIELD_PATHS,
  PRESET_SECTIONS,
  applySchemaDefaults,
  fieldVisible,
} from "../config/presetSchema";
import { deleteAt, getAt, isPlainObject, setAt, type JsonObject } from "../lib/presetConfig";
import { Field } from "./form/Field";
import { Section } from "./form/Section";

export type PresetConfigFormMode = "full" | "overrides";

type Props = {
  value: JsonObject;
  onChange: (next: JsonObject) => void;
  mode: PresetConfigFormMode;
  /** Preset config used as inherited placeholders in overrides mode. */
  inherited?: JsonObject | null;
};

export function PresetConfigForm({ value, onChange, mode, inherited = null }: Props) {
  const displayConfig = mode === "full" ? applySchemaDefaults(value) : value;
  const inheritedView = inherited ?? {};
  const effective =
    mode === "overrides" ? mergeForVisibility(inheritedView, value) : displayConfig;

  function updatePath(path: string, nextValue: unknown) {
    if (mode === "overrides" && nextValue === undefined) {
      onChange(deleteAt(value, path));
      return;
    }
    onChange(setAt(value, path, nextValue));
  }

  function clearPath(path: string) {
    onChange(deleteAt(value, path));
  }

  // In overrides mode, count how many of a section's schema fields currently
  // carry an explicit value. Drives the per-section "N overridden" badge and
  // lets the section reset stay disabled when there is nothing to clear.
  function countOverridden(fields: readonly { path: string }[]): number {
    return fields.reduce((total, field) => total + (getAt(value, field.path) !== undefined ? 1 : 0), 0);
  }

  const totalOverridden =
    mode === "overrides"
      ? PRESET_SECTIONS.reduce((total, section) => total + countOverridden(section.fields), 0)
      : 0;

  return (
    <div className="preset-config-form">
      {mode === "overrides" && totalOverridden > 0 ? (
        <div className="override-summary">
          <span>
            <strong>{totalOverridden}</strong> field{totalOverridden === 1 ? "" : "s"} overridden
            {" — everything else inherits from the preset."}
          </span>
          <button
            type="button"
            className="btn-link"
            onClick={() => {
              // Only schema-known paths are removed. Unknown keys the operator
              // added by hand in the JSON tab are intentionally preserved so a
              // reset can never silently discard configuration we cannot see.
              let next = value;
              for (const section of PRESET_SECTIONS) {
                for (const field of section.fields) {
                  next = deleteAt(next, field.path);
                }
              }
              onChange(next);
            }}
          >
            Reset all to preset
          </button>
        </div>
      ) : null}
      {PRESET_SECTIONS.map((section, index) => {
        const fields = section.fields.filter((field) => fieldVisible(field, effective));
        if (fields.length === 0) {
          return null;
        }
        const overriddenCount =
          mode === "overrides" ? countOverridden(section.fields) : 0;
        return (
          <Section
            key={section.id}
            id={section.id}
            title={section.title}
            help={section.help}
            defaultOpen={index < 3 || section.id.startsWith("optimizer")}
            badge={overriddenCount > 0 ? overriddenCount : undefined}
          >
            <div className="form-grid advanced-field-grid">
              {fields.map((field) => {
                const raw = getAt(value, field.path);
                const display =
                  mode === "full" && raw === undefined
                    ? getAt(displayConfig, field.path)
                    : raw;
                const inheritedValue = getAt(inheritedView, field.path);
                const overridden = mode === "overrides" && raw !== undefined;
                return (
                  <Field
                    key={field.path}
                    field={field}
                    value={mode === "full" ? display : raw}
                    inherited={mode === "overrides" ? inheritedValue : undefined}
                    overridden={overridden}
                    onChange={(next) => {
                      if (
                        mode === "full" &&
                        next === undefined &&
                        field.defaultValue !== undefined
                      ) {
                        updatePath(field.path, field.defaultValue);
                        return;
                      }
                      updatePath(field.path, next);
                    }}
                    onClear={mode === "overrides" ? () => clearPath(field.path) : undefined}
                  />
                );
              })}
            </div>
            {mode === "overrides" ? (
              <div className="advanced-section-actions">
                <button
                  type="button"
                  className="btn-link"
                  disabled={countOverridden(section.fields) === 0}
                  onClick={() => {
                    let next = value;
                    for (const field of section.fields) {
                      next = deleteAt(next, field.path);
                    }
                    onChange(next);
                  }}
                >
                  Clear section overrides
                </button>
              </div>
            ) : null}
          </Section>
        );
      })}
      <p className="advanced-form-footnote">
        {mode === "full"
          ? `${ALL_FIELD_PATHS.length} known fields. Unknown keys in the loaded JSON are preserved on save.`
          : "Only explicit overrides are saved. Empty fields inherit from the selected preset."}
      </p>
    </div>
  );
}

function mergeForVisibility(base: JsonObject, overlay: JsonObject): JsonObject {
  function walk(a: unknown, b: unknown): unknown {
    if (!isPlainObject(b)) {
      return b !== undefined ? b : a;
    }
    const aObj = isPlainObject(a) ? a : {};
    const out: JsonObject = { ...aObj };
    for (const [key, child] of Object.entries(b)) {
      out[key] = walk(aObj[key], child);
    }
    return out;
  }
  return walk(base, overlay) as JsonObject;
}
