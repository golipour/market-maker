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

  return (
    <div className="preset-config-form">
      {PRESET_SECTIONS.map((section, index) => {
        const fields = section.fields.filter((field) => fieldVisible(field, effective));
        if (fields.length === 0) {
          return null;
        }
        return (
          <Section
            key={section.id}
            id={section.id}
            title={section.title}
            help={section.help}
            defaultOpen={index < 3 || section.id.startsWith("optimizer")}
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
