import type { FieldDef } from "../../config/presetSchema";
import { formatInherited } from "../../lib/presetConfig";

type Props = {
  field: FieldDef;
  value: unknown;
  inherited?: unknown;
  overridden?: boolean;
  onChange: (value: unknown) => void;
  onClear?: () => void;
};

export function Field({ field, value, inherited, overridden, onChange, onClear }: Props) {
  const id = `field-${field.path.replace(/\./g, "-")}`;
  const showInherited = value === undefined && inherited !== undefined;
  const placeholder = showInherited ? formatInherited(inherited) : undefined;

  return (
    <div className={`form-field advanced-field${overridden ? " is-overridden" : ""}`}>
      <div className="advanced-field-label-row">
        <label htmlFor={id}>
          {field.label}
          {field.help ? (
            <span className="field-info" title={field.help} aria-label={field.help}>
              ⓘ
            </span>
          ) : null}
        </label>
        {onClear && overridden ? (
          <button type="button" className="btn-link" onClick={onClear}>
            Clear
          </button>
        ) : null}
      </div>
      {field.type === "boolean" && onClear ? (
        <select
          id={id}
          value={value === undefined ? "" : value ? "true" : "false"}
          onChange={(e) => {
            const next = e.target.value;
            if (next === "") {
              onChange(undefined);
            } else {
              onChange(next === "true");
            }
          }}
        >
          <option value="">
            (inherit{inherited !== undefined ? `: ${String(inherited)}` : ""})
          </option>
          <option value="true">true</option>
          <option value="false">false</option>
        </select>
      ) : null}
      {field.type === "boolean" && !onClear ? (
        <label className="checkbox-row" htmlFor={id}>
          <input
            id={id}
            type="checkbox"
            checked={Boolean(value)}
            onChange={(e) => onChange(e.target.checked)}
          />
        </label>
      ) : null}
      {field.type === "select" ? (
        <select
          id={id}
          value={value === undefined ? "" : String(value)}
          onChange={(e) => {
            const next = e.target.value;
            onChange(next === "" ? undefined : next);
          }}
        >
          {onClear ? <option value="">(inherit)</option> : null}
          {(field.options ?? []).map((opt) => (
            <option key={opt.value} value={opt.value}>
              {opt.label}
            </option>
          ))}
        </select>
      ) : null}
      {field.type === "number" ? (
        <input
          id={id}
          type="number"
          step={field.step ?? "any"}
          min={field.min}
          max={field.max}
          value={value === undefined || value === null ? "" : String(value)}
          placeholder={placeholder}
          onChange={(e) => {
            const raw = e.target.value;
            if (raw === "") {
              onChange(undefined);
              return;
            }
            const num = Number(raw);
            onChange(Number.isFinite(num) ? num : undefined);
          }}
        />
      ) : null}
      {field.type === "text" ? (
        <input
          id={id}
          type="text"
          value={value === undefined || value === null ? "" : String(value)}
          placeholder={placeholder}
          onChange={(e) => onChange(e.target.value === "" ? undefined : e.target.value)}
        />
      ) : null}
      {showInherited && field.type !== "boolean" ? (
        <span className="field-inherited-hint">Inherited: {placeholder}</span>
      ) : null}
    </div>
  );
}
