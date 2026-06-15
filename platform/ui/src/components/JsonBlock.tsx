interface JsonBlockProps {
  value: unknown;
  maxHeight?: string;
}

export function JsonBlock({ value, maxHeight = "420px" }: JsonBlockProps) {
  const text =
    value === undefined || value === null
      ? "—"
      : JSON.stringify(value, null, 2);

  return (
    <pre className="json-block" style={{ maxHeight }}>
      {text}
    </pre>
  );
}
