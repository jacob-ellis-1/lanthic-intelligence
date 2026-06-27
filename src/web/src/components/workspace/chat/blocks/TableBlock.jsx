export default function TableBlock({ block }) {
  const data = block.data || {};
  const rows = Array.isArray(data.rows) ? data.rows : [];
  const columns = Array.isArray(data.columns) && data.columns.length
    ? data.columns
    : Object.keys(rows[0] || {});

  if (!rows.length || !columns.length) {
    return null;
  }

  return (
    <section className="chat-analysis-block table-block">
      <div className="chat-block-header">
        <span>{block.title || "Table"}</span>
      </div>

      <div className="chat-table-wrap">
        <table>
          <thead>
            <tr>
              {columns.map((column) => (
                <th key={column.key || column}>{column.label || column}</th>
              ))}
            </tr>
          </thead>

          <tbody>
            {rows.map((row, rowIndex) => (
              <tr key={row.id || rowIndex}>
                {columns.map((column) => {
                  const key = column.key || column;
                  return <td key={key}>{String(row[key] ?? "")}</td>;
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}