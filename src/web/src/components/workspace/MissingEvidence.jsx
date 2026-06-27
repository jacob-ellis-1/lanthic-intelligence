import { CircleHelp } from "lucide-react";

export default function MissingEvidence({
  items = [],
  totalCount,
  onViewAll
}) {
  const normalisedItems = Array.isArray(items)
    ? items.map(normaliseMissingEvidenceItem).filter(Boolean)
    : [];

  if (!normalisedItems.length) {
    return null;
  }

  const shownTotal = totalCount ?? normalisedItems.length;

  return (
    <section className="missing-evidence-card">
      <div className="missing-card-heading">
        <span className="small-section-icon">
          <CircleHelp size={17} />
        </span>

        <h2>Missing evidence</h2>
      </div>

      <ul>
        {normalisedItems.slice(0, 4).map((item, index) => (
          <li key={`${item}-${index}`}>{item}</li>
        ))}
      </ul>

      {shownTotal > 4 ? (
        <button className="inline-link-button" type="button" onClick={onViewAll}>
          View all gaps ({shownTotal}) →
        </button>
      ) : null}
    </section>
  );
}

function normaliseMissingEvidenceItem(item) {
  if (typeof item === "string") {
    return item.trim();
  }

  if (!item || typeof item !== "object") {
    return "";
  }

  return String(
    item.text ||
      item.question ||
      item.gap ||
      item.summary ||
      item.description ||
      JSON.stringify(item)
  ).trim();
}