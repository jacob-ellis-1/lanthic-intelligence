import { FileText } from "lucide-react";

const DEFAULT_VISIBLE = 3;

export default function EvidenceSupport({
  evidence = [],
  totalCount,
  onViewAll
}) {
  if (!evidence.length) {
    return null;
  }

  const visibleEvidence = evidence.slice(0, DEFAULT_VISIBLE);
  const shownTotal = totalCount ?? evidence.length;

  return (
    <section className="investigation-section evidence-support-section">
      <TimelineIcon icon={FileText} />

      <div className="investigation-section-content">
        <h2>Evidence support</h2>

        <div className="evidence-support-table">
          {visibleEvidence.map((item) => (
            <article className="evidence-support-row" key={item.id || item.text}>
              <div className="evidence-support-text">
                <span className="evidence-dot" />
                <p>{item.text}</p>
              </div>

              <div className="evidence-support-source">
                <strong>{item.source}</strong>
                <em>{item.date}</em>
              </div>
            </article>
          ))}
        </div>

        {shownTotal > visibleEvidence.length ? (
          <button
            className="inline-link-button"
            type="button"
            onClick={onViewAll}
          >
            Show all supporting evidence ({shownTotal}) →
          </button>
        ) : null}
      </div>
    </section>
  );
}

function TimelineIcon({ icon: Icon }) {
  return (
    <span className="investigation-timeline-icon">
      <Icon size={18} />
    </span>
  );
}