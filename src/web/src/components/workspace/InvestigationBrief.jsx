import { Lightbulb } from "lucide-react";

export default function InvestigationBrief({
  title = "Investigation brief",
  answerLead = "Yes.",
  summary
}) {
  if (!summary) {
    return null;
  }

  return (
    <section className="investigation-section investigation-brief-section">
      <TimelineIcon icon={Lightbulb} />

      <div className="investigation-section-content">
        <h2>{title}</h2>

        <p className="brief-answer">
          {answerLead ? <strong>{answerLead}</strong> : null}
          {answerLead ? " " : null}
          {summary}
        </p>
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