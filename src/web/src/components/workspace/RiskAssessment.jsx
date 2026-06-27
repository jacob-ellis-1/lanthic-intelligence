import { Shield } from "lucide-react";

export default function RiskAssessment({
  overallRisk = "High",
  factors = [],
  summary
}) {
  if (!factors.length && !summary) {
    return null;
  }

  return (
    <section className="investigation-section risk-assessment-section">
      <TimelineIcon icon={Shield} />

      <div className="investigation-section-content">
        <h2>Risk assessment</h2>

        <div className="risk-assessment-grid">
          <div className="risk-score-card">
            <div className="overall-risk-row">
              <span>Overall risk</span>
              <strong>{overallRisk}</strong>
            </div>

            <div className="risk-factor-bars">
              {factors.map((factor) => (
                <RiskFactorBar
                  key={factor.label}
                  label={factor.label}
                  value={factor.value}
                  tone={factor.tone}
                />
              ))}
            </div>
          </div>

          {summary ? (
            <div className="risk-summary-card">
              <p>{summary}</p>
            </div>
          ) : null}
        </div>
      </div>
    </section>
  );
}

function RiskFactorBar({ label, value = 0, tone = "medium" }) {
  const width = `${Math.max(0, Math.min(100, Number(value))) || 0}%`;

  return (
    <div className={`risk-factor-bar risk-factor-bar-${tone}`}>
      <span>{label}</span>

      <div>
        <i style={{ width }} />
      </div>
    </div>
  );
}

function TimelineIcon({ icon: Icon }) {
  return (
    <span className="investigation-timeline-icon">
      <Icon size={18} />
    </span>
  );
}