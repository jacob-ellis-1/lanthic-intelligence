import RiskAssessment from "../../RiskAssessment.jsx";

export default function RiskAssessmentBlock({ block }) {
  const data = block.data || {};

  return (
    <section className="chat-analysis-block risk-block">
      <RiskAssessment
        overallRisk={data.overallRisk || data.overall_risk}
        factors={data.factors || []}
        summary={data.summary || ""}
      />
    </section>
  );
}