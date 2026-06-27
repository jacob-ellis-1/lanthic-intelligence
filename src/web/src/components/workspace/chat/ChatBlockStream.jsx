import TextBlock from "./blocks/TextBlock.jsx";
import EvidenceBlock from "./blocks/EvidenceBlock.jsx";
import ReasoningPathBlock from "./blocks/ReasoningPathBlock.jsx";
import RiskAssessmentBlock from "./blocks/RiskAssessmentBlock.jsx";
import ForecastBlock from "./blocks/ForecastBlock.jsx";
import MissingEvidenceBlock from "./blocks/MissingEvidenceBlock.jsx";
import TableBlock from "./blocks/TableBlock.jsx";
import EmptyBlock from "./blocks/EmptyBlock.jsx";

export default function ChatBlockStream({
  blocks = [],
  onViewEvidence,
  onViewMissingEvidence,
  onViewReasoningPath
}) {
  if (!blocks.length) {
    return <EmptyBlock title="No analysis blocks" />;
  }

  return (
    <div className="chat-block-stream">
      {blocks.map((block) => {
        switch (block.type) {
          case "text":
          case "brief":
          case "warning":
          case "recommendation":
          case "metric":
            return <TextBlock key={block.id} block={block} />;

          case "evidence":
          case "evidence_support":
            return (
              <EvidenceBlock
                key={block.id}
                block={block}
                onViewAll={onViewEvidence}
              />
            );

          case "reasoning_path":
          case "reasoning":
          case "kg_path":
            return (
              <ReasoningPathBlock
                key={block.id}
                block={block}
                onViewInKG={onViewReasoningPath}
              />
            );

          case "risk_assessment":
          case "risk":
            return <RiskAssessmentBlock key={block.id} block={block} />;

          case "forecast":
          case "forecast_check":
            return <ForecastBlock key={block.id} block={block} />;

          case "missing_evidence":
          case "gaps":
            return (
              <MissingEvidenceBlock
                key={block.id}
                block={block}
                onViewAll={onViewMissingEvidence}
              />
            );

          case "table":
            return <TableBlock key={block.id} block={block} />;

          default:
            return <TextBlock key={block.id} block={block} />;
        }
      })}
    </div>
  );
}