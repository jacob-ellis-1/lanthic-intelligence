import MissingEvidence from "../../MissingEvidence.jsx";

export default function MissingEvidenceBlock({ block, onViewAll }) {
  const data = block.data || {};

  return (
    <section className="chat-analysis-block missing-evidence-block">
      <MissingEvidence
        items={data.items || []}
        totalCount={data.totalCount || data.total_count || data.items?.length || 0}
        onViewAll={onViewAll}
      />
    </section>
  );
}