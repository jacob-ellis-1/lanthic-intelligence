import EvidenceSupport from "../../EvidenceSupport.jsx";

export default function EvidenceBlock({ block, onViewAll }) {
  const data = block.data || {};

  return (
    <section className="chat-analysis-block evidence-block">
      <EvidenceSupport
        evidence={data.evidence || []}
        totalCount={data.totalCount || data.total_count || data.evidence?.length || 0}
        onViewAll={onViewAll}
      />
    </section>
  );
}