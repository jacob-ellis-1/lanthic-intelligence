export default function TextBlock({ block }) {
  const data = block.data || {};
  const lead = data.lead || data.answerLead || data.answer_lead || data.title;
  const body = data.body || data.summary || data.text || data.content;

  return (
    <section className="chat-analysis-block text-block">
      <BlockHeader title={block.title || "Analysis"} />

      {lead ? <strong>{lead}</strong> : null}
      {body ? <p>{body}</p> : null}

      {!lead && !body ? (
        <p>No text content was returned for this block.</p>
      ) : null}
    </section>
  );
}

function BlockHeader({ title }) {
  return (
    <div className="chat-block-header">
      <span>{title}</span>
    </div>
  );
}