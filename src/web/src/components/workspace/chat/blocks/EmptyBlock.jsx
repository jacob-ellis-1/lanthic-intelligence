export default function EmptyBlock({ title = "Empty block" }) {
  return (
    <section className="chat-analysis-block empty-block">
      <div className="chat-block-header">
        <span>{title}</span>
      </div>
      <p>No content is available for this analysis block.</p>
    </section>
  );
}