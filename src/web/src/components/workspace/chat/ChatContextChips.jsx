export default function ChatContextChips({ items = [] }) {
  if (!items.length) {
    return null;
  }

  return (
    <section className="chat-context-used">
      <strong>KG context used</strong>

      <div>
        {items.map((item) => (
          <span key={item.id}>
            {item.graphKind === "edge" ? "Relation" : item.type || "Entity"}: {item.label}
          </span>
        ))}
      </div>
    </section>
  );
}