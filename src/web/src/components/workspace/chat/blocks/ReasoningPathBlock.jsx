import ReasoningPath from "../../ReasoningPath.jsx";

export default function ReasoningPathBlock({ block, onViewInKG }) {
  const data = block.data || {};
  const nodes = data.nodes || data.reasoningPath || data.reasoning_path || [];
  const path = extractReasoningPath(data, block);
  const summary = data.summary || data.text || data.description || "";

  return (
    <section className="chat-analysis-block reasoning-block">
      <div className="chat-reasoning-block-topline">
        <div>
          <span>{block.title || "Reasoning path"}</span>
          {summary ? <p>{summary}</p> : null}
        </div>

        {path.length ? (
          <button
            type="button"
            onClick={() => onViewInKG?.({
              ...block,
              data: {
                ...data,
                path
              }
            })}
          >
            View in KG
          </button>
        ) : null}
      </div>

      <ReasoningPath nodes={nodes} />
    </section>
  );
}

function extractReasoningPath(data = {}, block = {}) {
  const candidates =
    data.path ||
    data.graphPath ||
    data.graph_path ||
    data.kgPath ||
    data.kg_path ||
    data.items ||
    block.meta?.path ||
    block.meta?.graphPath ||
    block.meta?.graph_path ||
    block.meta?.graphItemIds ||
    block.meta?.graph_item_ids ||
    [];

  if (Array.isArray(candidates)) {
    return candidates;
  }

  return [];
}