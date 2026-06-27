import InvestigationProgress from "../InvestigationProgress.jsx";
import ChatUserQuestion from "./ChatUserQuestion.jsx";
import ChatContextChips from "./ChatContextChips.jsx";
import ChatBlockStream from "./ChatBlockStream.jsx";

export default function ChatTurnCard({
  turn,
  index,
  isLatest = false,
  isRunning = false,
  progress,
  onViewEvidence,
  onViewMissingEvidence,
  onViewReasoningPath
}) {
  return (
    <article className={`chat-turn-card ${isLatest ? "latest" : ""}`}>
      <div className="chat-turn-topline">
        <span>Turn {index + 1}</span>
        {turn.createdAt ? <time>{formatTurnTime(turn.createdAt)}</time> : null}
      </div>

      <ChatUserQuestion question={turn.question} />

      <ChatContextChips items={turn.selectedGraphContext} />

      {isRunning ? (
        <InvestigationProgress
          stages={progress?.stages || []}
          loopMessage={progress?.loopMessage || ""}
          show
        />
      ) : null}

      <ChatBlockStream
        blocks={turn.blocks}
        onViewEvidence={onViewEvidence}
        onViewMissingEvidence={onViewMissingEvidence}
        onViewReasoningPath={onViewReasoningPath}
      />
    </article>
  );
}

function formatTurnTime(value) {
  try {
    return new Intl.DateTimeFormat(undefined, {
      hour: "2-digit",
      minute: "2-digit",
      day: "2-digit",
      month: "short"
    }).format(new Date(value));
  } catch {
    return value;
  }
}