import { useMemo } from "react";
import ChatTurnCard from "./ChatTurnCard.jsx";
import { normaliseChatTurns } from "./chatBlocks.js";

export default function ChatTurnStream({
  turns = [],
  fallbackData,
  progress,
  isRunning = false,
  onViewEvidence,
  onViewMissingEvidence,
  onViewReasoningPath
}) {
  const normalisedTurns = useMemo(
    () => normaliseChatTurns(turns, fallbackData),
    [turns, fallbackData]
  );

  if (!normalisedTurns.length) {
    return (
      <section className="chat-thread-empty">
        <strong>No turns yet</strong>
        <p>Ask a question or add documents to begin the investigation thread.</p>
      </section>
    );
  }

  return (
    <section className="chat-turn-stream" aria-label="Investigation thread">
      <div className="chat-thread-heading">
        <div>
          <strong>Investigation thread</strong>
          <span>{normalisedTurns.length} turn{normalisedTurns.length === 1 ? "" : "s"}</span>
        </div>
      </div>

      {normalisedTurns.map((turn, index) => (
        <ChatTurnCard
          key={turn.id}
          turn={turn}
          index={index}
          isLatest={index === normalisedTurns.length - 1}
          isRunning={isRunning && index === normalisedTurns.length - 1}
          progress={progress}
          onViewEvidence={onViewEvidence}
          onViewMissingEvidence={onViewMissingEvidence}
          onViewReasoningPath={onViewReasoningPath}
        />
      ))}
    </section>
  );
}