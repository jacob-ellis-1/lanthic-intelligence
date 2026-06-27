import { ArrowRight } from "lucide-react";

export default function QuestionComposer({
  question,
  onQuestionChange,
  onRun,
  isRunning = false,
  placeholder = "Ask a focused question about this topic..."
}) {
  function handleSubmit(event) {
    event.preventDefault();

    if (!question?.trim() || isRunning) {
      return;
    }

    onRun?.(question.trim());
  }

  return (
    <section className="workspace-composer">
      <form onSubmit={handleSubmit}>
        <label htmlFor="investigation-question">
          What do you need to know?
        </label>

        <textarea
          id="investigation-question"
          value={question}
          onChange={(event) => onQuestionChange?.(event.target.value)}
          placeholder={placeholder}
        />

        <div className="composer-action-row">
          <button
            className="run-investigation-button"
            type="submit"
            disabled={!question?.trim() || isRunning}
          >
            {isRunning ? "Investigating..." : "Run investigation"}
            <ArrowRight size={18} />
          </button>
        </div>
      </form>
    </section>
  );
}