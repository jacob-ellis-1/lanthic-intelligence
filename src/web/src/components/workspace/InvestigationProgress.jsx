import { FileSearch, FileText, RotateCcw, Shield, Target } from "lucide-react";

const DEFAULT_STAGES = [
  {
    id: "search",
    label: "Searching relevant sources",
    status: "active",
    icon: FileSearch
  },
  {
    id: "support",
    label: "Checking support",
    status: "pending",
    icon: FileText
  },
  {
    id: "risk",
    label: "Assessing risk",
    status: "pending",
    icon: Shield
  },
  {
    id: "gaps",
    label: "Reviewing gaps",
    status: "pending",
    icon: Target
  }
];

export default function InvestigationProgress({
  stages = DEFAULT_STAGES,
  loopMessage,
  show = true
}) {
  if (!show) {
    return null;
  }

  return (
    <section className="investigation-progress" aria-label="Investigation progress">
      <div className="progress-accent-line" />

      <div className="progress-stage-list">
        {stages.map((stage, index) => {
          const Icon = stage.icon || iconForStage(stage.id);

          return (
            <div
              className={`progress-stage progress-stage-${stage.status || "pending"}`}
              key={stage.id || stage.label}
            >
              <span className="progress-stage-icon">
                <Icon size={18} />
              </span>

              <strong>{stage.label}</strong>

              {index < stages.length - 1 ? (
                <span className="progress-arrow">→</span>
              ) : null}
            </div>
          );
        })}
      </div>

      {loopMessage ? (
        <div className="agent-loop-message">
          <RotateCcw size={15} />
          <span>{loopMessage}</span>
        </div>
      ) : null}
    </section>
  );
}

function iconForStage(id) {
  if (id === "support") return FileText;
  if (id === "risk") return Shield;
  if (id === "gaps") return Target;
  return FileSearch;
}