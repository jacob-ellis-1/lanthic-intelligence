import { GitBranch } from "lucide-react";

export default function ReasoningPath({ nodes = [] }) {
  if (!nodes.length) {
    return null;
  }

  return (
    <section className="investigation-section reasoning-path-section">
      <TimelineIcon icon={GitBranch} />

      <div className="investigation-section-content">
        <h2>Reasoning path</h2>

        <div className="reasoning-path">
          {nodes.map((node, index) => (
            <div className="reasoning-path-item" key={`${node.label}-${index}`}>
              <article>
                <span>{node.label}</span>
              </article>

              {index < nodes.length - 1 ? (
                <span className="reasoning-path-arrow">→</span>
              ) : null}
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

function TimelineIcon({ icon: Icon }) {
  return (
    <span className="investigation-timeline-icon">
      <Icon size={18} />
    </span>
  );
}