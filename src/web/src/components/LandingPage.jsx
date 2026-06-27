import { ArrowRight } from "lucide-react";
import { createDemoSession } from "../api.js";
import markUrl from "../assets/lanthic-mark.svg";

const intelligenceItems = [
  {
    label: "Maintained",
    text: "A broad source base is organised before the analyst begins."
  },
  {
    label: "Connected",
    text: "Evidence, context, prior signals, and open questions remain linked."
  },
  {
    label: "Reviewable",
    text: "The source trail stays visible when an investigation becomes a brief."
  },
  {
    label: "Focused",
    text: "The workspace is built for complex risk analysis, not generic search."
  }
];

const reasoningItems = [
  "Evidence remains attached to the conclusion it supports.",
  "Uncertainty is surfaced instead of hidden inside confident prose.",
  "Analysts can inspect the path before using the answer."
];

const workspacePanels = [
  "Question",
  "Source trail",
  "Reasoning",
  "Forecast",
  "Brief"
];

const loopStages = [
  "Question",
  "Source base",
  "Reasoning",
  "Forecast",
  "Brief"
];

export default function LandingPage({ onEnter, onDemo }) {
  async function handleDemo() {
    const session = await createDemoSession();
    onDemo(session);
  }

  return (
    <main className="landing-page">
      <header className="site-nav">
        <button className="site-wordmark" onClick={() => (window.location.hash = "/")}>
          <img src={markUrl} alt="" />
          <span>Lanthic Intelligence</span>
        </button>

        <nav className="site-links" aria-label="Primary navigation">
          <a href="#intelligence">Intelligence base</a>
          <a href="#reasoning">Reasoning</a>
          <a href="#signals">Forecasting</a>
          <a href="#workspace">Workspace</a>
          <button className="nav-button" onClick={onEnter}>
            Sign in
          </button>
        </nav>
      </header>

      <section className="landing-hero">
        <div className="hero-grid" />

        <div className="hero-copy">
          <p className="section-kicker">Lanthic Intelligence</p>

          <h1>Grounded intelligence for complex supply-chain risk.</h1>

          <p className="hero-lede">
            Lanthic gives analysts a large maintained source base, reviewable
            reasoning, forward-looking risk signals, and a focused workspace for
            turning difficult questions into defensible briefs.
          </p>

          <div className="hero-actions">
            <button className="primary-button" onClick={handleDemo}>
              Open demo workspace
              <ArrowRight size={18} />
            </button>

            <button className="secondary-button" onClick={onEnter}>
              Sign in
            </button>
          </div>
        </div>

        <div className="hero-visual" aria-hidden="true">
          <div className="hero-mineral" />
          <div className="hero-graphite" />

          <div className="workspace-card hero-workspace-card">
            <div className="workspace-card-header">
              <img src={markUrl} alt="" />
              <span>Investigation workspace</span>
            </div>

            <div className="workspace-question">
              <span>Active question</span>
              <strong>What changed, why does it matter, and what supports it?</strong>
            </div>

            <div className="workspace-map">
              <div>
                <span>Source base</span>
                <strong>Context already in place</strong>
              </div>
              <div>
                <span>Reasoning</span>
                <strong>Evidence path visible</strong>
              </div>
              <div>
                <span>Brief</span>
                <strong>Ready for review</strong>
              </div>
            </div>

            <div className="workspace-evidence">
              <span />
              <span />
              <span />
              <span />
            </div>

            <p>
              A clear source trail keeps the answer, reasoning, and unresolved
              gaps in view.
            </p>
          </div>
        </div>
      </section>

      <section id="intelligence" className="landing-section intelligence-section">
        <div className="section-copy">
          <p className="section-kicker">Large intelligence base</p>
          <h2>Start from an organised source base, not an empty workspace.</h2>
          <p>
            Lanthic gives analysts access to a broad maintained body of source
            material for complex supply-chain risk. Evidence, market context,
            prior signals, and open questions are organised into a working
            intelligence layer so each investigation begins with context already
            in place.
          </p>
        </div>

        <div className="intelligence-layout">
          <SourceArchiveVisual />

          <div className="principle-grid compact-principle-grid">
            {intelligenceItems.map((item) => (
              <article className="principle-card" key={item.label}>
                <span>{item.label}</span>
                <p>{item.text}</p>
              </article>
            ))}
          </div>
        </div>
      </section>

      <section id="reasoning" className="landing-section reasoning-section">
        <div className="split-section">
          <div className="section-copy">
            <p className="section-kicker">Grounded reasoning</p>
            <h2>Reasoning you can inspect.</h2>
            <p>
              Lanthic keeps conclusions connected to the evidence and assumptions
              behind them. Analysts can review what supports an answer, where
              confidence is limited, and which gaps still need attention.
            </p>

            <ul className="section-list">
              {reasoningItems.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          </div>

          <ReasoningPathVisual />
        </div>
      </section>

      <section id="signals" className="landing-section signal-section">
        <div className="split-section split-section-reverse">
          <ForecastVisual />

          <div className="section-copy">
            <p className="section-kicker">Forward-looking risk signals</p>
            <h2>Move beyond retrospective summaries.</h2>
            <p>
              Lanthic helps analysts surface directional risk signals, changing
              exposure, and early warning patterns that can be reviewed alongside
              the underlying evidence.
            </p>

            <div className="signal-copy-grid">
              <div>
                <strong>Signals</strong>
                <span>Track change as it emerges.</span>
              </div>
              <div>
                <strong>Exposure</strong>
                <span>Keep likely impact in view.</span>
              </div>
              <div>
                <strong>Confidence</strong>
                <span>Separate strong signals from weak ones.</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section id="workspace" className="landing-section workspace-section">
        <div className="section-copy narrow">
          <p className="section-kicker">Analyst workspace</p>
          <h2>A workspace for turning investigation into briefing.</h2>
          <p>
            Lanthic keeps the analyst’s loop in one focused surface: ask the
            question, inspect the source trail, review the reasoning, refine the
            view, and leave with a brief that still carries its evidence.
          </p>
        </div>

        <AnalystWorkspaceVisual />
      </section>

      <section className="landing-section loop-section">
        <div className="section-copy narrow">
          <p className="section-kicker">Investigation loop</p>
          <h2>From question to defensible brief.</h2>
          <p>
            Each session keeps the question, source material, reasoning, forecast,
            review state, and final brief connected. The result is faster work
            without losing accountability.
          </p>
        </div>

        <InvestigationLoopVisual />
      </section>

      <section className="landing-final">
        <img src={markUrl} alt="" />
        <p className="section-kicker">Demo workspace</p>
        <h2>Open a grounded intelligence session.</h2>
        <p>
          Explore how Lanthic keeps source material, reasoning, evidence,
          forecasting, and uncertainty together in one analyst workspace.
        </p>

        <div className="hero-actions centered">
          <button className="primary-button" onClick={handleDemo}>
            Open demo workspace
            <ArrowRight size={18} />
          </button>

          <button className="secondary-button" onClick={onEnter}>
            Sign in
          </button>
        </div>
      </section>

      <footer className="landing-footer">
        <div className="footer-brand">
          <img src={markUrl} alt="" />
          <div>
            <strong>Lanthic Intelligence</strong>
            <p>Grounded intelligence for complex supply-chain risk.</p>
          </div>
        </div>

        <div className="footer-columns">
          <FooterColumn
            title="Product"
            links={["Intelligence base", "Grounded reasoning", "Analyst workspace"]}
          />
          <FooterColumn
            title="Principles"
            links={[
              "Evidence visible",
              "Reasoning reviewable",
              "Signals forward-looking",
              "Uncertainty preserved"
            ]}
          />
          <FooterColumn
            title="Access"
            links={["Demo workspace", "Sign in"]}
            onDemo={handleDemo}
            onEnter={onEnter}
          />
        </div>
      </footer>
    </main>
  );
}

function SourceArchiveVisual() {
  const frame = { width: 1000, height: 620 };
  const focus = { x: 285, y: 160, w: 430, h: 290 };

  const pct = (value, total) => `${(value / total) * 100}%`;

  const docs = [
    { id: "a", x: 110, y: 115, w: 125, h: 82, opacity: 0.66, blur: "0.25px", shadow: 0.07 },
    { id: "b", x: 290, y: 30, w: 220, h: 105, opacity: 0.96, blur: "0px", shadow: 0.13 },
    { id: "c", x: 620, y: 105, w: 150, h: 85, opacity: 0.74, blur: "0.16px", shadow: 0.08 },
    { id: "d", x: 780, y: 60, w: 180, h: 92, opacity: 0.92, blur: "0px", shadow: 0.13 },
    { id: "e", x: 65, y: 225, w: 180, h: 110, opacity: 0.96, blur: "0px", shadow: 0.15 },
    { id: "f", x: 780, y: 245, w: 155, h: 100, opacity: 0.92, blur: "0px", shadow: 0.14 },
    { id: "g", x: 125, y: 445, w: 125, h: 90, opacity: 0.64, blur: "0.24px", shadow: 0.08 },
    { id: "h", x: 330, y: 530, w: 210, h: 95, opacity: 0.9, blur: "0px", shadow: 0.12 },
    { id: "i", x: 645, y: 450, w: 135, h: 90, opacity: 0.58, blur: "0.3px", shadow: 0.07 }
  ];

  const solidLinks = [
    { x1: 235, y1: 136, x2: focus.x, y2: focus.y + 78 },
    { x1: 400, y1: 135, x2: focus.x + 130, y2: focus.y },
    { x1: 695, y1: 190, x2: focus.x + 310, y2: focus.y },
    { x1: 780, y1: 101, x2: focus.x + focus.w, y2: focus.y + 76 },
    { x1: 245, y1: 286, x2: focus.x, y2: focus.y + 150 },
    { x1: 780, y1: 302, x2: focus.x + focus.w, y2: focus.y + 156 },
    { x1: 188, y1: 445, x2: focus.x + 125, y2: focus.y + focus.h },
    { x1: 435, y1: 530, x2: focus.x + 220, y2: focus.y + focus.h },
    { x1: 712, y1: 450, x2: focus.x + 340, y2: focus.y + focus.h }
  ];

  const dashedLinks = [
    { x1: 235, y1: 132, x2: 290, y2: 78 },
    { x1: 510, y1: 72, x2: 780, y2: 92 },
    { x1: 870, y1: 152, x2: 858, y2: 245 },
    { x1: 110, y1: 156, x2: 65, y2: 258 },
    { x1: 245, y1: 330, x2: 188, y2: 445 },
    { x1: 540, y1: 570, x2: 645, y2: 495 }
  ];

  return (
    <div className="source-archive" aria-hidden="true">
      <div className="archive-floor" />

      <svg
        className="archive-connection-layer"
        viewBox={`0 0 ${frame.width} ${frame.height}`}
        preserveAspectRatio="none"
        role="presentation"
      >
        {dashedLinks.map((line, index) => (
          <g key={`dash-${index}`}>
            <line
              className="archive-dashed-link"
              x1={line.x1}
              y1={line.y1}
              x2={line.x2}
              y2={line.y2}
            />
            <circle className="archive-dashed-dot" cx={line.x1} cy={line.y1} r="3.2" />
            <circle className="archive-dashed-dot" cx={line.x2} cy={line.y2} r="3.2" />
          </g>
        ))}

        {solidLinks.map((line, index) => (
          <g key={`solid-${index}`}>
            <line
              className="archive-solid-link"
              x1={line.x1}
              y1={line.y1}
              x2={line.x2}
              y2={line.y2}
            />
            <circle className="archive-solid-dot" cx={line.x1} cy={line.y1} r="4" />
          </g>
        ))}
      </svg>

      <div className="archive-doc-layer">
        {docs.map((doc) => (
          <span
            key={doc.id}
            className={`archive-node archive-node-${doc.id}`}
            style={{
              left: pct(doc.x, frame.width),
              top: pct(doc.y, frame.height),
              width: pct(doc.w, frame.width),
              height: pct(doc.h, frame.height),
              "--node-opacity": doc.opacity,
              "--node-blur": doc.blur,
              "--node-shadow": doc.shadow
            }}
          >
            <span className="archive-doc">
              <b />
              <i />
              <i />
              <i />
            </span>
          </span>
        ))}
      </div>

      <div
        className="archive-focus-card"
        style={{
          left: pct(focus.x, frame.width),
          top: pct(focus.y, frame.height),
          width: pct(focus.w, frame.width)
        }}
      >
        <span>Working intelligence layer</span>
        <strong>Source material organised around the question.</strong>

        <div className="archive-focus-lines">
          <i />
          <i />
          <i />
        </div>

        <div className="archive-focus-tags">
          <em>Evidence</em>
          <em>Context</em>
          <em>Gaps</em>
        </div>
      </div>
    </div>
  );
}

function ReasoningPathVisual() {
  const frame = { width: 800, height: 620 };
  const pct = (value, total) => `${(value / total) * 100}%`;

  const topSource = { x: 78, y: 92, w: 228, h: 112 };
  const bottomSource = { x: 198, y: 386, w: 206, h: 108 };
  const conclusion = { x: 690, y: 205, w: 330, h: 182 };

  const nodeY = 318;
  const nodes = [
    { key: "supported", x: 446, y: nodeY, label: "Supported", placement: "above" },
    { key: "review", x: 536, y: nodeY, label: "Needs review", placement: "below" },
    { key: "gap", x: 626, y: nodeY, label: "Open gap", placement: "above" }
  ];

  const sourceLinks = [
    { x1: topSource.x + topSource.w, y1: topSource.y + topSource.h * 0.78, x2: nodes[0].x, y2: nodes[0].y },
    { x1: bottomSource.x + bottomSource.w, y1: bottomSource.y + bottomSource.h * 0.42, x2: nodes[0].x, y2: nodes[0].y }
  ];

  const spineSegments = [
    { x1: nodes[0].x, y1: nodeY, x2: nodes[1].x, y2: nodeY, className: "reasoning-spine-strong" },
    { x1: nodes[1].x, y1: nodeY, x2: nodes[2].x, y2: nodeY, className: "reasoning-spine-mid" },
    { x1: nodes[2].x, y1: nodeY, x2: conclusion.x, y2: nodeY, className: "reasoning-spine-light" }
  ];

  return (
    <div className="reasoning-path-visual" aria-hidden="true">
      <div className="reasoning-grid-panel" />

      <svg
        className="reasoning-connection-layer"
        viewBox={`0 0 ${frame.width} ${frame.height}`}
        preserveAspectRatio="none"
        role="presentation"
      >
        {sourceLinks.map((line, index) => (
          <g key={`source-link-${index}`}>
            <line
              className="reasoning-source-link"
              x1={line.x1}
              y1={line.y1}
              x2={line.x2}
              y2={line.y2}
            />
            <circle className="reasoning-source-dot" cx={line.x1} cy={line.y1} r="4" />
            <circle className="reasoning-source-dot" cx={line.x2} cy={line.y2} r="4" />
          </g>
        ))}

        {spineSegments.map((segment, index) => (
          <line
            key={`spine-${index}`}
            className={`reasoning-spine-segment ${segment.className}`}
            x1={segment.x1}
            y1={segment.y1}
            x2={segment.x2}
            y2={segment.y2}
          />
        ))}

        {nodes.map((node) => (
          <circle
            key={`node-${node.key}`}
            className="reasoning-node-circle"
            cx={node.x}
            cy={node.y}
            r="9"
          />
        ))}
      </svg>

      <div
        className="reasoning-source-card reasoning-source-top"
        style={{
          left: pct(topSource.x, frame.width),
          top: pct(topSource.y, frame.height),
          width: pct(topSource.w, frame.width),
          minHeight: pct(topSource.h, frame.height)
        }}
      >
        <span className="reasoning-source-kicker">Source</span>
        <div className="reasoning-source-lines">
          <i />
          <i />
          <i />
        </div>
      </div>

      <div
        className="reasoning-source-card reasoning-source-bottom"
        style={{
          left: pct(bottomSource.x, frame.width),
          top: pct(bottomSource.y, frame.height),
          width: pct(bottomSource.w, frame.width),
          minHeight: pct(bottomSource.h, frame.height)
        }}
      >
        <span className="reasoning-source-kicker">Source</span>
        <div className="reasoning-source-lines">
          <i />
          <i />
          <i />
        </div>
      </div>

      {nodes.map((node) => (
        <span
          key={`chip-${node.key}`}
          className={`reasoning-node-chip reasoning-node-chip-${node.placement}`}
          style={{
            left: pct(node.x, frame.width),
            top: pct(node.y, frame.height)
          }}
        >
          {node.label}
        </span>
      ))}

      <div
        className="reasoning-conclusion-card"
        style={{
          left: pct(conclusion.x, frame.width),
          top: pct(conclusion.y, frame.height),
          width: pct(conclusion.w, frame.width)
        }}
      >
        <span className="reasoning-conclusion-kicker">Reviewable conclusion</span>
        <strong>Evidence and uncertainty stay attached.</strong>

        <div className="reasoning-conclusion-lines">
          <i />
          <i />
          <i />
        </div>
      </div>
    </div>
  );
}

function ForecastVisual() {
  const frame = { width: 1000, height: 760 };
  const pct = (value, total) => `${(value / total) * 100}%`;

  const panel = { x: 74, y: 108, w: 760, h: 520 };
  const chart = { x: 118, y: 190, w: 660, h: 300 };
  const topSource = { x: -90, y: 180, w: 182, h: 70 };
  const bottomSource = { x: 800, y: 420, w: 182, h: 66 };

  const forecastStart = {
    x: chart.x + 438,
    y: chart.y + 60
  };

  const sourceLinks = [
  {
    x1: topSource.x + topSource.w,
    y1: topSource.y + topSource.h * 0.55,
    x2: chart.x + 438,
    y2: chart.y + 60
  },
  {
    x1: bottomSource.x + bottomSource.w * 0.5,
    y1: bottomSource.y,
    x2: chart.x + 610,
    y2: chart.y + 66
  }
];

  const gridX = Array.from({ length: 13 }, (_, i) => i * (chart.w / 12));
  const gridY = Array.from({ length: 7 }, (_, i) => i * (chart.h / 6));

  return (
    <div className="forecast-visual" aria-hidden="true">
      <div className="forecast-graphite-accent" />

      <svg
        className="forecast-overlay"
        viewBox={`0 0 ${frame.width} ${frame.height}`}
        preserveAspectRatio="none"
        role="presentation"
      >
        {sourceLinks.map((line, index) => (
          <g key={`source-${index}`}>
            <line
              className="forecast-evidence-link"
              x1={line.x1}
              y1={line.y1}
              x2={line.x2}
              y2={line.y2}
            />
            <circle className="forecast-evidence-dot" cx={line.x1} cy={line.y1} r="3.5" />
            <circle className="forecast-evidence-dot" cx={line.x2} cy={line.y2} r="3.5" />
          </g>
        ))}
      </svg>

      <div
        className="forecast-source-card forecast-source-top"
        style={{
          left: pct(topSource.x, frame.width),
          top: pct(topSource.y, frame.height),
          width: pct(topSource.w, frame.width),
          minHeight: pct(topSource.h, frame.height)
        }}
      >
        <div className="forecast-source-lines">
          <i />
          <i />
          <i />
        </div>
      </div>

      <div
        className="forecast-source-card forecast-source-bottom"
        style={{
          left: pct(bottomSource.x, frame.width),
          top: pct(bottomSource.y, frame.height),
          width: pct(bottomSource.w, frame.width),
          minHeight: pct(bottomSource.h, frame.height)
        }}
      >
        <div className="forecast-source-lines">
          <i />
          <i />
          <i />
        </div>
      </div>

      <div
        className="forecast-panel"
        style={{
          left: pct(panel.x, frame.width),
          top: pct(panel.y, frame.height),
          width: pct(panel.w, frame.width),
          minHeight: pct(panel.h, frame.height)
        }}
      >
        <div className="forecast-header">
          <span className="forecast-kicker">Forward signal</span>
          <strong className="forecast-title">Reviewed with evidence</strong>
        </div>

        <div
          className="forecast-chart-shell"
          style={{
            left: pct(chart.x - panel.x, panel.w),
            top: pct(chart.y - panel.y, panel.h),
            width: pct(chart.w, panel.w),
            height: pct(chart.h, panel.h)
          }}
        >
          <svg
            className="forecast-chart"
            viewBox={`0 0 ${chart.w} ${chart.h}`}
            preserveAspectRatio="none"
            role="presentation"
          >
            {gridY.map((y, index) => (
              <line
                key={`gy-${index}`}
                className="forecast-grid-line"
                x1="0"
                y1={y}
                x2={chart.w}
                y2={y}
              />
            ))}

            {gridX.map((x, index) => (
              <line
                key={`gx-${index}`}
                className="forecast-grid-line"
                x1={x}
                y1="0"
                x2={x}
                y2={chart.h}
              />
            ))}

            <polygon
              className="forecast-band"
              points="
                438,36
                528,28
                610,18
                660,10
                660,58
                610,66
                528,76
                438,84
              "
            />

            <line className="forecast-divider" x1="438" y1="24" x2="438" y2="286" />

            <path
              className="forecast-history-line"
              d="
                M 28 212
                C 98 194, 154 184, 210 164
                C 254 148, 298 112, 362 98
                C 394 91, 424 82, 438 60
              "
            />

            <path
              className="forecast-forecast-line"
              d="
                M 438 60
                C 510 54, 578 44, 660 34
              "
            />

            <circle className="forecast-now-dot" cx="438" cy="60" r="7" />
          </svg>
        </div>

        <div className="forecast-metric-row">
          <div className="forecast-metric-card">
            <span>Signal</span>
            <strong>Rising</strong>
          </div>

          <div className="forecast-metric-card">
            <span>Exposure</span>
            <strong>Elevated</strong>
          </div>

          <div className="forecast-metric-card">
            <span>Confidence</span>
            <strong>Review</strong>
          </div>
        </div>
      </div>
    </div>
  );
}

function AnalystWorkspaceVisual() {
  const leftItems = ["active", "", "", ""];
  const drawerItems = ["primary", "primary", "muted"];

  return (
    <div className="analyst-workspace-visual" aria-hidden="true">
      <div className="workspace-topbar">
        <img className="workspace-brand-mark" src={markUrl} alt="" />

        <div className="workspace-top-actions">
          <span />
          <span />
          <span />
          <span className="workspace-user-chip">
            <i />
            <b />
          </span>
        </div>
      </div>

      <div className="workspace-shell">
        <aside className="workspace-left-rail">
          <span className="workspace-side-title" />
          <span className="workspace-control-button" />

          <div className="workspace-investigation-list">
            {leftItems.map((state, index) => (
              <div
                key={index}
                className={`workspace-investigation-card ${
                  state === "active" ? "is-active" : ""
                }`}
              >
                <span className="workspace-doc-icon" />
                <div className="workspace-card-lines">
                  <i />
                  <i />
                  <i />
                </div>
              </div>
            ))}
          </div>
        </aside>

        <main className="workspace-main-area">
          <div className="workspace-accent-shape" />

          <div className="workspace-tabs">
            <span className="is-active" />
            <span />
          </div>

          <section className="workspace-thread-card workspace-question-card">
            <span className="workspace-small-kicker" />
            <div className="workspace-question-lines">
              <i />
              <i />
            </div>
          </section>

          <section className="workspace-thread-card workspace-brief-card">
            <span className="workspace-green-kicker" />
            <div className="workspace-brief-lines">
              <i />
              <i />
              <i />
              <i />
              <i />
            </div>
          </section>

          <section className="workspace-thread-card workspace-evidence-card">
            <span className="workspace-evidence-badge" />
            <span className="workspace-green-kicker compact" />

            <div className="workspace-evidence-table">
              <div>
                <b />
                <i />
                <i />
              </div>
              <div>
                <b />
                <i />
                <i />
              </div>
              <div>
                <b />
                <i />
                <i />
              </div>
            </div>

            <div className="workspace-reasoning-strip">
              <span />
              <i />
              <span />
              <i />
              <span />
              <i />
              <span className="is-final" />
            </div>
          </section>
        </main>

        <aside className="workspace-right-drawer">
          <span className="workspace-side-title" />

          <div className="workspace-drawer-tabs">
            <span className="is-active" />
            <span />
            <span />
          </div>

          <div className="workspace-drawer-list">
            {drawerItems.map((state, index) => (
              <div key={index} className="workspace-drawer-card">
                <span className={`workspace-drawer-thumb ${state}`} />

                <div className="workspace-drawer-content">
                  <b />
                  <i />
                  <i />
                  <i />
                  <em />
                </div>

              </div>
            ))}
          </div>
        </aside>
      </div>
    </div>
  );
}

function InvestigationLoopVisual() {
  const steps = [
    { number: "01", label: "Question" },
    { number: "02", label: "Source base" },
    { number: "03", label: "Reasoning" },
    { number: "04", label: "Forecast" },
    { number: "05", label: "Brief", featured: true }
  ];

  return (
    <div className="investigation-loop-visual" aria-hidden="true">
      <div className="loop-rail">
        <span className="loop-rail-line" />
        {steps.map((step, index) => (
          <span
            key={step.number}
            className={`loop-rail-node ${step.featured ? "is-featured" : ""}`}
            style={{ left: `${10 + index * 20}%` }}
          />
        ))}
      </div>

      <div className="loop-cards">
        {steps.map((step) => (
          <div
            key={step.number}
            className={`loop-card ${step.featured ? "is-featured" : ""}`}
          >
            <span className="loop-card-number">{step.number}</span>
            <strong className="loop-card-label">{step.label}</strong>
          </div>
        ))}
      </div>
    </div>
  );
}

function FooterColumn({ title, links, onDemo, onEnter }) {
  function handleClick(label) {
    if (label === "Demo workspace" && onDemo) {
      onDemo();
    }

    if (label === "Sign in" && onEnter) {
      onEnter();
    }
  }

  return (
    <div className="footer-column">
      <strong>{title}</strong>

      {links.map((link) => (
        <button key={link} onClick={() => handleClick(link)}>
          {link}
        </button>
      ))}
    </div>
  );
}