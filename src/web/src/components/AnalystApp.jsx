import { useEffect, useMemo, useRef, useState } from "react";
import { ChevronDown, Download, LogOut, Plus, UserCircle } from "lucide-react";
import markUrl from "../assets/lanthic-mark.svg";
import {
  addDocumentsToInvestigation,
  createInvestigation,
  downloadTextFile,
  exportInvestigation,
  getInvestigation,
  getWorkspaceState,
  listInvestigations,
  runInvestigationTurn,
  saveWorkspaceState,
  getInvestigationSubgraph,
} from "../api.js";
import { investigationMockData } from "./workspace/investigationMockData.js";
import InvestigationSidebar from "./workspace/InvestigationSidebar.jsx";
import QuestionComposer from "./workspace/QuestionComposer.jsx";
import InvestigationProgress from "./workspace/InvestigationProgress.jsx";
import InvestigationBrief from "./workspace/InvestigationBrief.jsx";
import EvidenceSupport from "./workspace/EvidenceSupport.jsx";
import ReasoningPath from "./workspace/ReasoningPath.jsx";
import RiskAssessment from "./workspace/RiskAssessment.jsx";
import ForecastCheck from "./workspace/ForecastCheck.jsx";
import MissingEvidence from "./workspace/MissingEvidence.jsx";
import EvidenceDrawer from "./workspace/EvidenceDrawer.jsx";
import WorkspaceTabs from "./workspace/WorkspaceTabs.jsx";
import KGViewer from "./workspace/KGViewer.jsx";
import ChatTurnStream from "./workspace/chat/ChatTurnStream.jsx";
import { appendTurnFromRunResult } from "./workspace/chat/chatBlocks.js";

const DEFAULT_INVESTIGATION_ID = "inv_demo_001";
const NEW_INVESTIGATION_TITLE = "New rare-earth investigation";

const DRAWER_WORKSPACE_BUCKETS = ["sources", "evidence", "assumptions", "graphItems"];

function drawerStateStorageKey(investigationId) {
  return `lanthic:drawerWorkspaceState:${investigationId || DEFAULT_INVESTIGATION_ID}`;
}

function saveDrawerStateFallback(investigationId, state) {
  if (typeof window === "undefined" || !window.localStorage) {
    return;
  }

  try {
    window.localStorage.setItem(
      drawerStateStorageKey(investigationId),
      JSON.stringify(state)
    );
  } catch {
    // Local drawer persistence is best-effort only.
  }
}

const DEFAULT_DRAWER_WORKSPACE_STATE = {
  activeTab: "Sources",
  bookmarks: {
    sources: [],
    evidence: [],
    assumptions: [],
    graphItems: []
  },
  pins: {
    sources: [],
    evidence: [],
    assumptions: [],
    graphItems: []
  },
  selectedDrawerItem: null
};

const BASE_PROGRESS_STAGES = [
  {
    id: "search",
    label: "Searching relevant sources",
    status: "complete"
  },
  {
    id: "support",
    label: "Checking support",
    status: "complete"
  },
  {
    id: "risk",
    label: "Assessing risk",
    status: "complete"
  },
  {
    id: "gaps",
    label: "Reviewing gaps",
    status: "complete"
  }
];

const WORKSPACE_TABS = [
  { id: "chat", label: "Chat" },
  { id: "kg", label: "Local KG" }
];

const PENDING_PROGRESS_STAGES = BASE_PROGRESS_STAGES.map((stage) => ({
  ...stage,
  status: "pending"
}));

const COMPLETE_PROGRESS_STAGES = BASE_PROGRESS_STAGES.map((stage) => ({
  ...stage,
  status: "complete"
}));

const RUNNING_PROGRESS_STAGES = [
  {
    id: "sarg",
    label: "SARG running",
    status: "active"
  },
  {
    id: "kg",
    label: "KG + evidence retrieval",
    status: "active"
  },
  {
    id: "reasoning",
    label: "Reasoning and gap check",
    status: "active"
  },
  {
    id: "synthesis",
    label: "Synthesis and tools",
    status: "active"
  }
];

const RUNNING_PROGRESS = {
  stages: RUNNING_PROGRESS_STAGES,
  loopMessage:
    "SARG is running. Exact reasoning steps and tool outputs will appear in the investigation thread when this turn completes."
};

const RUN_SEQUENCE = [
  {
    active: "search",
    loopMessage: ""
  },
  {
    active: "support",
    loopMessage: ""
  },
  {
    active: "risk",
    loopMessage: ""
  },
  {
    active: "gaps",
    loopMessage:
      "The investigation found an evidence gap and is checking whether more support is available."
  },
  {
    active: "search",
    loopMessage:
      "Gathering additional evidence before finalising the assessment."
  },
  {
    active: "support",
    loopMessage:
      "Rechecking support against the additional evidence."
  }
];

export default function AnalystApp({
  session,
  workspaceData,
  onBack,
  onSignOut
}) {
  const timers = useRef([]);
  const fileInputRef = useRef(null);
  const drawerSaveTimer = useRef(null);

  const initialInvestigationId =
    session?.investigationId ||
    workspaceData?.investigationId ||
    DEFAULT_INVESTIGATION_ID;

  const [selectedInvestigationId, setSelectedInvestigationId] = useState(initialInvestigationId);
  const [workspaceResult, setWorkspaceResult] = useState(workspaceData || null);
  const [investigations, setInvestigations] = useState([]);
  const [isRunning, setIsRunning] = useState(false);
  const [isLoadingInvestigation, setIsLoadingInvestigation] = useState(false);
  const [drawerOpen, setDrawerOpen] = useState(true);
  const [drawerWorkspaceState, setDrawerWorkspaceState] = useState(
    DEFAULT_DRAWER_WORKSPACE_STATE
  );
  const [notice, setNotice] = useState("");
  const [activeWorkspaceTab, setActiveWorkspaceTab] = useState("chat");
  const [localSubgraph, setLocalSubgraph] = useState(null);
  const [isLoadingSubgraph, setIsLoadingSubgraph] = useState(false);
  const [subgraphError, setSubgraphError] = useState("");
  const [selectedGraphItems, setSelectedGraphItems] = useState([]);
  const [graphContextItems, setGraphContextItems] = useState([]);
  const [kgFocusRequest, setKgFocusRequest] = useState(null);
  const [chatTurns, setChatTurns] = useState([]);
  const [currentInvestigation, setCurrentInvestigation] = useState(
    workspaceData?.currentInvestigation || null
  );

  const [question, setQuestion] = useState(workspaceData?.question || "");

  const [progress, setProgress] = useState({
    stages: workspaceData?.progress?.stages || PENDING_PROGRESS_STAGES,
    loopMessage: workspaceData?.progress?.loopMessage || ""
  });
  const data = useMemo(() => {
    const shell = buildBlankWorkspaceData(
      currentInvestigation ||
        workspaceData?.currentInvestigation ||
        investigationMockData.currentInvestigation,
      workspaceResult?.documents || []
    );

    const merged = workspaceResult
      ? mergeInvestigationData(shell, workspaceResult)
      : shell;

    const recentInvestigations = investigations.length
      ? investigations
      : merged.recentInvestigations;

    return {
      ...merged,
      currentInvestigation: {
        ...merged.currentInvestigation,
        ...(currentInvestigation || {})
      },
      recentInvestigations
    };
  }, [workspaceResult, investigations, currentInvestigation, workspaceData]);

  const hasRunResult = Boolean(
    workspaceResult?.answer ||
      workspaceResult?.analysisBlocks?.length ||
      workspaceResult?.selected_reasoning_paths?.length ||
      workspaceResult?.agent_step_count ||
      ["answered", "partial", "complete"].includes(
        String(workspaceResult?.status || "").toLowerCase()
      )
  );

  useEffect(() => {
    refreshInvestigations({
      preferredInvestigationId: initialInvestigationId,
      loadPreferred: true
    });

    return () => clearTimers();
  }, []);

  useEffect(() => {
    if (!workspaceData) {
      return;
    }

    const nextInvestigationId =
      workspaceData.investigationId ||
      workspaceData.currentInvestigation?.investigationId ||
      selectedInvestigationId ||
      DEFAULT_INVESTIGATION_ID;

    setSelectedInvestigationId(nextInvestigationId);
    setWorkspaceResult(workspaceData);

    if (workspaceData.currentInvestigation) {
      setCurrentInvestigation(normaliseInvestigationSummary(workspaceData.currentInvestigation));
    }

    if (workspaceData.question) {
      setQuestion(workspaceData.question);
    }

    setProgress({
      stages: workspaceData.progress?.stages || COMPLETE_PROGRESS_STAGES,
      loopMessage: workspaceData.progress?.loopMessage || ""
    });
  }, [workspaceData]);

  useEffect(() => {
    const investigationId = selectedInvestigationId || DEFAULT_INVESTIGATION_ID;
    let cancelled = false;

    async function loadSubgraph() {
      setIsLoadingSubgraph(true);
      setSubgraphError("");

      try {
        const graph = await getInvestigationSubgraph(investigationId);

        if (!cancelled) {
          setLocalSubgraph(graph);
          setSelectedGraphItems([]);
        }
      } catch (error) {
        if (!cancelled) {
          setLocalSubgraph(null);
          setSubgraphError(`Could not load local KG: ${error.message}`);
        }
      } finally {
        if (!cancelled) {
          setIsLoadingSubgraph(false);
        }
      }
    }

    loadSubgraph();

    return () => {
      cancelled = true;
    };
  }, [selectedInvestigationId]);

  function clearTimers() {
    timers.current.forEach((timer) => window.clearTimeout(timer));
    timers.current = [];
  }

  function startProgressAnimation() {
    clearTimers();
    setProgress(RUNNING_PROGRESS);
  }

  async function refreshInvestigations({
    preferredInvestigationId = selectedInvestigationId,
    loadPreferred = false
  } = {}) {
    try {
      const payload = await listInvestigations();
      const nextInvestigations = Array.isArray(payload?.investigations)
        ? payload.investigations.map(normaliseInvestigationSummary)
        : [];

      setInvestigations(nextInvestigations);

      const preferred =
        nextInvestigations.find((item) => item.investigationId === preferredInvestigationId) ||
        nextInvestigations[0];

      if (preferred) {
        setCurrentInvestigation(preferred);

        if (!selectedInvestigationId || loadPreferred) {
          setSelectedInvestigationId(preferred.investigationId);
        }

        if (loadPreferred) {
          await loadInvestigation(preferred.investigationId, {
            fromRefresh: true,
            fallbackSummary: preferred
          });
          
        }
      }
    } catch (error) {
      setNotice(`Could not load saved investigations: ${error.message}`);
    }
  }

  async function loadInvestigation(
    investigationId,
    {
      fromRefresh = false,
      fallbackSummary = null
    } = {}
  ) {
    if (!investigationId) {
      return;
    }

    if (!fromRefresh) {
      setIsLoadingInvestigation(true);
      setNotice("");
    }

    try {
      const investigation = await getInvestigation(investigationId);
      const summary = normaliseInvestigationSummary(investigation || fallbackSummary);

      setSelectedInvestigationId(summary.investigationId || investigationId);
      setCurrentInvestigation(summary);

      const turns = Array.isArray(investigation?.turns)
        ? investigation.turns
        : [];

      setChatTurns(turns);

      const latestTurn = turns.length ? turns[turns.length - 1] : null;
      const latestResult = latestTurn?.result;

      if (latestResult) {
        setWorkspaceResult({
          ...latestResult,
          currentInvestigation: summary,
          recentInvestigations: investigations
        });
        setQuestion(latestResult.question || latestTurn.question || summary.title || "");
        setProgress({
          stages: latestResult.progress?.stages || COMPLETE_PROGRESS_STAGES,
          loopMessage: latestResult.progress?.loopMessage || ""
        });
      } else {
        setWorkspaceResult(null);
        setQuestion("");
        setChatTurns([]);
        setProgress({
          stages: PENDING_PROGRESS_STAGES,
          loopMessage: ""
        });

        if (!fromRefresh) {
          setNotice(`Loaded investigation: ${summary.title}`);
        }
      }
    } catch (error) {
      setNotice(`Could not load investigation: ${error.message}`);
    } finally {
      setIsLoadingInvestigation(false);
    }
  }

  async function handleNewInvestigation() {
    if (isRunning || isLoadingInvestigation) {
      return;
    }

    setNotice("Creating investigation...");

    try {
      const created = await createInvestigation({
        title: NEW_INVESTIGATION_TITLE,
        question: NEW_INVESTIGATION_TITLE,
        runId: session?.run_id,
        corpusId: session?.corpus_id,
        branchId: session?.branch_id
      });

      const summary = normaliseInvestigationSummary(created);

      setSelectedInvestigationId(summary.investigationId);
      setCurrentInvestigation(summary);
      setQuestion("");
      setWorkspaceResult(null);
      setChatTurns([]);
      setProgress({
        stages: PENDING_PROGRESS_STAGES,
        loopMessage: ""
      });
      setDrawerOpen(true);
      setDrawerWorkspaceState(DEFAULT_DRAWER_WORKSPACE_STATE);
      setNotice("New investigation created. Enter a question and run it.");

      await refreshInvestigations({
        preferredInvestigationId: summary.investigationId,
        loadPreferred: false
      });
    } catch (error) {
      setNotice(`Could not create investigation: ${error.message}`);
    }
  }

  async function handleRun(nextQuestion) {
    const cleanQuestion = String(nextQuestion || "").trim();

    if (!cleanQuestion || isRunning || isLoadingInvestigation) {
      return;
    }

    const investigationId = selectedInvestigationId || DEFAULT_INVESTIGATION_ID;

    setQuestion(cleanQuestion);
    setIsRunning(true);
    setNotice("");
    startProgressAnimation();

    try {
      const result = await runInvestigationTurn(investigationId, {
        question: cleanQuestion,
        runId: session?.run_id,
        corpusId: session?.corpus_id,
        branchId: session?.branch_id,
        selectedGraphContext: graphContextItems
      });

      clearTimers();

      setChatTurns((current) =>
        appendTurnFromRunResult(current, result, {
          question: cleanQuestion,
          selectedGraphContext: graphContextItems
        })
      );

      const resultInvestigationId = result.investigationId || investigationId;
      const summary = normaliseInvestigationSummary(
        result.currentInvestigation || {
          investigationId: resultInvestigationId,
          title: cleanQuestion,
          chatName: cleanQuestion,
          status: "Answer ready",
          updatedAt: "Just now"
        }
      );

      setSelectedInvestigationId(resultInvestigationId);
      setCurrentInvestigation(summary);
      setWorkspaceResult({
        ...result,
        currentInvestigation: summary
      });
      setQuestion(result.question || cleanQuestion);
      setProgress({
        stages: result.progress?.stages || COMPLETE_PROGRESS_STAGES,
        loopMessage: result.progress?.loopMessage || ""
      });
      setDrawerOpen(true);
      setNotice("Investigation complete.");

      await refreshInvestigations({
        preferredInvestigationId: resultInvestigationId,
        loadPreferred: false
      });
      await refreshLocalSubgraph(resultInvestigationId);      
    } catch (error) {
      clearTimers();

      setProgress({
        stages: PENDING_PROGRESS_STAGES,
        loopMessage: ""
      });
      setNotice(`Investigation failed: ${error.message}`);
    } finally {
      setIsRunning(false);
    }
  }

  async function handleSelectInvestigation(item) {
    const investigationId = item?.investigationId;

    if (!investigationId) {
      setQuestion(item?.title || "");
      setNotice("Loaded prompt. Run it to create a new investigation turn.");
      return;
    }

    await loadInvestigation(investigationId);
    setSelectedGraphItems([]);
    setGraphContextItems([]);
  }

  async function handleExport() {
    const investigationId = selectedInvestigationId || DEFAULT_INVESTIGATION_ID;

    setNotice("Preparing export...");

    try {
      const markdown = await exportInvestigation(investigationId);
      const filename = `${safeFilename(currentInvestigation?.title || "lanthic-investigation")}.md`;

      downloadTextFile(filename, markdown);
      setNotice("Export downloaded.");
    } catch (error) {
      setNotice(`Export failed: ${error.message}`);
    }
  }

  function handleAddDocuments() {
    if (!selectedInvestigationId) {
      setNotice("Create or select an investigation before uploading documents.");
      return;
    }

    fileInputRef.current?.click();
  }

  async function handleFilesSelected(event) {
    const files = Array.from(event.target.files || []);
    event.target.value = "";

    if (!files.length) {
      return;
    }

    const investigationId = selectedInvestigationId || DEFAULT_INVESTIGATION_ID;

    setNotice(`Uploading ${files.length} document${files.length === 1 ? "" : "s"}...`);

    try {
      const result = await addDocumentsToInvestigation(investigationId, files);
      const uploadedDocuments = Array.isArray(result?.documents)
        ? result.documents
        : [];

      setWorkspaceResult((current) =>
        addUploadedDocumentsToWorkspace(
          current || buildBlankWorkspaceData(currentInvestigation, []),
          uploadedDocuments
        )
      );

      if (result?.investigation) {
        setCurrentInvestigation(normaliseInvestigationSummary(result.investigation));
      }

      setDrawerOpen(true);
      setNotice(result.message || "Documents attached. Re-run the investigation to include them.");

      await refreshInvestigations({
        preferredInvestigationId: investigationId,
        loadPreferred: false
      });
      await refreshLocalSubgraph(investigationId);
    } catch (error) {
      setNotice(`Upload failed: ${error.message}`);
    }
  }

  function handleDrawerWorkspaceStateChange(nextState) {
    const investigationId = selectedInvestigationId || DEFAULT_INVESTIGATION_ID;
    const normalised = normaliseDrawerWorkspaceState(nextState);

    setDrawerWorkspaceState(normalised);
    saveDrawerStateFallback(investigationId, normalised);

    if (drawerSaveTimer.current) {
      window.clearTimeout(drawerSaveTimer.current);
    }

    drawerSaveTimer.current = window.setTimeout(() => {
      saveWorkspaceState(investigationId, normalised).catch(() => {
        // Drawer state is already saved locally. Do not interrupt the demo path
        // for transient backend reload/fetch failures.
      });
    }, 450);
  }

  async function refreshLocalSubgraph(investigationId = selectedInvestigationId || DEFAULT_INVESTIGATION_ID) {
    try {
      const graph = await getInvestigationSubgraph(investigationId);
      setLocalSubgraph(graph);
    } catch (error) {
      setSubgraphError(`Could not refresh local KG: ${error.message}`);
    }
  }

  function handleGraphSelectionChange(items = []) {
    setSelectedGraphItems(items);

    if (!items.length) {
      return;
    }

    const evidenceCards = buildGraphEvidenceCards(items);

    setDrawerOpen(true);

    handleDrawerWorkspaceStateChange({
      ...drawerWorkspaceState,
      activeTab: evidenceCards.length ? "Evidence" : "Sources",
      selectedDrawerItem: graphSelectionToDrawerSelection(items, evidenceCards)
    });
  }
  function handleInspectGraphItem(item) {
    if (!item) {
      return;
    }

    const evidenceCards = buildGraphEvidenceCards([item]);

    setDrawerOpen(true);

    handleDrawerWorkspaceStateChange({
      ...drawerWorkspaceState,
      activeTab: evidenceCards.length ? "Evidence" : "Sources",
      selectedDrawerItem: graphSelectionToDrawerSelection([item], evidenceCards)
    });
  }

  function handleAddGraphSelectionToPrompt(items = selectedGraphItems) {
    if (!items.length) {
      return;
    }

    setGraphContextItems((current) => mergeGraphContextItems(current, items));
    setActiveWorkspaceTab("chat");
    setDrawerOpen(true);
    setNotice(`${items.length} graph item${items.length === 1 ? "" : "s"} added to the next prompt.`);
  }

  function handleRemoveGraphContextItem(itemId) {
    setGraphContextItems((current) =>
      current.filter((item) => item.id !== itemId)
    );
  }

  function handleClearGraphContext() {
    setGraphContextItems([]);
  }

  function handleViewReasoningPath(block) {
    const items = normaliseReasoningPathSelection(block, localSubgraph);

    if (!items.length) {
      setNotice("This reasoning block does not include KG path identifiers yet.");
      return;
    }

    const evidenceCards = buildGraphEvidenceCards(items);

    setSelectedGraphItems(items);
    setActiveWorkspaceTab("kg");
    setDrawerOpen(true);
    setKgFocusRequest({
      ids: items.map((item) => item.id).filter(Boolean),
      nonce: Date.now()
    });

    handleDrawerWorkspaceStateChange({
      ...drawerWorkspaceState,
      activeTab: evidenceCards.length ? "Evidence" : "Sources",
      selectedDrawerItem: graphSelectionToDrawerSelection(items, evidenceCards)
    });

    setNotice(`Selected ${items.length} reasoning-path item${items.length === 1 ? "" : "s"} in the Local KG.`);
  }

  function handleAccount() {
    setNotice(`Signed in as ${userEmail}. Local workspace state is stored under runs/ui_state/.`);
  }

  function handleExit() {
    clearTimers();

    if (typeof onSignOut === "function") {
      onSignOut();
      return;
    }

    if (typeof onBack === "function") {
      onBack();
      return;
    }

    window.location.hash = "/";
  }

  const userEmail =
    session?.user?.email ||
    session?.email ||
    "analyst@lanthic.local";

  const graphEvidenceCards = useMemo(
    () => buildGraphEvidenceCards(selectedGraphItems),
    [selectedGraphItems]
  );

  const drawerEvidence = graphEvidenceCards.length
    ? [...graphEvidenceCards, ...data.drawer.evidence]
    : data.drawer.evidence;

  const currentInvestigationDisplay = useMemo(
    () =>
      buildCurrentInvestigationDisplay({
        currentInvestigation: data.currentInvestigation,
        workspaceResult,
        chatTurns,
        question,
        isRunning,
        isLoadingInvestigation
      }),
    [data.currentInvestigation, workspaceResult, chatTurns, question, isRunning, isLoadingInvestigation]
  );

  return (
    <main className="analyst-workspace">
      <header className="lanthic-workspace-topbar">
        <button
          className="workspace-brand-lockup"
          type="button"
          onClick={onBack || handleExit}
        >
          <img src={markUrl} alt="" />
          <span>Lanthic Intelligence</span>
        </button>

        <div className="workspace-topbar-actions">
          <button
            className="workspace-export-button"
            type="button"
            onClick={handleNewInvestigation}
            disabled={isRunning || isLoadingInvestigation}
          >
            <Plus size={16} />
            New
          </button>

          <button
            className="workspace-export-button"
            type="button"
            onClick={handleExport}
            disabled={isRunning}
          >
            <Download size={16} />
            Export
          </button>

          <button
            className="workspace-account-button"
            type="button"
            onClick={handleAccount}
          >
            <UserCircle size={18} />
            <span>{userEmail}</span>
            <ChevronDown size={15} />
          </button>

          <button
            className="workspace-exit-button"
            type="button"
            onClick={handleExit}
            aria-label="Exit workspace"
          >
            <LogOut size={17} />
          </button>
        </div>
      </header>

      <input
        ref={fileInputRef}
        type="file"
        multiple
        style={{ display: "none" }}
        onChange={handleFilesSelected}
      />

      {notice ? (
        <div className="workspace-notice-bar">
          <span>{notice}</span>
          <button type="button" onClick={() => setNotice("")}>
            Dismiss
          </button>
        </div>
      ) : null}

      <div className={`lanthic-workspace-layout ${drawerOpen ? "" : "drawer-closed"}`}>
        <InvestigationSidebar
          currentInvestigation={currentInvestigationDisplay}
          recentInvestigations={data.recentInvestigations}
          selectedInvestigationId={selectedInvestigationId}
          onSelectInvestigation={handleSelectInvestigation}
          onAddDocuments={handleAddDocuments}
        />

        <section className="workspace-center" aria-label="Investigation workspace">
          <WorkspaceTabs
            activeTab={activeWorkspaceTab}
            onChange={setActiveWorkspaceTab}
            tabs={[
              WORKSPACE_TABS[0],
              {
                ...WORKSPACE_TABS[1],
                count: localSubgraph?.nodes?.length || 0
              }
            ]}
          />

          {activeWorkspaceTab === "chat" ? (
            <>
              <div className="chat-tab-layout">
                {(isRunning || isLoadingInvestigation) ? (
                  <InvestigationProgress
                    stages={progress.stages}
                    loopMessage={progress.loopMessage}
                    show
                  />
                ) : null}

                <ChatTurnStream
                  turns={chatTurns}
                  fallbackData={hasRunResult ? data : null}
                  progress={null}
                  isRunning={false}
                  onViewEvidence={() => setDrawerOpen(true)}
                  onViewMissingEvidence={() => setDrawerOpen(true)}
                  onViewReasoningPath={handleViewReasoningPath}
                />

                <div className="chat-composer-dock">
                  {graphContextItems.length ? (
                    <div className="graph-context-strip">
                      <div>
                        <strong>Selected KG context</strong>
                        <span>{graphContextItems.length} item{graphContextItems.length === 1 ? "" : "s"} will be sent with the next run.</span>
                      </div>

                      <div className="graph-context-chip-row">
                        {graphContextItems.map((item) => (
                          <button
                            type="button"
                            key={item.id}
                            className="graph-context-chip"
                            onClick={() => handleRemoveGraphContextItem(item.id)}
                            title="Remove context item"
                          >
                            {item.graphKind === "edge" ? "Edge" : item.type || "Node"}: {item.label}
                            <span>×</span>
                          </button>
                        ))}

                        <button
                          className="graph-context-clear"
                          type="button"
                          onClick={handleClearGraphContext}
                        >
                          Clear context
                        </button>
                      </div>
                    </div>
                  ) : null}

                  <QuestionComposer
                    question={question}
                    onQuestionChange={setQuestion}
                    onRun={handleRun}
                    isRunning={isRunning || isLoadingInvestigation}
                  />
                </div>
              </div>
            </>
          ) : (
            <KGViewer
              investigationId={selectedInvestigationId}
              graph={localSubgraph}
              isLoading={isLoadingSubgraph}
              error={subgraphError}
              selectedItems={selectedGraphItems}
              focusSelectionRequest={kgFocusRequest}
              onSelectionChange={handleGraphSelectionChange}
              onInspectItem={handleInspectGraphItem}
              onAddSelectionToPrompt={handleAddGraphSelectionToPrompt}
            />
          )}
        </section>

        {drawerOpen ? (
          <EvidenceDrawer
            sources={data.drawer.sources}
            evidence={drawerEvidence}
            assumptions={data.drawer.assumptions}
            workspaceState={drawerWorkspaceState}
            onWorkspaceStateChange={handleDrawerWorkspaceStateChange}
            onClose={() => setDrawerOpen(false)}
            onPin={() => setDrawerOpen(true)}
          />
        ) : (
          <button
            className="evidence-drawer-reopen"
            type="button"
            onClick={() => setDrawerOpen(true)}
          >
            Open evidence drawer
          </button>
        )}
      </div>
    </main>
  );
}

function buildProgressStages(activeId) {
  const activeIndex = BASE_PROGRESS_STAGES.findIndex((stage) => stage.id === activeId);

  return BASE_PROGRESS_STAGES.map((stage, index) => {
    if (index < activeIndex) {
      return {
        ...stage,
        status: "complete"
      };
    }

    if (stage.id === activeId) {
      return {
        ...stage,
        status: "active"
      };
    }

    return {
      ...stage,
      status: "pending"
    };
  });
}

function mergeInvestigationData(base, incoming) {
  if (!incoming) {
    return base;
  }

  return {
    ...base,
    ...incoming,
    currentInvestigation: {
      ...base.currentInvestigation,
      ...(incoming.currentInvestigation || {})
    },
    progress: {
      ...base.progress,
      ...(incoming.progress || {})
    },
    brief: {
      ...base.brief,
      ...(incoming.brief || {})
    },
    evidenceSupport: {
      ...base.evidenceSupport,
      ...(incoming.evidenceSupport || {})
    },
    riskAssessment: {
      ...base.riskAssessment,
      ...(incoming.riskAssessment || {})
    },
    forecastCheck: {
      ...base.forecastCheck,
      ...(incoming.forecastCheck || {})
    },
    missingEvidence: {
      ...base.missingEvidence,
      ...(incoming.missingEvidence || {})
    },
    drawer: {
      ...base.drawer,
      ...(incoming.drawer || {}),
      sources: incoming.drawer?.sources || base.drawer.sources,
      evidence: incoming.drawer?.evidence || base.drawer.evidence,
      assumptions: incoming.drawer?.assumptions || base.drawer.assumptions
    },
    recentInvestigations:
      incoming.recentInvestigations || base.recentInvestigations,
    reasoningPath:
      incoming.reasoningPath || base.reasoningPath
  };
}

function buildCurrentInvestigationDisplay({
  currentInvestigation = {},
  workspaceResult = null,
  chatTurns = [],
  question = "",
  isRunning = false,
  isLoadingInvestigation = false
} = {}) {
  const latestTurn = chatTurns.length ? chatTurns[chatTurns.length - 1] : null;
  const latestResult = latestTurn?.result || workspaceResult || null;
  const latestBlocks = Array.isArray(latestResult?.analysisBlocks)
    ? latestResult.analysisBlocks
    : [];
  const gapStatus = String(latestResult?.gap_assessment?.status || "").toLowerCase();
  const rawStatus = String(latestResult?.status || currentInvestigation.status || "").toLowerCase();
  const hasAnswer = Boolean(
    latestResult?.answer ||
      latestBlocks.length ||
      latestResult?.selected_reasoning_paths?.length
  );
  const hasRisk = Boolean(
    latestResult?.risk_analysis ||
      latestBlocks.some((block) => block?.type === "risk_assessment")
  );

  let status = "Ready";

  if (isRunning) {
    status = "Running";
  } else if (isLoadingInvestigation) {
    status = "Loading";
  } else if (!hasAnswer && !chatTurns.length) {
    status = "No turns yet";
  } else if (hasRisk) {
    status = "Risk assessed";
  } else if (["partial", "insufficient"].includes(gapStatus)) {
    status = "Partial answer";
  } else if (hasAnswer || ["answered", "complete", "sufficient"].includes(rawStatus)) {
    status = "Answer ready";
  } else if (rawStatus && rawStatus !== "ready") {
    status = titleCaseStatus(rawStatus);
  }

  const title =
    currentInvestigation.chatName ||
    currentInvestigation.title ||
    latestResult?.question ||
    latestTurn?.question ||
    question ||
    NEW_INVESTIGATION_TITLE;

  return {
    ...currentInvestigation,
    title,
    chatName: title,
    status,
    updatedAt: isRunning
      ? "Running now"
      : currentInvestigation.updatedAt || currentInvestigation.lastModifiedAt || "Just now"
  };
}

function titleCaseStatus(value) {
  return String(value || "")
    .replace(/[_-]+/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function normaliseInvestigationSummary(value = {}) {
  const documentCount = Number(value.documentCount ?? value.documents?.length ?? 0);
  const turnCount = Number(value.turnCount ?? value.turns?.length ?? 0);

  return {
    investigationId:
      value.investigationId ||
      value.id ||
      DEFAULT_INVESTIGATION_ID,
    title:
      value.title ||
      value.chatName ||
      NEW_INVESTIGATION_TITLE,
    chatName:
      value.chatName ||
      value.title ||
      NEW_INVESTIGATION_TITLE,
    status:
      value.status ||
      "Ready",
    updatedAt:
      value.updatedAt ||
      value.lastModifiedAt ||
      "Just now",
    createdAt:
      value.createdAt,
    lastModifiedAt:
      value.lastModifiedAt || value.updatedAt,
    turnCount:
      Number.isFinite(turnCount) ? turnCount : 0,
    documentCount:
      Number.isFinite(documentCount) ? documentCount : 0
  };
}

function buildBlankWorkspaceData(investigation, documents = []) {
  const summary = normaliseInvestigationSummary(investigation);
  const uploadedSources = documents.map(documentToSource);
  const uploadedEvidence = documents.map(documentToEvidence);

  return {
    investigationId: summary.investigationId,
    question: "",
    currentInvestigation: summary,
    progress: {
      stages: PENDING_PROGRESS_STAGES,
      loopMessage: ""
    },
    brief: {
      answerLead: "",
      summary: ""
    },
    evidenceSupport: {
      totalCount: uploadedEvidence.length,
      evidence: uploadedEvidence.slice(0, 5).map((item) => ({
        id: item.id,
        text: item.text,
        source: item.source,
        date: item.date
      }))
    },
    reasoningPath: [],
    riskAssessment: {
      overallRisk: "",
      factors: [],
      summary: ""
    },
    forecastCheck: {
      status: "",
      summary: "",
      showChart: false
    },
    missingEvidence: {
      totalCount: 0,
      items: []
    },
    drawer: {
      sources: uploadedSources,
      evidence: uploadedEvidence,
      assumptions: []
    }
  };
}

function addUploadedDocumentsToWorkspace(current, documents = []) {
  if (!documents.length) {
    return current;
  }

  const sources = documents.map(documentToSource);
  const evidence = documents.map(documentToEvidence);

  const existingDrawer = current?.drawer || {};
  const existingEvidenceSupport = current?.evidenceSupport || {};

  const nextEvidence = [
    ...evidence,
    ...(existingDrawer.evidence || [])
  ];

  return {
    ...current,
    drawer: {
      ...existingDrawer,
      sources: [
        ...sources,
        ...(existingDrawer.sources || [])
      ],
      evidence: nextEvidence,
      assumptions: existingDrawer.assumptions || []
    },
    evidenceSupport: {
      ...existingEvidenceSupport,
      totalCount: nextEvidence.length,
      evidence: nextEvidence.slice(0, 5).map((item) => ({
        id: item.id,
        text: item.text,
        source: item.source,
        date: item.date
      }))
    },
    missingEvidence: {
      ...(current?.missingEvidence || {}),
      items: [
        "Re-run the investigation to include newly uploaded documents in the reasoning pass",
        ...((current?.missingEvidence?.items || []).filter(
          (item) => item !== "Re-run the investigation to include newly uploaded documents in the reasoning pass"
        ))
      ],
      totalCount: Math.max(1, current?.missingEvidence?.totalCount || 0)
    }
  };
}

function documentToSource(document, index = 0) {
  const title =
    document.sourceTitle ||
    document.filename ||
    `Uploaded document ${index + 1}`;

  return {
    id: document.documentId || `uploaded_source_${index}`,
    logo: "U",
    title,
    date: humanDate(document.uploadedAt),
    excerpt:
      document.excerpt ||
      "Uploaded file is attached to this investigation.",
    tag: document.textExtractAvailable
      ? "Uploaded file"
      : "Uploaded file · extraction pending"
  };
}

function documentToEvidence(document, index = 0) {
  const title =
    document.sourceTitle ||
    document.filename ||
    `Uploaded document ${index + 1}`;

  return {
    id: `uploaded_evidence_${document.documentId || index}`,
    title: document.textExtractAvailable
      ? "Uploaded document excerpt"
      : "Uploaded document attached",
    source: title,
    date: humanDate(document.uploadedAt),
    text:
      document.excerpt ||
      "The document is attached locally but no text excerpt could be extracted automatically.",
    support: "User-provided evidence"
  };
}

function humanDate(value) {
  if (!value) {
    return "Just now";
  }

  try {
    return new Date(value).toLocaleString(undefined, {
      year: "numeric",
      month: "short",
      day: "numeric"
    });
  } catch {
    return value;
  }
}

function normaliseDrawerWorkspaceState(value = {}) {
  const source = value && typeof value === "object" ? value : {};
  const activeTab = ["Sources", "Evidence", "Assumptions"].includes(source.activeTab)
    ? source.activeTab
    : DEFAULT_DRAWER_WORKSPACE_STATE.activeTab;

  return {
    activeTab,
    showPinnedOnly: Boolean(source.showPinnedOnly),
    bookmarks: normaliseDrawerWorkspaceBuckets(source.bookmarks),
    pins: normaliseDrawerWorkspaceBuckets(source.pins),
    selectedDrawerItem:
      source.selectedDrawerItem && typeof source.selectedDrawerItem === "object"
        ? source.selectedDrawerItem
        : null
  };
}

function normaliseDrawerWorkspaceBuckets(value = {}) {
  const source = value && typeof value === "object" ? value : {};

  return DRAWER_WORKSPACE_BUCKETS.reduce((result, bucket) => {
    const raw = Array.isArray(source[bucket]) ? source[bucket] : [];
    const seen = new Set();

    result[bucket] = raw
      .map((item) => String(item || "").trim())
      .filter((item) => {
        if (!item || seen.has(item)) {
          return false;
        }

        seen.add(item);
        return true;
      });

    return result;
  }, {});
}

function graphMetadataToText(metadata) {
  if (!metadata || typeof metadata !== "object") {
    return "";
  }

  const entries = Object.entries(metadata)
    .filter(([, value]) => value !== undefined && value !== null && value !== "")
    .slice(0, 8);

  if (!entries.length) {
    return "";
  }

  return entries
    .map(([key, value]) => {
      const rendered =
        typeof value === "string"
          ? value
          : JSON.stringify(value);

      return `${key}: ${rendered}`;
    })
    .join("\n");
}

function mergeGraphContextItems(current = [], incoming = []) {
  const seen = new Set(current.map((item) => item.id));
  const merged = [...current];

  incoming.forEach((item) => {
    if (!item?.id || seen.has(item.id)) {
      return;
    }

    seen.add(item.id);
    merged.push(item);
  });

  return merged;
}

function normaliseReasoningPathSelection(block = {}, graph = {}) {
  const lookup = buildGraphLookup(graph);
  const rawPath = extractReasoningPathItems(block);
  const selected = [];
  const seen = new Set();

  rawPath.forEach((item) => {
    const normalised = normaliseReasoningPathItem(item, lookup);

    if (!normalised?.id || seen.has(normalised.id)) {
      return;
    }

    seen.add(normalised.id);
    selected.push(normalised);
  });

  return selected;
}

function extractReasoningPathItems(block = {}) {
  const data = block.data || block;
  const meta = block.meta || block.metadata || {};

  const candidates =
    data.path ||
    data.graphPath ||
    data.graph_path ||
    data.kgPath ||
    data.kg_path ||
    data.items ||
    meta.path ||
    meta.graphPath ||
    meta.graph_path ||
    meta.graphItemIds ||
    meta.graph_item_ids ||
    [];

  if (!Array.isArray(candidates)) {
    return [];
  }

  return candidates;
}

function normaliseReasoningPathItem(item, lookup) {
  if (typeof item === "string") {
    return lookup.byId.get(item) || {
      graphKind: "node",
      id: item,
      label: item,
      type: "entity",
      evidence: []
    };
  }

  if (!item || typeof item !== "object") {
    return null;
  }

  const id = String(item.id || item.key || item.graphId || item.graph_id || "").trim();

  if (id && lookup.byId.has(id)) {
    return {
      ...lookup.byId.get(id),
      ...item,
      id
    };
  }

  if (id) {
    return {
      graphKind: item.graphKind || item.kind || (item.relation ? "edge" : "node"),
      id,
      label: item.label || item.title || id,
      type: item.type || item.entityType || item.entity_type || "entity",
      relation: item.relation || item.relationType || item.relation_type || "",
      source: item.source || "",
      target: item.target || "",
      text: item.text || item.summary || "",
      evidence: Array.isArray(item.evidence) ? item.evidence : [],
      metadata: item.metadata || item.meta || {}
    };
  }

  return null;
}

function buildGraphLookup(graph = {}) {
  const byId = new Map();
  const nodes = Array.isArray(graph?.nodes) ? graph.nodes : [];
  const edges = Array.isArray(graph?.edges) ? graph.edges : [];

  nodes.forEach((node) => {
    if (!node?.id) {
      return;
    }

    const id = String(node.id);

    byId.set(id, {
      graphKind: "node",
      id,
      label: node.label || id,
      type: node.taxonomyType || node.type || node.metadata?.entityType || "entity",
      text: node.text || "",
      evidence: Array.isArray(node.evidence) ? node.evidence : [],
      metadata: node.metadata || {}
    });
  });

  edges.forEach((edge, index) => {
    if (!edge?.source || !edge?.target) {
      return;
    }

    const relation = edge.relation || edge.label || "related_to";
    const id = String(edge.id || `edge:${edge.source}:${relation}:${edge.target}:${index}`);

    byId.set(id, {
      graphKind: "edge",
      id,
      label: String(edge.label || relation).replace(/_/g, " "),
      relation,
      source: String(edge.source),
      target: String(edge.target),
      text: edge.text || "",
      evidence: Array.isArray(edge.evidence) ? edge.evidence : [],
      metadata: edge.metadata || edge
    });
  });

  return { byId };
}

function graphSelectionToDrawerSelection(items = [], evidenceCards = []) {
  if (items.length === 1) {
    return graphItemToDrawerSelection(items[0], evidenceCards);
  }

  const labels = items
    .slice(0, 5)
    .map((item) => item.label || item.id)
    .join(", ");

  return {
    kind: "graphItems",
    key: `graph_selection_${items.map((item) => item.id).join("_")}`,
    title: `${items.length} KG items selected`,
    subtitle: evidenceCards.length
      ? `${evidenceCards.length} linked evidence excerpt${evidenceCards.length === 1 ? "" : "s"}`
      : "No linked evidence excerpts",
    text:
      evidenceCards.length
        ? `Selected: ${labels}\n\nThe Evidence tab is showing all evidence linked to the selected KG items.`
        : `Selected: ${labels}\n\nNo evidence excerpts were attached to this graph selection.`,
    item: {
      graphItems: items,
      evidence: evidenceCards
    }
  };
}

function graphItemToDrawerSelection(item = {}, evidenceCards = []) {
  const isEdge = item.graphKind === "edge";
  const subtitle = isEdge
    ? [item.sourceLabel || item.source, item.relation, item.targetLabel || item.target].filter(Boolean).join(" → ")
    : [item.type, item.id].filter(Boolean).join(" · ");

  const evidenceSummary = evidenceCards.length
    ? `\n\nLinked evidence excerpts: ${evidenceCards.length}. See the Evidence tab for the full linked evidence set.`
    : "";

  return {
    kind: "graphItems",
    key: item.id || item.label,
    title: item.label || item.id || "Graph item",
    subtitle,
    text:
      item.text ||
      graphMetadataToText(item.metadata) ||
      (isEdge
        ? "This relation is part of the local extracted KG."
        : "This entity is part of the local extracted KG.") +
        evidenceSummary,
    item: {
      ...item,
      evidence: evidenceCards
    }
  };
}

function buildGraphEvidenceCards(items = []) {
  const seen = new Set();
  const cards = [];

  items.forEach((item) => {
    const evidence = Array.isArray(item?.evidence) ? item.evidence : [];

    evidence.forEach((entry, index) => {
      const id = String(entry.id || entry.evidence_id || `${item.id}_evidence_${index}`);

      if (seen.has(id)) {
        return;
      }

      seen.add(id);

      cards.push({
        id: `kg_${id}`,
        title: entry.title || `Evidence linked to ${item.label || item.id}`,
        source: entry.source || item.label || "Local KG evidence",
        date: entry.date || "Evidence block",
        text: entry.text || "No evidence text available.",
        support: entry.support || `Linked to ${item.graphKind === "edge" ? "relation" : "entity"}: ${item.label || item.id}`
      });
    });
  });

  return cards;
}

function safeFilename(value) {
  return String(value || "lanthic-investigation")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 80) || "lanthic-investigation";
}