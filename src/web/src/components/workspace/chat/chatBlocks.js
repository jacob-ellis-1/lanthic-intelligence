export function normaliseChatTurns(turns = [], fallbackData = null) {
  const rawTurns = Array.isArray(turns) ? turns : [];

  if (rawTurns.length) {
    return rawTurns.map((turn, index) => normaliseTurn(turn, index)).filter(Boolean);
  }

  if (fallbackData) {
    return [
      normaliseTurn(
        {
          id: "current_result",
          question: fallbackData.question || "Current investigation question",
          createdAt: fallbackData.currentInvestigation?.updatedAt || fallbackData.currentInvestigation?.createdAt,
          result: fallbackData
        },
        0
      )
    ].filter(Boolean);
  }

  return [];
}

export function appendTurnFromRunResult(currentTurns = [], result = {}, fallback = {}) {
  if (Array.isArray(result?.turns)) {
    return result.turns;
  }

  const rawTurn =
    result?.turn ||
    result?.latestTurn ||
    result?.latest_turn ||
    {
      id: result?.turn_id || result?.id || `turn_${Date.now()}`,
      question: fallback.question || result?.question || "",
      createdAt: result?.createdAt || result?.created_at || new Date().toISOString(),
      selectedGraphContext:
        fallback.selectedGraphContext ||
        result?.selectedGraphContext ||
        result?.selected_graph_context ||
        [],
      result: result?.result || result?.data || result
    };

  const nextId = rawTurn.id || rawTurn.turn_id || `turn_${Date.now()}`;

  return [
    ...currentTurns.filter((turn) => String(turn?.id || turn?.turn_id) !== String(nextId)),
    rawTurn
  ];
}

export function normaliseTurn(rawTurn = {}, index = 0) {
  const result = extractTurnResult(rawTurn);
  const question =
    firstText(
      rawTurn.question,
      rawTurn.userQuestion,
      rawTurn.user_question,
      result.question,
      result.userQuestion,
      result.user_question
    ) || `Turn ${index + 1}`;

  return {
    id: String(rawTurn.id || rawTurn.turn_id || rawTurn.turnId || `turn_${index + 1}`),
    question,
    createdAt: firstText(rawTurn.createdAt, rawTurn.created_at, rawTurn.timestamp, result.createdAt, result.created_at),
    status: firstText(rawTurn.status, result.status, "complete"),
    selectedGraphContext:
      normaliseGraphContext(
        rawTurn.selectedGraphContext ||
        rawTurn.selected_graph_context ||
        result.selectedGraphContext ||
        result.selected_graph_context ||
        []
      ),
    blocks: deriveBlocksFromResult(result)
  };
}

export function deriveBlocksFromResult(result = {}) {
  const explicitBlocks =
    result.analysisBlocks ||
    result.analysis_blocks ||
    result.blocks ||
    result.blockStream ||
    result.block_stream;

  if (Array.isArray(explicitBlocks) && explicitBlocks.length) {
    return explicitBlocks.map(normaliseAnalysisBlock).filter(Boolean);
  }

  const blocks = [];

  const brief = result.brief || {
    answerLead: result.answerLead || result.answer_lead || result.answer || result.summary,
    summary: result.summary
  };

  if (brief.answerLead || brief.answer_lead || brief.summary) {
    blocks.push({
      id: "brief",
      type: "text",
      title: "Brief",
      data: {
        lead: brief.answerLead || brief.answer_lead,
        body: brief.summary
      },
      meta: {}
    });
  }

  const evidenceSupport = result.evidenceSupport || result.evidence_support;

  if (evidenceSupport?.evidence?.length) {
    blocks.push({
      id: "evidence_support",
      type: "evidence",
      title: "Evidence support",
      data: evidenceSupport,
      meta: {}
    });
  }

  const reasoningPath = result.reasoningPath || result.reasoning_path || result.reasoning;

  if (Array.isArray(reasoningPath) && reasoningPath.length) {
    blocks.push({
      id: "reasoning_path",
      type: "reasoning_path",
      title: "Reasoning path",
      data: {
        nodes: reasoningPath
      },
      meta: {}
    });
  }

  const riskAssessment = result.riskAssessment || result.risk_assessment;

  if (riskAssessment?.overallRisk || riskAssessment?.overall_risk || riskAssessment?.factors?.length) {
    blocks.push({
      id: "risk_assessment",
      type: "risk_assessment",
      title: "Risk assessment",
      data: riskAssessment,
      meta: {}
    });
  }

  const forecastCheck = result.forecastCheck || result.forecast_check;

  if (forecastCheck?.status || forecastCheck?.summary) {
    blocks.push({
      id: "forecast_check",
      type: "forecast",
      title: "Forecast check",
      data: forecastCheck,
      meta: {}
    });
  }

  const missingEvidence = result.missingEvidence || result.missing_evidence;

  if (missingEvidence?.items?.length) {
    blocks.push({
      id: "missing_evidence",
      type: "missing_evidence",
      title: "Missing evidence",
      data: missingEvidence,
      meta: {}
    });
  }

  return blocks;
}

export function normaliseGraphContext(items = []) {
  if (!Array.isArray(items)) {
    return [];
  }

  return items
    .filter(Boolean)
    .map((item, index) => ({
      id: String(item.id || item.key || `context_${index}`),
      label: item.label || item.title || item.id || "KG item",
      type: item.type || item.graphKind || item.kind || "KG item",
      graphKind: item.graphKind || item.kind || "node",
      relation: item.relation || "",
      source: item.source || "",
      target: item.target || ""
    }));
}

function normaliseAnalysisBlock(block = {}, index = 0) {
  if (!block || typeof block !== "object") {
    return null;
  }

  const type = block.type || "text";

  return {
    ...block,
    id: String(block.id || block.blockId || block.block_id || `${type}_${index}`),
    type,
    title: block.title || titleForBlockType(type),
    data: block.data || block.payload || block.content || block,
    meta: block.meta || block.metadata || {}
  };
}

function extractTurnResult(rawTurn = {}) {
  return (
    rawTurn.result ||
    rawTurn.output ||
    rawTurn.response ||
    rawTurn.analysis ||
    rawTurn.data ||
    rawTurn
  );
}

function firstText(...values) {
  for (const value of values) {
    if (typeof value === "string" && value.trim()) {
      return value.trim();
    }
  }

  return "";
}

function titleForBlockType(type = "") {
  return String(type || "text")
    .replace(/_/g, " ")
    .replace(/^\w/, (letter) => letter.toUpperCase());
}