import { useEffect, useMemo, useRef, useState } from "react";
import CytoscapeComponent from "react-cytoscapejs";
import cytoscape from "cytoscape";
import fcose from "cytoscape-fcose";

let fcoseRegistered = false;

if (!fcoseRegistered) {
  cytoscape.use(fcose);
  fcoseRegistered = true;
}

const FORCE_LAYOUT_OPTIONS = {
  name: "fcose",
  quality: "proof",
  randomize: true,
  animate: true,
  animationDuration: 650,
  fit: true,
  padding: 58,
  nodeDimensionsIncludeLabels: true,
  uniformNodeDimensions: false,
  packComponents: true,
  nodeSeparation: 48,
  nodeRepulsion: () => 9200,
  idealEdgeLength: () => 132,
  edgeElasticity: () => 0.34,
  nestingFactor: 0.1,
  gravity: 0.18,
  gravityRange: 3.8,
  gravityCompound: 0.8,
  gravityRangeCompound: 1.5,
  numIter: 3200,
  tile: true,
  initialEnergyOnIncremental: 0.35
};

const INCREMENTAL_LAYOUT_OPTIONS = {
  ...FORCE_LAYOUT_OPTIONS,
  quality: "default",
  randomize: false,
  animationDuration: 420,
  numIter: 1600,
  initialEnergyOnIncremental: 0.18
};

const NODE_TYPE_COLOURS = {
  country: "#2f6f9f",
  company: "#6d5bd0",
  organization: "#6d5bd0",
  organisation: "#6d5bd0",
  facility: "#c47a32",
  mine: "#c47a32",
  mining_site: "#c47a32",
  site: "#c47a32",
  township: "#5f8f4f",
  region: "#5f8f4f",
  state: "#5f8f4f",
  province: "#5f8f4f",
  commodity: "#b7892b",
  mineral: "#b7892b",
  material: "#b7892b",
  product: "#b7892b",
  event: "#9a4f4f",
  policy: "#4f6f8f",
  entity: "#172033",
  node: "#172033"
};

const CY_STYLESHEET = [
  {
    selector: "node",
    style: {
      "background-color": "data(color)",
      "border-width": 0,
      "color": "#172033",
      "font-size": 10,
      "font-weight": 760,
      "height": 40,
      "label": "data(label)",
      "overlay-opacity": 0,
      "text-background-color": "#f7f4ec",
      "text-background-opacity": 0.9,
      "text-background-padding": 4,
      "text-margin-y": -8,
      "text-max-width": 124,
      "text-valign": "top",
      "text-wrap": "wrap",
      "width": 40
    }
  },
  {
    selector: "edge",
    style: {
      "curve-style": "bezier",
      "font-size": 8,
      "label": "data(label)",
      "line-color": "rgba(23, 32, 51, 0.28)",
      "overlay-opacity": 0,
      "target-arrow-color": "rgba(23, 32, 51, 0.32)",
      "target-arrow-shape": "triangle",
      "text-background-color": "#f7f4ec",
      "text-background-opacity": 0.76,
      "text-background-padding": 3,
      "text-rotation": "autorotate",
      "width": 1.6
    }
  },
  {
    selector: "node:selected",
    style: {
      "border-color": "#e5f24c",
      "border-width": 4,
      "shadow-blur": 18,
      "shadow-color": "#e5f24c",
      "shadow-opacity": 0.72,
      "underlay-color": "#5bc56a",
      "underlay-opacity": 0.26,
      "underlay-padding": 11
    }
  },
  {
    selector: "edge:selected",
    style: {
      "line-color": "#5bc56a",
      "target-arrow-color": "#5bc56a",
      "width": 3
    }
  }
];

export default function KGViewer({
  investigationId,
  graph,
  isLoading = false,
  error = "",
  selectedItems = [],
  focusSelectionRequest = null,
  onSelectionChange,
  onInspectItem,
  onAddSelectionToPrompt
}) {
  const cyRef = useRef(null);
  const saveTimerRef = useRef(null);
  const selectionBeforeTapRef = useRef([]);
  const storageKey = useMemo(
    () => kgLayoutStorageKey(investigationId || graph?.investigationId || "default"),
    [investigationId, graph?.investigationId]
  );

  const [savedLayout, setSavedLayout] = useState(() => loadSavedLayout(storageKey));

  useEffect(() => {
    setSavedLayout(loadSavedLayout(storageKey));
  }, [storageKey]);

  useEffect(() => {
    const cy = cyRef.current;
    const ids = Array.isArray(focusSelectionRequest?.ids)
      ? focusSelectionRequest.ids
      : [];

    if (!cy || !ids.length) {
      return;
    }

    applySelection(cy, ids);

    const selected = readSelection(cy);
    onSelectionChange?.(selected);

    const selectedElements = cy.$(":selected");
    if (selectedElements.length) {
      cy.fit(selectedElements, 92);
    }
  }, [focusSelectionRequest?.nonce]);

  const elements = useMemo(
    () => graphToElements(graph, savedLayout?.positions || {}),
    [graph, savedLayout]
  );

  const counts = useMemo(() => graphCounts(graph), [graph]);

  const hasSavedPositions = useMemo(
    () => hasUsableSavedPositions(graph, savedLayout),
    [graph, savedLayout]
  );

  useEffect(() => {
    const cy = cyRef.current;

    if (!cy) {
      return undefined;
    }

    function handleTapStart(event) {
      if (event.target === cy) {
        selectionBeforeTapRef.current = readSelection(cy);
        return;
      }

      selectionBeforeTapRef.current = readSelection(cy);
    }

    function handleElementTap(event) {
      const element = event.target;
      const originalEvent = event.originalEvent || {};
      const additive = Boolean(originalEvent.shiftKey || originalEvent.metaKey || originalEvent.ctrlKey);

      const previousSelection = selectionBeforeTapRef.current || [];
      const previousIds = previousSelection.map((item) => item.id);
      let nextIds;

      if (additive) {
        if (previousIds.includes(element.id())) {
          nextIds = previousIds.filter((id) => id !== element.id());
        } else {
          nextIds = [...previousIds, element.id()];
        }
      } else {
        nextIds = [element.id()];
      }

      applySelection(cy, nextIds);

      const selection = readSelection(cy);
      onSelectionChange?.(selection);

      const inspected = elementToSelectionItem(element);
      if (inspected) {
        onInspectItem?.(inspected);
      }
    }

    function handleCanvasTap(event) {
      if (event.target !== cy) {
        return;
      }

      const originalEvent = event.originalEvent || {};
      if (originalEvent.shiftKey || originalEvent.metaKey || originalEvent.ctrlKey) {
        return;
      }

      cy.elements().unselect();
      onSelectionChange?.([]);
    }

    function handleBoxSelection() {
      window.setTimeout(() => {
        onSelectionChange?.(readSelection(cy));
      }, 0);
    }

    function handleLayoutSave() {
      saveCurrentLayoutDebounced(cy, storageKey, saveTimerRef);
    }

    cy.off("tapstart", handleTapStart);
    cy.off("tap", "node, edge", handleElementTap);
    cy.off("tap", handleCanvasTap);
    cy.off("boxselect", "node, edge", handleBoxSelection);
    cy.off("dragfree", "node", handleLayoutSave);
    cy.off("pan zoom", handleLayoutSave);
    cy.off("layoutstop", handleLayoutSave);

    cy.on("tapstart", handleTapStart);
    cy.on("tap", "node, edge", handleElementTap);
    cy.on("tap", handleCanvasTap);
    cy.on("boxselect", "node, edge", handleBoxSelection);
    cy.on("dragfree", "node", handleLayoutSave);
    cy.on("pan zoom", handleLayoutSave);
    cy.on("layoutstop", handleLayoutSave);

    return () => {
      cy.off("tapstart", handleTapStart);
      cy.off("tap", "node, edge", handleElementTap);
      cy.off("tap", handleCanvasTap);
      cy.off("boxselect", "node, edge", handleBoxSelection);
      cy.off("dragfree", "node", handleLayoutSave);
      cy.off("pan zoom", handleLayoutSave);
      cy.off("layoutstop", handleLayoutSave);
    };
  }, [elements, onInspectItem, onSelectionChange, storageKey]);

  useEffect(() => {
    const cy = cyRef.current;

    if (!cy || !elements.length) {
      return;
    }

    const timer = window.setTimeout(() => {
      if (hasSavedPositions) {
        restoreSavedViewport(cy, savedLayout);
        return;
      }

      runForceLayout(cy, FORCE_LAYOUT_OPTIONS, storageKey, saveTimerRef);
    }, 80);

    return () => window.clearTimeout(timer);
  }, [elements, hasSavedPositions, savedLayout, storageKey]);

  function handleFitView() {
    const cy = cyRef.current;
    if (cy) {
      cy.fit(undefined, 58);
      saveCurrentLayoutDebounced(cy, storageKey, saveTimerRef);
    }
  }

  function handleRelayout() {
    const cy = cyRef.current;
    if (!cy) {
      return;
    }

    clearSavedLayout(storageKey);
    setSavedLayout(null);
    runForceLayout(cy, FORCE_LAYOUT_OPTIONS, storageKey, saveTimerRef);
  }

  function handleTidyLayout() {
    const cy = cyRef.current;
    if (!cy) {
      return;
    }

    runForceLayout(cy, INCREMENTAL_LAYOUT_OPTIONS, storageKey, saveTimerRef);
  }

  function handleClearSelection() {
    const cy = cyRef.current;
    if (cy) {
      cy.elements().unselect();
    }

    onSelectionChange?.([]);
  }

  function handleAddSelection() {
    if (!selectedItems.length) {
      return;
    }

    onAddSelectionToPrompt?.(selectedItems);
  }

  if (isLoading) {
    return (
      <section className="kg-viewer-card">
        <div className="kg-viewer-empty">Loading local KG…</div>
      </section>
    );
  }

  if (error) {
    return (
      <section className="kg-viewer-card">
        <div className="kg-viewer-empty">{error}</div>
      </section>
    );
  }

  if (!elements.length) {
    return (
      <section className="kg-viewer-card">
        <div className="kg-viewer-empty">
          No extracted entity-relation graph is available yet. Run ingestion or select a run with PostRAG entity/relation outputs.
        </div>
      </section>
    );
  }

  return (
    <section className="kg-viewer-card" aria-label="Local knowledge graph viewer">
      <div className="kg-viewer-toolbar">
        <div>
          <strong>Local entity-relation KG</strong>
          <span>{counts.nodes} entities · {counts.edges} relations</span>
        </div>

        <div className="kg-viewer-actions">
          <button type="button" onClick={handleFitView}>Fit view</button>
          <button type="button" onClick={handleTidyLayout}>Tidy</button>
          <button type="button" onClick={handleRelayout}>Re-layout</button>
          <button type="button" onClick={handleClearSelection}>Clear</button>
          <button
            className="primary"
            type="button"
            disabled={!selectedItems.length}
            onClick={handleAddSelection}
          >
            Add selection to prompt
          </button>
        </div>
      </div>

      <div className="kg-viewer-help">
        Pan, zoom, and drag entities. Positions are preserved for this investigation. Re-layout recomputes the force layout; Tidy improves spacing while keeping the current layout close.
      </div>

      <div className="kg-canvas">
        <CytoscapeComponent
          elements={elements}
          stylesheet={CY_STYLESHEET}
          layout={{ name: "preset", fit: false }}
          cy={(cy) => {
            cyRef.current = cy;
            cy.boxSelectionEnabled(true);
            cy.autoungrabify(false);
            cy.userPanningEnabled(true);
            cy.userZoomingEnabled(true);
          }}
          boxSelectionEnabled
          selectionType="additive"
          minZoom={0.18}
          maxZoom={2.6}
          style={{
            width: "100%",
            height: "100%"
          }}
        />
      </div>

      <div className="kg-selection-panel">
        <strong>{selectedItems.length ? `${selectedItems.length} selected` : "No KG selection"}</strong>

        {selectedItems.length ? (
          <div className="kg-selection-list">
            {selectedItems.slice(0, 6).map((item) => (
              <span key={item.id}>
                {item.graphKind === "edge" ? "Relation" : item.type || "Entity"}: {item.label}
              </span>
            ))}
            {selectedItems.length > 6 ? <span>+{selectedItems.length - 6} more</span> : null}
          </div>
        ) : (
          <p>Select entities or relations to inspect their linked evidence in the evidence drawer.</p>
        )}
      </div>
    </section>
  );
}

function graphToElements(graph, savedPositions = {}) {
  const nodes = Array.isArray(graph?.nodes) ? graph.nodes : [];
  const edges = Array.isArray(graph?.edges) ? graph.edges : [];

  const nodeElements = nodes
    .filter((node) => node?.id)
    .map((node) => {
      const id = String(node.id);
      const type = String(node.taxonomyType || node.type || node.metadata?.entityType || "entity").toLowerCase();
      const savedPosition = savedPositions[id];

      return {
        data: {
          id,
          label: shortLabel(node.label || id),
          fullLabel: node.label || id,
          type,
          color: colourForNodeType(type),
          text: node.text || "",
          evidence: Array.isArray(node.evidence) ? node.evidence : [],
          metadata: node.metadata || {}
        },
        ...(isValidPosition(savedPosition) ? { position: savedPosition } : {})
      };
    });

  const nodeIds = new Set(nodeElements.map((node) => node.data.id));

  const edgeElements = edges
    .filter((edge) => edge?.source && edge?.target)
    .filter((edge) => nodeIds.has(String(edge.source)) && nodeIds.has(String(edge.target)))
    .map((edge, index) => {
      const relation = edge.relation || edge.label || "related_to";
      const id = edge.id || `edge:${edge.source}:${relation}:${edge.target}:${index}`;

      return {
        data: {
          id: String(id),
          source: String(edge.source),
          target: String(edge.target),
          relation,
          label: String(relation).replace(/_/g, " "),
          text: edge.text || "",
          evidence: Array.isArray(edge.evidence) ? edge.evidence : [],
          metadata: edge.metadata || edge
        }
      };
    });

  return [...nodeElements, ...edgeElements];
}

function graphCounts(graph) {
  return {
    nodes: Array.isArray(graph?.nodes) ? graph.nodes.length : 0,
    edges: Array.isArray(graph?.edges) ? graph.edges.length : 0
  };
}

function applySelection(cy, ids = []) {
  const selectedIds = new Set(ids);

  cy.batch(() => {
    cy.elements().forEach((element) => {
      if (selectedIds.has(element.id())) {
        element.select();
      } else {
        element.unselect();
      }
    });
  });
}

function readSelection(cy) {
  return cy.$(":selected").map(elementToSelectionItem).filter(Boolean);
}

function elementToSelectionItem(element) {
  if (!element) {
    return null;
  }

  if (element.isNode()) {
    return {
      graphKind: "node",
      id: element.id(),
      label: element.data("fullLabel") || element.data("label") || element.id(),
      type: element.data("type") || "entity",
      text: element.data("text") || "",
      evidence: element.data("evidence") || [],
      metadata: element.data("metadata") || {}
    };
  }

  return {
    graphKind: "edge",
    id: element.id(),
    label: element.data("label") || element.data("relation") || "related to",
    relation: element.data("relation") || "related_to",
    source: element.data("source"),
    target: element.data("target"),
    text: element.data("text") || "",
    evidence: element.data("evidence") || [],
    metadata: element.data("metadata") || {}
  };
}

function runForceLayout(cy, options, storageKey, saveTimerRef) {
  const layout = cy.layout(options);

  layout.one("layoutstop", () => {
    saveCurrentLayout(cy, storageKey);
    if (saveTimerRef.current) {
      window.clearTimeout(saveTimerRef.current);
      saveTimerRef.current = null;
    }
  });

  layout.run();
}

function kgLayoutStorageKey(investigationId) {
  return `lanthic_kg_layout_${investigationId || "default"}`;
}

function loadSavedLayout(storageKey) {
  try {
    const raw = window.localStorage.getItem(storageKey);
    const parsed = raw ? JSON.parse(raw) : null;

    if (!parsed || typeof parsed !== "object") {
      return null;
    }

    return {
      positions: parsed.positions && typeof parsed.positions === "object"
        ? parsed.positions
        : {},
      zoom: Number.isFinite(Number(parsed.zoom)) ? Number(parsed.zoom) : null,
      pan: isValidPosition(parsed.pan) ? parsed.pan : null,
      savedAt: parsed.savedAt || null
    };
  } catch {
    return null;
  }
}

function saveCurrentLayoutDebounced(cy, storageKey, saveTimerRef) {
  if (saveTimerRef.current) {
    window.clearTimeout(saveTimerRef.current);
  }

  saveTimerRef.current = window.setTimeout(() => {
    saveCurrentLayout(cy, storageKey);
    saveTimerRef.current = null;
  }, 350);
}

function saveCurrentLayout(cy, storageKey) {
  if (!cy) {
    return;
  }

  const positions = {};

  cy.nodes().forEach((node) => {
    positions[node.id()] = node.position();
  });

  try {
    window.localStorage.setItem(
      storageKey,
      JSON.stringify({
        positions,
        zoom: cy.zoom(),
        pan: cy.pan(),
        savedAt: new Date().toISOString()
      })
    );
  } catch {
    // Ignore local layout persistence failures.
  }
}

function clearSavedLayout(storageKey) {
  try {
    window.localStorage.removeItem(storageKey);
  } catch {
    // Ignore local layout persistence failures.
  }
}

function hasUsableSavedPositions(graph, savedLayout) {
  const nodes = Array.isArray(graph?.nodes) ? graph.nodes : [];
  const positions = savedLayout?.positions || {};

  if (!nodes.length || !positions || typeof positions !== "object") {
    return false;
  }

  const positionedCount = nodes.filter((node) =>
    isValidPosition(positions[String(node.id)])
  ).length;

  return positionedCount >= Math.max(2, Math.floor(nodes.length * 0.65));
}

function restoreSavedViewport(cy, savedLayout) {
  if (!cy || !savedLayout) {
    return;
  }

  if (Number.isFinite(savedLayout.zoom) && isValidPosition(savedLayout.pan)) {
    cy.zoom(savedLayout.zoom);
    cy.pan(savedLayout.pan);
  } else {
    cy.fit(undefined, 58);
  }
}

function isValidPosition(value) {
  return Boolean(
    value &&
      typeof value === "object" &&
      Number.isFinite(Number(value.x)) &&
      Number.isFinite(Number(value.y))
  );
}

function colourForNodeType(type) {
  return NODE_TYPE_COLOURS[String(type || "").toLowerCase()] || NODE_TYPE_COLOURS.entity;
}

function shortLabel(value) {
  const text = String(value || "").trim();

  if (text.length <= 42) {
    return text;
  }

  return `${text.slice(0, 39).trim()}…`;
}