import { useMemo, useState } from "react";
import { FileText, Plus, SlidersHorizontal } from "lucide-react";

const DEFAULT_CURRENT = {
  title: "Will rare-earth supply constraints materially affect EV production in 2026?",
  status: "Answer ready",
  updatedAt: "Updated just now",
  documentCount: 0,
  turnCount: 0
};

const DEFAULT_RECENT = [
  {
    title: "REE export curbs from China: impact on Western supply",
    updatedAt: "2h ago",
    status: "Answer ready",
    documentCount: 0,
    turnCount: 1
  },
  {
    title: "NdPr price outlook 2025–2027",
    updatedAt: "Yesterday",
    status: "Ready",
    documentCount: 0,
    turnCount: 0
  },
  {
    title: "Australia heavy mineral sands project pipeline",
    updatedAt: "2 days ago",
    status: "Documents attached",
    documentCount: 2,
    turnCount: 0
  },
  {
    title: "Critical minerals policy shifts in the U.S.",
    updatedAt: "3 days ago",
    status: "Answer ready",
    documentCount: 1,
    turnCount: 2
  }
];

const STATUS_OPTIONS = [
  { value: "all", label: "All statuses" },
  { value: "ready", label: "Ready" },
  { value: "answer ready", label: "Answer ready" },
  { value: "documents attached", label: "Documents attached" },
  { value: "running", label: "Running" }
];

const DOCUMENT_OPTIONS = [
  { value: "all", label: "All documents" },
  { value: "has-documents", label: "Has uploaded documents" },
  { value: "no-documents", label: "No uploaded documents" }
];

const SORT_OPTIONS = [
  { value: "modified-desc", label: "Last modified" },
  { value: "created-desc", label: "Created date" },
  { value: "name-asc", label: "Name" },
  { value: "turns-desc", label: "Most turns" },
  { value: "documents-desc", label: "Most documents" }
];

export default function InvestigationSidebar({
  currentInvestigation = DEFAULT_CURRENT,
  recentInvestigations = DEFAULT_RECENT,
  selectedInvestigationId,
  onSelectInvestigation,
  onAddDocuments,
  onFilterInvestigations
}) {
  const [showFilters, setShowFilters] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [statusFilter, setStatusFilter] = useState("all");
  const [documentFilter, setDocumentFilter] = useState("all");
  const [sortMode, setSortMode] = useState("modified-desc");

  const filteredInvestigations = useMemo(() => {
    return filterAndSortInvestigations(recentInvestigations, {
      searchQuery,
      statusFilter,
      documentFilter,
      sortMode
    });
  }, [recentInvestigations, searchQuery, statusFilter, documentFilter, sortMode]);

  const currentDocumentCount = getDocumentCount(currentInvestigation);
  const currentTurnCount = getTurnCount(currentInvestigation);
  const hasActiveFilters =
    searchQuery.trim() ||
    statusFilter !== "all" ||
    documentFilter !== "all" ||
    sortMode !== "modified-desc";

  function handleToggleFilters() {
    const nextValue = !showFilters;
    setShowFilters(nextValue);
    onFilterInvestigations?.(nextValue);
  }

  function handleClearFilters() {
    setSearchQuery("");
    setStatusFilter("all");
    setDocumentFilter("all");
    setSortMode("modified-desc");
  }

  return (
    <aside className="investigation-sidebar">
      <div className="investigation-sidebar-header">
        <h1>Investigations</h1>

        <button
          type="button"
          aria-label="Investigation filters"
          aria-pressed={showFilters}
          onClick={handleToggleFilters}
        >
          <SlidersHorizontal size={18} />
        </button>
      </div>

      {showFilters ? (
        <section className="investigation-filter-panel" aria-label="Investigation filters">
          <label className="filter-field">
            <span>Search</span>
            <input
              type="search"
              value={searchQuery}
              placeholder="Search investigations"
              onChange={(event) => setSearchQuery(event.target.value)}
            />
          </label>

          <label className="filter-field">
            <span>Status</span>
            <select
              value={statusFilter}
              onChange={(event) => setStatusFilter(event.target.value)}
            >
              {STATUS_OPTIONS.map((option) => (
                <option value={option.value} key={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </label>

          <label className="filter-field">
            <span>Documents</span>
            <select
              value={documentFilter}
              onChange={(event) => setDocumentFilter(event.target.value)}
            >
              {DOCUMENT_OPTIONS.map((option) => (
                <option value={option.value} key={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </label>

          <label className="filter-field">
            <span>Sort</span>
            <select
              value={sortMode}
              onChange={(event) => setSortMode(event.target.value)}
            >
              {SORT_OPTIONS.map((option) => (
                <option value={option.value} key={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </label>

          <div className="filter-actions-row">
            <span>
              {filteredInvestigations.length} shown
            </span>

            <button
              className="filter-clear-button"
              type="button"
              onClick={handleClearFilters}
              disabled={!hasActiveFilters}
            >
              Clear filters
            </button>
          </div>
        </section>
      ) : null}

      <section className="current-investigation-block">
        <p className="workspace-kicker">Current investigation</p>

        <button
          className="current-investigation-card"
          type="button"
          onClick={() => onSelectInvestigation?.(currentInvestigation)}
        >
          <span className="small-facet-accent" />

          <strong>{getInvestigationTitle(currentInvestigation)}</strong>

          <span className="answer-status-pill">
            {getInvestigationStatus(currentInvestigation)}
            <span aria-hidden="true">✓</span>
          </span>

          <span className="current-investigation-meta">
            <span>{formatCountLabel(currentDocumentCount, "doc", "docs")}</span>
            <span>{formatCountLabel(currentTurnCount, "turn", "turns")}</span>
          </span>

          <span className="current-investigation-footline">
            <span>Last modified</span>
            <em>{getInvestigationUpdatedLabel(currentInvestigation)}</em>
          </span>
        </button>
      </section>

      <section className="recent-investigations-block">
        <p className="workspace-kicker">Recent investigations</p>

        <div className="recent-investigation-list">
          {filteredInvestigations.length ? (
            filteredInvestigations.map((item) => {
              const isSelected = isSelectedInvestigation(item, selectedInvestigationId);

              return (
                <button
                  type="button"
                  key={getInvestigationKey(item)}
                  className={isSelected ? "active" : ""}
                  aria-current={isSelected ? "true" : undefined}
                  onClick={() => onSelectInvestigation?.(item)}
                >
                  <FileText size={16} />

                  <span className="recent-investigation-copy">
                    <span className="recent-investigation-title">
                      {getInvestigationTitle(item)}
                    </span>

                    <span className="investigation-meta-row">
                      {getInvestigationStatus(item)}
                      {getDocumentCount(item) ? ` · ${getDocumentCount(item)} docs` : ""}
                      {getTurnCount(item) ? ` · ${getTurnCount(item)} turns` : ""}
                    </span>
                  </span>

                  <em>{getInvestigationUpdatedLabel(item)}</em>
                </button>
              );
            })
          ) : (
            <div className="investigation-list-empty">
              No matching investigations.
            </div>
          )}
        </div>
      </section>

      <button
        className="add-documents-card"
        type="button"
        onClick={onAddDocuments}
      >
        <span className="add-documents-icon">
          <Plus size={26} />
        </span>

        <strong>Add documents to this investigation</strong>

        <p>
          Upload reports, articles, and data to strengthen your analysis.
        </p>

        <span className="large-facet-accent" />
      </button>
    </aside>
  );
}

function filterAndSortInvestigations(items = [], filters) {
  const search = normaliseText(filters.searchQuery);

  const filtered = items.filter((item) => {
    const title = normaliseText(getInvestigationTitle(item));
    const status = normaliseText(getInvestigationStatus(item));
    const documentCount = getDocumentCount(item);

    const matchesSearch =
      !search ||
      title.includes(search) ||
      status.includes(search) ||
      normaliseText(item.chatName).includes(search);

    const matchesStatus =
      filters.statusFilter === "all" ||
      status === filters.statusFilter;

    const matchesDocuments =
      filters.documentFilter === "all" ||
      (filters.documentFilter === "has-documents" && documentCount > 0) ||
      (filters.documentFilter === "no-documents" && documentCount === 0);

    return matchesSearch && matchesStatus && matchesDocuments;
  });

  return filtered.sort((a, b) => compareInvestigations(a, b, filters.sortMode));
}

function compareInvestigations(a, b, sortMode) {
  if (sortMode === "created-desc") {
    return getDateValue(b.createdAt) - getDateValue(a.createdAt);
  }

  if (sortMode === "name-asc") {
    return getInvestigationTitle(a).localeCompare(getInvestigationTitle(b));
  }

  if (sortMode === "turns-desc") {
    return getTurnCount(b) - getTurnCount(a);
  }

  if (sortMode === "documents-desc") {
    return getDocumentCount(b) - getDocumentCount(a);
  }

  return getModifiedValue(b) - getModifiedValue(a);
}

function isSelectedInvestigation(item = {}, selectedInvestigationId) {
  if (!selectedInvestigationId) {
    return false;
  }

  return String(item.investigationId || item.id || "") === String(selectedInvestigationId);
}

function getInvestigationKey(item) {
  return (
    item.investigationId ||
    item.id ||
    `${getInvestigationTitle(item)}-${getInvestigationUpdatedLabel(item)}`
  );
}

function getInvestigationTitle(item = {}) {
  return item.title || item.chatName || "Untitled investigation";
}

function getInvestigationStatus(item = {}) {
  return item.status || "Ready";
}

function getInvestigationUpdatedLabel(item = {}) {
  return item.updatedAt || item.lastModifiedAt || "Just now";
}

function getDocumentCount(item = {}) {
  const value = item.documentCount ?? item.documents?.length ?? 0;
  const number = Number(value);
  return Number.isFinite(number) ? number : 0;
}

function getTurnCount(item = {}) {
  const value = item.turnCount ?? item.turns?.length ?? 0;
  const number = Number(value);
  return Number.isFinite(number) ? number : 0;
}

function getModifiedValue(item = {}) {
  return getDateValue(item.lastModifiedAt || item.updatedAt || item.createdAt);
}

function getDateValue(value) {
  if (!value) {
    return 0;
  }

  const parsed = Date.parse(value);

  if (Number.isFinite(parsed)) {
    return parsed;
  }

  return 0;
}

function formatCountLabel(value, singular, plural) {
  const count = Number(value);

  if (!Number.isFinite(count) || count === 0) {
    return `0 ${plural}`;
  }

  return `${count} ${count === 1 ? singular : plural}`;
}

function normaliseText(value) {
  return String(value || "")
    .trim()
    .toLowerCase();
}