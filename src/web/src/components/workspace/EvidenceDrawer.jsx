import { Bookmark, Pin, X } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

const TABS = ["Sources", "Evidence", "Assumptions"];

const LABEL_BY_KIND = {
  sources: "Source",
  evidence: "Evidence",
  assumptions: "Assumption",
  graphItems: "Graph item"
};

const PLURAL_LABEL_BY_KIND = {
  sources: "sources",
  evidence: "evidence items",
  assumptions: "assumptions",
  graphItems: "graph items"
};

const EMPTY_STATE = {
  activeTab: "Sources",
  showPinnedOnly: false,
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

export default function EvidenceDrawer({
  sources = [],
  evidence = [],
  assumptions = [],
  workspaceState = EMPTY_STATE,
  onWorkspaceStateChange,
  onClose
}) {
  const safeState = useMemo(
    () => normaliseWorkspaceState(workspaceState),
    [workspaceState]
  );

  const [activeTab, setActiveTab] = useState(safeState.activeTab);

  useEffect(() => {
    setActiveTab(safeState.activeTab);
  }, [safeState.activeTab]);

  const selectedItem = safeState.selectedDrawerItem;

  function updateWorkspaceState(patch) {
    const nextState = normaliseWorkspaceState({
      ...safeState,
      ...patch,
      bookmarks: patch.bookmarks || safeState.bookmarks,
      pins: patch.pins || safeState.pins
    });

    onWorkspaceStateChange?.(nextState);
  }

  function handleTabChange(tab) {
    setActiveTab(tab);
    updateWorkspaceState({ activeTab: tab });
  }

  function togglePinnedOnly() {
    updateWorkspaceState({
      showPinnedOnly: !safeState.showPinnedOnly
    });
  }

  function selectItem(kind, key, item) {
    updateWorkspaceState({
      selectedDrawerItem: toDrawerSelection(kind, key, item)
    });
  }

  function clearSelectedItem() {
    updateWorkspaceState({
      selectedDrawerItem: null
    });
  }

  function toggleBookmark(kind, key) {
    updateWorkspaceState({
      bookmarks: toggleInBucket(safeState.bookmarks, kind, key)
    });
  }

  function togglePin(kind, key) {
    updateWorkspaceState({
      pins: toggleInBucket(safeState.pins, kind, key)
    });
  }

  const tabCounts = {
    Sources: sources.length,
    Evidence: evidence.length,
    Assumptions: assumptions.length
  };

  return (
    <aside className="evidence-drawer">
      <div className="evidence-drawer-header">
        <h1>Evidence drawer</h1>

        <div>
          <button
            className={safeState.showPinnedOnly ? "active" : ""}
            type="button"
            aria-label={safeState.showPinnedOnly ? "Show all drawer items" : "Show pinned drawer items"}
            title={safeState.showPinnedOnly ? "Show all items" : "Show pinned only"}
            onClick={togglePinnedOnly}
          >
            <Pin size={17} />
          </button>

          <button type="button" aria-label="Close evidence drawer" onClick={onClose}>
            <X size={18} />
          </button>
        </div>
      </div>

      {safeState.showPinnedOnly ? (
        <div className="drawer-mode-banner">
          <span>Showing pinned items only.</span>
          <button type="button" onClick={togglePinnedOnly}>
            Show all
          </button>
        </div>
      ) : null}

      <div className="evidence-drawer-tabs">
        {TABS.map((tab) => (
          <button
            type="button"
            key={tab}
            className={activeTab === tab ? "active" : ""}
            onClick={() => handleTabChange(tab)}
          >
            {tab}
            <span>{tabCounts[tab]}</span>
          </button>
        ))}
      </div>

      <div className="evidence-drawer-content">
        {selectedItem ? (
          <DrawerDetail
            selectedItem={selectedItem}
            bookmarks={safeState.bookmarks}
            pins={safeState.pins}
            onClose={clearSelectedItem}
            onToggleBookmark={toggleBookmark}
            onTogglePin={togglePin}
          />
        ) : null}

        {activeTab === "Sources" ? (
          <SourcesTab
            sources={sources}
            state={safeState}
            onSelect={selectItem}
            onToggleBookmark={toggleBookmark}
            onTogglePin={togglePin}
          />
        ) : null}

        {activeTab === "Evidence" ? (
          <EvidenceTab
            evidence={evidence}
            state={safeState}
            onSelect={selectItem}
            onToggleBookmark={toggleBookmark}
            onTogglePin={togglePin}
          />
        ) : null}

        {activeTab === "Assumptions" ? (
          <AssumptionsTab
            assumptions={assumptions}
            state={safeState}
            onSelect={selectItem}
            onToggleBookmark={toggleBookmark}
            onTogglePin={togglePin}
          />
        ) : null}
      </div>
    </aside>
  );
}

function SourcesTab({
  sources,
  state,
  onSelect,
  onToggleBookmark,
  onTogglePin
}) {
  if (!sources.length) {
    return <p className="drawer-empty">No sources are attached yet.</p>;
  }

  const { pinnedItems, regularItems } = splitPinnedItems("sources", sources, state);

  return (
    <DrawerCardList
      kind="sources"
      pinnedItems={pinnedItems}
      regularItems={regularItems}
      showPinnedOnly={state.showPinnedOnly}
      renderItem={(entry, pinned) => (
        <SourceCard
          key={entry.key}
          entry={entry}
          pinned={pinned}
          state={state}
          onSelect={onSelect}
          onToggleBookmark={onToggleBookmark}
          onTogglePin={onTogglePin}
        />
      )}
    />
  );
}

function EvidenceTab({
  evidence,
  state,
  onSelect,
  onToggleBookmark,
  onTogglePin
}) {
  if (!evidence.length) {
    return <p className="drawer-empty">No evidence excerpts are attached yet.</p>;
  }

  const { pinnedItems, regularItems } = splitPinnedItems("evidence", evidence, state);

  return (
    <DrawerCardList
      kind="evidence"
      pinnedItems={pinnedItems}
      regularItems={regularItems}
      showPinnedOnly={state.showPinnedOnly}
      renderItem={(entry, pinned) => (
        <EvidenceCard
          key={entry.key}
          entry={entry}
          pinned={pinned}
          state={state}
          onSelect={onSelect}
          onToggleBookmark={onToggleBookmark}
          onTogglePin={onTogglePin}
        />
      )}
    />
  );
}

function AssumptionsTab({
  assumptions,
  state,
  onSelect,
  onToggleBookmark,
  onTogglePin
}) {
  if (!assumptions.length) {
    return <p className="drawer-empty">No explicit assumptions have been recorded.</p>;
  }

  const { pinnedItems, regularItems } = splitPinnedItems("assumptions", assumptions, state);

  return (
    <DrawerCardList
      kind="assumptions"
      pinnedItems={pinnedItems}
      regularItems={regularItems}
      showPinnedOnly={state.showPinnedOnly}
      renderItem={(entry, pinned) => (
        <AssumptionCard
          key={entry.key}
          entry={entry}
          pinned={pinned}
          state={state}
          onSelect={onSelect}
          onToggleBookmark={onToggleBookmark}
          onTogglePin={onTogglePin}
        />
      )}
    />
  );
}

function DrawerCardList({
  kind,
  pinnedItems,
  regularItems,
  showPinnedOnly,
  renderItem
}) {
  const plural = PLURAL_LABEL_BY_KIND[kind] || "items";

  if (showPinnedOnly && !pinnedItems.length) {
    return (
      <p className="drawer-empty">
        No pinned {plural} in this tab.
      </p>
    );
  }

  return (
    <div className="drawer-card-list">
      {pinnedItems.length ? (
        <div className="drawer-section-group">
          <span className="drawer-section-label">Pinned {plural}</span>
          {pinnedItems.map((entry) => renderItem(entry, true))}
        </div>
      ) : null}

      {!showPinnedOnly ? (
        <div className="drawer-section-group">
          {pinnedItems.length ? (
            <span className="drawer-section-label">All {plural}</span>
          ) : null}

          {regularItems.map((entry) => renderItem(entry, false))}
        </div>
      ) : null}
    </div>
  );
}

function SourceCard({
  entry,
  pinned,
  state,
  onSelect,
  onToggleBookmark,
  onTogglePin
}) {
  const source = entry.item;
  const bookmarked = includesKey(state.bookmarks.sources, entry.key);
  const isPinned = pinned || includesKey(state.pins.sources, entry.key);
  const selected = isSelected(state.selectedDrawerItem, "sources", entry.key);

  return (
    <article
      className={`source-drawer-card ${selected ? "selected" : ""}`}
      role="button"
      tabIndex={0}
      onClick={() => onSelect("sources", entry.key, source)}
      onKeyDown={(event) => handleKeyboardSelect(event, () => onSelect("sources", entry.key, source))}
    >
      <div className="source-card-topline">
        <span className="source-logo-tile">{source.logo || initials(source.title)}</span>

        <div>
          <strong>{source.title}</strong>
          <em>{source.date}</em>
        </div>

        <ItemActions
          kind="sources"
          itemKey={entry.key}
          bookmarked={bookmarked}
          pinned={isPinned}
          onToggleBookmark={onToggleBookmark}
          onTogglePin={onTogglePin}
        />
      </div>

      <p>{source.excerpt}</p>

      <ItemBadges bookmarked={bookmarked} pinned={isPinned} tag={source.tag} />
    </article>
  );
}

function EvidenceCard({
  entry,
  pinned,
  state,
  onSelect,
  onToggleBookmark,
  onTogglePin
}) {
  const item = entry.item;
  const bookmarked = includesKey(state.bookmarks.evidence, entry.key);
  const isPinned = pinned || includesKey(state.pins.evidence, entry.key);
  const selected = isSelected(state.selectedDrawerItem, "evidence", entry.key);

  return (
    <article
      className={`evidence-drawer-card ${selected ? "selected" : ""}`}
      role="button"
      tabIndex={0}
      onClick={() => onSelect("evidence", entry.key, item)}
      onKeyDown={(event) => handleKeyboardSelect(event, () => onSelect("evidence", entry.key, item))}
    >
      <div className="drawer-card-heading-row">
        <strong>{item.title || item.source}</strong>

        <ItemActions
          kind="evidence"
          itemKey={entry.key}
          bookmarked={bookmarked}
          pinned={isPinned}
          onToggleBookmark={onToggleBookmark}
          onTogglePin={onTogglePin}
        />
      </div>

      <p>{item.text}</p>
      <em>{item.support || "Supporting evidence"}</em>

      <ItemBadges bookmarked={bookmarked} pinned={isPinned} />
    </article>
  );
}

function AssumptionCard({
  entry,
  pinned,
  state,
  onSelect,
  onToggleBookmark,
  onTogglePin
}) {
  const item = entry.item;
  const bookmarked = includesKey(state.bookmarks.assumptions, entry.key);
  const isPinned = pinned || includesKey(state.pins.assumptions, entry.key);
  const selected = isSelected(state.selectedDrawerItem, "assumptions", entry.key);

  return (
    <article
      className={`assumption-drawer-card ${selected ? "selected" : ""}`}
      role="button"
      tabIndex={0}
      onClick={() => onSelect("assumptions", entry.key, item)}
      onKeyDown={(event) => handleKeyboardSelect(event, () => onSelect("assumptions", entry.key, item))}
    >
      <div className="drawer-card-heading-row">
        <strong>{item.title}</strong>

        <ItemActions
          kind="assumptions"
          itemKey={entry.key}
          bookmarked={bookmarked}
          pinned={isPinned}
          onToggleBookmark={onToggleBookmark}
          onTogglePin={onTogglePin}
        />
      </div>

      <p>{item.text}</p>

      <ItemBadges bookmarked={bookmarked} pinned={isPinned} />
    </article>
  );
}

function ItemActions({
  kind,
  itemKey,
  bookmarked,
  pinned,
  onToggleBookmark,
  onTogglePin
}) {
  return (
    <span className="drawer-item-actions">
      <button
        className={`drawer-item-action ${bookmarked ? "active" : ""}`}
        type="button"
        title={bookmarked ? "Remove bookmark" : "Bookmark"}
        aria-label={bookmarked ? "Remove bookmark" : "Bookmark item"}
        onClick={(event) => {
          event.stopPropagation();
          onToggleBookmark(kind, itemKey);
        }}
      >
        <Bookmark size={15} />
      </button>

      <button
        className={`drawer-item-action ${pinned ? "active" : ""}`}
        type="button"
        title={pinned ? "Unpin" : "Pin to top"}
        aria-label={pinned ? "Unpin item" : "Pin item"}
        onClick={(event) => {
          event.stopPropagation();
          onTogglePin(kind, itemKey);
        }}
      >
        <Pin size={15} />
      </button>
    </span>
  );
}

function ItemBadges({
  bookmarked,
  pinned,
  tag
}) {
  if (!bookmarked && !pinned && !tag) {
    return null;
  }

  return (
    <div className="drawer-item-badges">
      {tag ? <span>{tag}</span> : null}
      {bookmarked ? <span>Bookmarked</span> : null}
      {pinned ? <span>Pinned</span> : null}
    </div>
  );
}

function DrawerDetail({
  selectedItem,
  bookmarks,
  pins,
  onClose,
  onToggleBookmark,
  onTogglePin
}) {
  const kind = selectedItem.kind || "evidence";
  const itemKey = selectedItem.key || "";
  const bookmarked = includesKey(bookmarks[kind], itemKey);
  const pinned = includesKey(pins[kind], itemKey);

  return (
    <section className="drawer-detail-card">
      <div className="drawer-detail-topline">
        <span>{LABEL_BY_KIND[kind] || "Evidence item"}</span>

        <button type="button" onClick={onClose} aria-label="Close item detail">
          <X size={15} />
        </button>
      </div>

      <h2>{selectedItem.title}</h2>

      {selectedItem.subtitle ? <em>{selectedItem.subtitle}</em> : null}

      {selectedItem.text ? <p>{selectedItem.text}</p> : null}

      <div className="drawer-detail-actions">
        <button
          className={bookmarked ? "active" : ""}
          type="button"
          onClick={() => onToggleBookmark(kind, itemKey)}
        >
          <Bookmark size={15} />
          {bookmarked ? "Bookmarked" : "Bookmark"}
        </button>

        <button
          className={pinned ? "active" : ""}
          type="button"
          onClick={() => onTogglePin(kind, itemKey)}
        >
          <Pin size={15} />
          {pinned ? "Pinned" : "Pin to top"}
        </button>
      </div>
    </section>
  );
}

function splitPinnedItems(kind, items, state) {
  const pinnedKeys = new Set(state.pins[kind] || []);
  const entries = items.map((item, index) => ({
    item,
    index,
    key: getItemKey(kind, item, index)
  }));

  return {
    pinnedItems: entries.filter((entry) => pinnedKeys.has(entry.key)),
    regularItems: entries.filter((entry) => !pinnedKeys.has(entry.key))
  };
}

function toDrawerSelection(kind, key, item = {}) {
  if (kind === "sources") {
    return {
      kind,
      key,
      title: item.title || "Untitled source",
      subtitle: [item.date, item.tag].filter(Boolean).join(" · "),
      text: item.excerpt || "No source excerpt available.",
      item
    };
  }

  if (kind === "assumptions") {
    return {
      kind,
      key,
      title: item.title || "Assumption",
      subtitle: "Analyst caution",
      text: item.text || "No assumption text available.",
      item
    };
  }

  return {
    kind,
    key,
    title: item.title || item.source || "Evidence excerpt",
    subtitle: [item.source, item.date, item.support].filter(Boolean).join(" · "),
    text: item.text || "No evidence text available.",
    item
  };
}

function toggleInBucket(bucketSource, kind, key) {
  const source = normaliseBucketState(bucketSource);
  const existing = source[kind] || [];
  const next = includesKey(existing, key)
    ? existing.filter((item) => item !== key)
    : [...existing, key];

  return {
    ...source,
    [kind]: next
  };
}

function normaliseWorkspaceState(value = {}) {
  const source = value && typeof value === "object" ? value : {};

  return {
    activeTab: TABS.includes(source.activeTab) ? source.activeTab : "Sources",
    showPinnedOnly: Boolean(source.showPinnedOnly),
    bookmarks: normaliseBucketState(source.bookmarks),
    pins: normaliseBucketState(source.pins),
    selectedDrawerItem:
      source.selectedDrawerItem && typeof source.selectedDrawerItem === "object"
        ? source.selectedDrawerItem
        : null
  };
}

function normaliseBucketState(value = {}) {
  const source = value && typeof value === "object" ? value : {};

  return {
    sources: normaliseStringList(source.sources),
    evidence: normaliseStringList(source.evidence),
    assumptions: normaliseStringList(source.assumptions),
    graphItems: normaliseStringList(source.graphItems)
  };
}

function normaliseStringList(value) {
  if (!Array.isArray(value)) {
    return [];
  }

  const seen = new Set();

  return value
    .map((item) => String(item || "").trim())
    .filter((item) => {
      if (!item || seen.has(item)) {
        return false;
      }

      seen.add(item);
      return true;
    });
}

function getItemKey(kind, item = {}, index = 0) {
  return String(
    item.id ||
      item.documentId ||
      item.sourceId ||
      item.title ||
      item.source ||
      item.text ||
      `${kind}_${index}`
  );
}

function includesKey(items = [], key) {
  return Array.isArray(items) && items.includes(key);
}

function isSelected(selectedItem, kind, key) {
  return selectedItem?.kind === kind && selectedItem?.key === key;
}

function handleKeyboardSelect(event, callback) {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    callback();
  }
}

function initials(value = "") {
  const parts = value
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2);

  if (!parts.length) {
    return "S";
  }

  return parts.map((part) => part[0]).join("").toUpperCase();
}