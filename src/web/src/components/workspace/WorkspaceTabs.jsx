export default function WorkspaceTabs({
  activeTab,
  onChange,
  tabs = []
}) {
  return (
    <div className="workspace-tab-bar" role="tablist" aria-label="Workspace view">
      {tabs.map((tab) => (
        <button
          key={tab.id}
          className={activeTab === tab.id ? "active" : ""}
          type="button"
          role="tab"
          aria-selected={activeTab === tab.id}
          onClick={() => onChange?.(tab.id)}
        >
          <span>{tab.label}</span>
          {tab.count !== undefined ? <em>{tab.count}</em> : null}
        </button>
      ))}
    </div>
  );
}