const API_BASE =
  import.meta.env.VITE_LANTHIC_API_BASE || "http://localhost:8000";

const SESSION_KEY = "lanthic_session";

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: {
      ...(options.body instanceof FormData ? {} : { "Content-Type": "application/json" }),
      ...(options.headers || {}),
    },
  });

  if (!response.ok) {
    let message = `Request failed: ${response.status}`;

    try {
      const payload = await response.json();
      message = payload.detail || payload.message || message;
    } catch {
      try {
        message = await response.text();
      } catch {
        // keep fallback
      }
    }

    throw new Error(message);
  }

  const contentType = response.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    return response.json();
  }

  return response.text();
}

export function getStoredSession() {
  try {
    const raw = window.localStorage.getItem(SESSION_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

export function storeSession(session) {
  try {
    window.localStorage.setItem(SESSION_KEY, JSON.stringify(session));
  } catch {
    // ignore storage failures
  }
}

export function clearSession() {
  try {
    window.localStorage.removeItem(SESSION_KEY);
  } catch {
    // ignore storage failures
  }
}

export async function createDemoSession(options = {}) {
  return request("/api/session/create", {
    method: "POST",
    body: JSON.stringify({
      workspace: "demo",
      ...options,
    }),
  });
}

export async function signIn(email, password) {
  return request("/api/auth/sign-in", {
    method: "POST",
    body: JSON.stringify({
      email,
      password,
      workspace: "demo",
    }),
  });
}

export async function listInvestigations() {
  return request("/api/investigations");
}

export async function getInvestigation(investigationId) {
  return request(`/api/investigations/${encodeURIComponent(investigationId)}`);
}

export async function createInvestigation({
  question,
  title,
  runId,
  corpusId,
  branchId,
} = {}) {
  return request("/api/investigations", {
    method: "POST",
    body: JSON.stringify({
      question,
      title,
      run_id: runId,
      corpus_id: corpusId,
      branch_id: branchId,
    }),
  });
}

export async function runInvestigationTurn(
  investigationId,
  {
    question,
    runId,
    corpusId,
    branchId,
    selectedGraphContext = []
  }
) {
  return request(`/api/investigations/${encodeURIComponent(investigationId)}/turns`, {
    method: "POST",
    body: JSON.stringify({
      question,
      run_id: runId,
      corpus_id: corpusId,
      branch_id: branchId,
      selected_graph_context: selectedGraphContext
    }),
  });
}

export async function addDocumentsToInvestigation(investigationId, files) {
  const formData = new FormData();

  Array.from(files || []).forEach((file) => {
    formData.append("files", file);
  });

  return request(`/api/investigations/${encodeURIComponent(investigationId)}/documents`, {
    method: "POST",
    body: formData,
  });
}

export async function getWorkspaceState(investigationId) {
  return request(`/api/investigations/${encodeURIComponent(investigationId)}/workspace-state`);
}

export async function saveWorkspaceState(investigationId, state) {
  return request(`/api/investigations/${encodeURIComponent(investigationId)}/workspace-state`, {
    method: "POST",
    body: JSON.stringify(state || {}),
  });
}

export async function exportInvestigation(investigationId) {
  return request(`/api/investigations/${encodeURIComponent(investigationId)}/export`);
}

export function downloadTextFile(filename, text, mimeType = "text/markdown") {
  const blob = new Blob([text], { type: mimeType });
  const url = window.URL.createObjectURL(blob);
  const anchor = document.createElement("a");

  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();

  window.URL.revokeObjectURL(url);
}

export async function getInvestigationSubgraph(investigationId) {
  return request(`/api/investigations/${encodeURIComponent(investigationId)}/subgraph`);
}