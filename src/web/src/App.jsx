import { useEffect, useMemo, useState } from "react";
import LandingPage from "./components/LandingPage.jsx";
import SignInPage from "./components/SignInPage.jsx";
import AnalystApp from "./components/AnalystApp.jsx";
import {
  clearSession,
  createDemoSession,
  getStoredSession,
  storeSession
} from "./api.js";

const LOCAL_DEMO_SESSION = {
  session_id: "local-demo-session",
  workspace: "demo",
  user: {
    name: "Demo analyst",
    email: "demo@lanthic.local"
  },
  run_id: "workspace-preview",
  corpus_id: "corpus-pending",
  branch_id: "workspace",
  mode: "demo"
};

function routeFromHash() {
  const value = window.location.hash.replace("#", "").replace("/", "");

  if (value === "signin") return "signin";
  if (value === "app") return "app";

  return "landing";
}

function normaliseSession(nextSession) {
  return {
    ...LOCAL_DEMO_SESSION,
    ...(nextSession || {}),
    user: {
      ...LOCAL_DEMO_SESSION.user,
      ...(nextSession?.user || {})
    }
  };
}

export default function App() {
  const [route, setRoute] = useState(routeFromHash);
  const [session, setSession] = useState(() => getStoredSession());

  useEffect(() => {
    const handleHash = () => setRoute(routeFromHash());

    window.addEventListener("hashchange", handleHash);
    return () => window.removeEventListener("hashchange", handleHash);
  }, []);

  const nav = useMemo(() => {
    function landing() {
      window.location.hash = "/";
    }

    function signIn() {
      window.location.hash = "/signin";
    }

    function openWorkspace(nextSession) {
      const safeSession = normaliseSession(nextSession);

      storeSession(safeSession);
      setSession(safeSession);
      window.location.hash = "/app";
    }

    async function openDemoWorkspace(nextSession) {
      if (nextSession && typeof nextSession === "object") {
        openWorkspace(nextSession);
        return;
      }

      try {
        const demoSession = await createDemoSession();
        openWorkspace(demoSession);
      } catch {
        openWorkspace(LOCAL_DEMO_SESSION);
      }
    }

    function signOut() {
      clearSession();
      setSession(null);
      window.location.hash = "/";
    }

    return {
      landing,
      signIn,
      openWorkspace,
      openDemoWorkspace,
      signOut
    };
  }, []);

  if (route === "signin") {
    return (
      <SignInPage
        onBack={nav.landing}
        onSignedIn={nav.openWorkspace}
        onSignIn={nav.openWorkspace}
        onSuccess={nav.openWorkspace}
        onEnter={nav.openWorkspace}
        onDemo={nav.openDemoWorkspace}
      />
    );
  }

  if (route === "app") {
    return (
      <AnalystApp
        session={session || LOCAL_DEMO_SESSION}
        onBack={nav.landing}
        onSignOut={nav.signOut}
      />
    );
  }

  return (
    <LandingPage
      onEnter={nav.signIn}
      onDemo={nav.openDemoWorkspace}
    />
  );
}