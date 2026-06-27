import { useState } from "react";
import { ArrowLeft, ArrowRight, Lock, Mail } from "lucide-react";
import { createDemoSession } from "../api.js";
import markUrl from "../assets/lanthic-mark.svg";

export default function SignInPage({
  onBack,
  onEnter,
  onSignIn,
  onSuccess,
  onDemo,
  onPricing
}) {
  const [email, setEmail] = useState("analyst@lanthic.local");
  const [password, setPassword] = useState("demo");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  function routeHome() {
    if (typeof onBack === "function") {
      onBack();
      return;
    }

    window.location.hash = "/";
  }

  function routePricing() {
    if (typeof onPricing === "function") {
      onPricing();
      return;
    }

    window.location.hash = "/pricing";
  }

  function enterWorkspace(session) {
    if (typeof onSignIn === "function") {
      onSignIn(session);
      return;
    }

    if (typeof onSuccess === "function") {
      onSuccess(session);
      return;
    }

    if (typeof onEnter === "function") {
      onEnter(session);
      return;
    }

    window.location.hash = "/app";
  }

  function enterDemo(session) {
    if (typeof onDemo === "function") {
      onDemo(session);
      return;
    }

    if (typeof onSuccess === "function") {
      onSuccess(session);
      return;
    }

    if (typeof onEnter === "function") {
      onEnter(session);
      return;
    }

    window.location.hash = "/app";
  }

  function handleSubmit(event) {
    event.preventDefault();
    setError("");

    const trimmedEmail = email.trim();

    if (!trimmedEmail || !password.trim()) {
      setError("Enter an email and password to continue.");
      return;
    }

    enterWorkspace({
      user: {
        name: "Analyst",
        email: trimmedEmail
      },
      mode: "signed-in"
    });
  }

  async function handleDemo() {
    setError("");
    setBusy(true);

    try {
      const session = await createDemoSession();
      enterDemo(session);
    } catch {
      enterDemo({
        user: {
          name: "Demo analyst",
          email: "demo@lanthic.local"
        },
        mode: "demo"
      });
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="page-shell signin-page">
      <div className="grid-field signin-grid-field" />

      <button className="back-button" onClick={routeHome} type="button">
        <ArrowLeft size={17} />
        Back
      </button>

      <section className="signin-layout compact-signin-layout">
        <div className="signin-mineral-panel compact-signin-panel">
          <div className="signin-panel-inner">
            <p className="eyebrow">Lanthic Intelligence</p>

            <h1>Return to the intelligence workspace.</h1>

            <p>
              Review the maintained source base, inspect grounded reasoning,
              track forward-looking signals, and continue briefing work from a
              focused analyst surface.
            </p>
          </div>
        </div>

        <section className="signin-card compact-signin-card" aria-label="Sign in">
          <div className="signin-card-header">
            <img src={markUrl} alt="" />

            <div>
              <p className="eyebrow">Secure workspace</p>
              <h2>Sign in</h2>
            </div>
          </div>

          <form className="signin-form" onSubmit={handleSubmit}>
            <label>
              Email
              <div className="input-wrap">
                <Mail size={17} />
                <input
                  type="email"
                  autoComplete="email"
                  value={email}
                  onChange={(event) => setEmail(event.target.value)}
                  placeholder="analyst@example.com"
                />
              </div>
            </label>

            <label>
              Password
              <div className="input-wrap">
                <Lock size={17} />
                <input
                  type="password"
                  autoComplete="current-password"
                  value={password}
                  onChange={(event) => setPassword(event.target.value)}
                  placeholder="••••••••"
                />
              </div>
            </label>

            {error ? <div className="form-error">{error}</div> : null}

            <button className="primary-button full-width" type="submit">
              Sign in
              <ArrowRight size={18} />
            </button>

            <button
              className="secondary-button full-width"
              type="button"
              onClick={handleDemo}
              disabled={busy}
            >
              {busy ? "Opening demo..." : "Open demo workspace"}
            </button>
          </form>

          <div className="signin-pricing-link">
            <span>Don’t have an account?</span>
            <button type="button" onClick={routePricing}>
              View pricing options
            </button>
          </div>

          <p className="signin-footnote">
            Demo access opens a prepared analyst session. Sign-in fields are
            provided for the prototype flow.
          </p>
        </section>
      </section>
    </main>
  );
}