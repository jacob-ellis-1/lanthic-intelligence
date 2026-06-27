import { TrendingUp } from "lucide-react";

export default function ForecastCheck({
  status = "Possible",
  summary,
  showChart = true
}) {
  if (!summary) {
    return null;
  }

  return (
    <section className="forecast-check-card">
      <div className="forecast-card-heading">
        <span className="small-section-icon">
          <TrendingUp size={17} />
        </span>

        <h2>Forecast check</h2>
      </div>

      <p>
        Forecastability: <strong>{status}</strong>
      </p>

      <span>{summary}</span>

      {showChart ? <MiniForecastChart /> : null}
    </section>
  );
}

function MiniForecastChart() {
  return (
    <svg className="mini-forecast-chart" viewBox="0 0 160 78" role="presentation">
      <path
        className="mini-chart-grid"
        d="M0 20 H160 M0 40 H160 M0 60 H160 M40 0 V78 M80 0 V78 M120 0 V78"
      />
      <path
        className="mini-chart-band"
        d="M8 58 C30 42, 48 46, 68 35 C91 22, 112 30, 151 12 L151 70 C112 60, 91 54, 68 60 C48 66, 30 62, 8 72 Z"
      />
      <path
        className="mini-chart-line"
        d="M8 58 C30 42, 48 46, 68 35 C91 22, 112 30, 151 12"
      />
      <path
        className="mini-chart-dotted"
        d="M8 68 C35 62, 62 61, 91 47 C112 36, 133 30, 151 24"
      />
    </svg>
  );
}