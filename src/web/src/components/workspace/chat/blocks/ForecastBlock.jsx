import ForecastCheck from "../../ForecastCheck.jsx";

export default function ForecastBlock({ block }) {
  const data = block.data || {};

  return (
    <section className="chat-analysis-block forecast-block">
      <ForecastCheck
        status={data.status}
        summary={data.summary}
        showChart={data.showChart || data.show_chart}
      />
    </section>
  );
}