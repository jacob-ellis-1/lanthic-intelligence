import markUrl from "../assets/lanthic-mark.svg";

export default function Logo({ size = "default", lockup = true }) {
  return (
    <div className={`logo logo-${size}`}>
      <img src={markUrl} alt="" className="logo-mark" />
      {lockup ? (
        <div className="logo-text">
          <span className="logo-name">Lanthic</span>
          <span className="logo-subname">Intelligence</span>
        </div>
      ) : null}
    </div>
  );
}