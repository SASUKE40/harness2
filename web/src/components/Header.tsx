import type { Bundle, Mode } from "../types";

const MODE_LABEL: Record<Mode, [string, string]> = {
  agg: ["Parallel", "width"],
  seq: ["Sequential", "depth"],
};

export function Header({
  bundle,
  mode,
  setMode,
  playing,
  onReplay,
  onLive,
}: {
  bundle: Bundle;
  mode: Mode;
  setMode: (m: Mode) => void;
  playing: boolean;
  onReplay: () => void;
  onLive: () => void;
}) {
  const jev = bundle.meta.jev_mode;
  return (
    <header className="topbar">
      <div className="brand">
        <img src="/favicon.svg" alt="" width={30} height={30} />
        <div>
          <strong>
            Harness<sup>2</sup> <span className="x">×</span> Jev
          </strong>
          <span>RSI observatory</span>
        </div>
      </div>
      <div className="topbar-right">
        <span className={`badge jev-${jev}`} title="Final-output selector">
          <span className="dot" />
          {jev === "offline" ? "Jev · offline stand-in" : jev === "live" ? `Jev · ${bundle.meta.jev_model}` : "Agent selector"}
        </span>
        <span className="badge">k = {bundle.meta.k}</span>
        <div className="segmented" role="tablist" aria-label="Recursion mode">
          {bundle.meta.modes.map((m) => (
            <button key={m} role="tab" aria-selected={mode === m} className={mode === m ? "on" : ""} onClick={() => setMode(m)}>
              {MODE_LABEL[m][0]} <small>{MODE_LABEL[m][1]}</small>
            </button>
          ))}
        </div>
        <button className="btn" onClick={onLive}>
          ◉ Live demo
        </button>
        <button className="btn primary" onClick={onReplay}>
          {playing ? "❚❚ Pause" : "▶ Replay stream"}
        </button>
      </div>
    </header>
  );
}
