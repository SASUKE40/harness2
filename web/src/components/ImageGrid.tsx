import { useState } from "react";
import { imageSrc, pct, stageOrBase, type TaskMetrics } from "../metrics";

function compare(truth: string[], got: string[]) {
  const remaining = [...truth];
  const chips: { label: string; state: "hit" | "extra" | "miss" }[] = [];
  for (const g of got) {
    const i = remaining.indexOf(g);
    if (i >= 0) {
      remaining.splice(i, 1);
      chips.push({ label: g, state: "hit" });
    } else chips.push({ label: g, state: "extra" });
  }
  remaining.forEach((r) => chips.push({ label: r, state: "miss" }));
  return chips;
}

export function ImageGrid({ metrics, inspected }: { metrics: TaskMetrics; inspected: string }) {
  const [showBase, setShowBase] = useState(true);
  const { task, record } = metrics;
  const rollout = stageOrBase(task, record, inspected);
  const base = task.base[0];
  return (
    <div className="card images">
      <div className="card-head">
        <div>
          <h2>
            What the harness sees · <span className="mono">{inspected}</span>
          </h2>
          <p className="muted">
            {rollout ? `${pct(rollout.accuracy)} of images exactly right · micro F1 ${rollout.f1.toFixed(2)}` : "No output"}
            {" · "}images as Jev received them
          </p>
        </div>
        <label className="toggle">
          <input type="checkbox" checked={showBase} onChange={(e) => setShowBase(e.target.checked)} />
          <span />
          Compare with base
        </label>
      </div>
      <div className="img-grid">
        {task.images.map((img) => {
          const got = rollout?.labels?.[img.name] ?? [];
          const ok = rollout?.per_image?.[img.name];
          const baseGot = base?.labels?.[img.name] ?? [];
          return (
            <figure key={img.name} className={`img-card ${ok ? "ok" : "ko"}`}>
              <div className="img-wrap">
                <img src={imageSrc(img)} alt={img.name} />
                <span className={`verdict ${ok ? "ok" : "ko"}`}>{ok ? "✓" : "✗"}</span>
                <span className="img-name">{img.name}</span>
              </div>
              <figcaption>
                <Row title="truth" chips={img.truth.map((label) => ({ label, state: "truth" as const }))} />
                <Row title={inspected} chips={compare(img.truth, got)} />
                {showBase && inspected !== base?.label && (
                  <Row title="base" chips={compare(img.truth, baseGot)} faint />
                )}
              </figcaption>
            </figure>
          );
        })}
      </div>
    </div>
  );
}

function Row({
  title,
  chips,
  faint,
}: {
  title: string;
  chips: { label: string; state: "hit" | "extra" | "miss" | "truth" }[];
  faint?: boolean;
}) {
  return (
    <div className={`chip-row ${faint ? "faint" : ""}`}>
      <span className="chip-title">{title}</span>
      <div className="chips">
        {chips.length ? (
          chips.map((c, i) => (
            <span key={i} className={`chip ${c.state}`}>
              {c.label}
            </span>
          ))
        ) : (
          <span className="chip none">none</span>
        )}
      </div>
    </div>
  );
}
