import { imageSrc, pct, type TaskMetrics } from "../metrics";

export function TaskRail({
  metrics,
  selected,
  onSelect,
}: {
  metrics: TaskMetrics[];
  selected: number;
  onSelect: (i: number) => void;
}) {
  return (
    <section className="rail" aria-label="Task stream">
      {metrics.map((m, i) => (
        <button key={m.task.id} className={`rail-item ${i === selected ? "on" : ""}`} onClick={() => onSelect(i)}>
          <div className="rail-top">
            <span className="rail-step">T{i + 1}</span>
            <span className="rail-title">{m.task.title}</span>
            <span className={`pill ${m.selector === "jev" ? "pill-jev" : "pill-agent"}`}>
              {m.selector === "jev" ? "Jev" : "Agent"}
            </span>
          </div>
          <div className="rail-thumbs">
            {m.task.images.slice(0, 4).map((img) => (
              <img key={img.name} src={imageSrc(img)} alt={img.name} />
            ))}
          </div>
          <div className="rail-stats">
            <span>
              <small>base</small> {pct(m.base)}
            </span>
            <span className="arrow">→</span>
            <span className="good">
              <small>adopted</small> {pct(m.adopted)}
            </span>
            <span className="lib">
              <small>library</small> {m.libraryBefore}
            </span>
          </div>
          <div className="rail-progress">
            <i style={{ width: pct(m.adopted) }} />
            <b style={{ width: pct(m.base) }} />
          </div>
          {i < metrics.length - 1 && <span className="rail-link" aria-hidden />}
        </button>
      ))}
    </section>
  );
}
