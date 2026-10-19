import type { TaskMetrics } from "../metrics";
import type { Bundle } from "../types";

export function KnowledgeMatrix({
  bundle,
  metrics,
  selected,
  onSelect,
}: {
  bundle: Bundle;
  metrics: TaskMetrics[];
  selected: number;
  onSelect: (i: number) => void;
}) {
  const fixes = Object.entries(bundle.fixes).filter(([, f]) => f.good && f.warning);
  return (
    <div className="card matrix">
      <div className="card-head">
        <div>
          <h2>Improvement harness</h2>
          <p className="muted">Harnessing-level recursion: what the library knows before each task</p>
        </div>
      </div>
      <table className="kmatrix">
        <thead>
          <tr>
            <th />
            {metrics.map((m, i) => (
              <th key={m.task.id} className={i === selected ? "sel" : ""} onClick={() => onSelect(i)}>
                T{i + 1}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {fixes.map(([name, fix]) => (
            <tr key={name}>
              <th>
                <span className={`kind k-${fix.kind}`}>{fix.kind}</span>
                {fix.title}
              </th>
              {metrics.map((m, i) => {
                const state = m.learned.includes(name)
                  ? "learned"
                  : m.transferred.includes(name)
                    ? "transferred"
                    : m.knownBefore.includes(name)
                      ? "known"
                      : m.needed.includes(name)
                        ? "missed"
                        : "none";
                return (
                  <td key={m.task.id} className={i === selected ? "sel" : ""} onClick={() => onSelect(i)}>
                    <span className={`kcell ${state}`} title={TITLES[state]}>
                      {GLYPH[state]}
                    </span>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <div className="klegend">
        {(["learned", "transferred", "known", "none"] as const).map((s) => (
          <span key={s}>
            <span className={`kcell ${s}`}>{GLYPH[s]}</span>
            {LEGEND[s]}
          </span>
        ))}
      </div>
      <p className="muted small">
        A fix is <b>needed</b> when the base trajectory shows its warning. Transferred fixes are applied by the very first
        proposal; new ones must be discovered by probing.
      </p>
    </div>
  );
}

const GLYPH: Record<string, string> = { learned: "★", transferred: "↻", known: "·", missed: "!", none: "" };
const LEGEND: Record<string, string> = {
  learned: "learned here",
  transferred: "transferred",
  known: "known, unused",
  none: "not needed",
};
const TITLES: Record<string, string> = {
  learned: "Discovered on this task and stored as a lesson",
  transferred: "Needed here and already in the library",
  known: "In the library but not needed on this task",
  missed: "Needed but not adopted",
  none: "Not needed on this task",
};
