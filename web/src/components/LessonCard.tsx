import type { TaskMetrics } from "../metrics";
import type { Bundle } from "../types";

function parseEdits(text = "") {
  return text
    .split(/\n\s*\n/)
    .map((block) =>
      Object.fromEntries(
        block
          .split("\n")
          .filter((l) => l.includes(":"))
          .map((l) => [l.slice(0, l.indexOf(":")).trim(), l.slice(l.indexOf(":") + 1).trim()]),
      ),
    )
    .filter((row) => row.change);
}

export function LessonCard({ bundle, metrics }: { bundle: Bundle; metrics: TaskMetrics }) {
  const exp = metrics.record.experience ?? {};
  const edits = parseEdits(exp.edits);
  const sentences = (exp.lesson ?? "").split(/(?<=\.)\s+/).filter(Boolean);
  const learned = new Set(metrics.learned.map((f) => bundle.fixes[f]?.lesson));
  return (
    <div className="card lesson">
      <div className="card-head">
        <div>
          <h2>Experience reflector</h2>
          <p className="muted">What this task adds to the improvement harness</p>
        </div>
        <span className="badge lib">library {metrics.libraryBefore} → {metrics.libraryBefore + (exp.lesson ? 1 : 0)}</span>
      </div>
      {exp.lesson ? (
        <>
          <p className="cue">
            <span>cue</span> {exp.cue}
          </p>
          <ul className="lesson-list">
            {sentences.map((s) => (
              <li key={s} className={learned.has(s) ? "new" : ""}>
                {learned.has(s) && <b>new</b>}
                {s}
              </li>
            ))}
          </ul>
          {edits.length > 0 && (
            <div className="edits">
              {edits.map((e, i) => (
                <div key={i} className={`edit-row ${e.verdict === "helped" ? "helped" : ""}`}>
                  <div className="edit-top">
                    <strong>{e.change}</strong>
                    <span className={`pill ${e.verdict === "helped" ? "pill-ok" : "pill-muted"}`}>{e.verdict}</span>
                  </div>
                  <small>
                    {e.component} · seen in {e.candidates}
                  </small>
                  <p>{e.observed}</p>
                </div>
              ))}
            </div>
          )}
          <p className="muted small">
            <b>evidence</b> {exp.evidence} <br />
            <b>confidence</b> {exp.confidence}
          </p>
        </>
      ) : (
        <p className="muted">No supported lesson was recorded for this task.</p>
      )}
    </div>
  );
}
