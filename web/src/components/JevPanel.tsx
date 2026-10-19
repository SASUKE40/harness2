import { DIMENSIONS, pct, type TaskMetrics } from "../metrics";
import type { Bundle } from "../types";

const BUDGET = 28000;

export function JevPanel({ bundle, metrics }: { bundle: Bundle; metrics: TaskMetrics }) {
  const { record } = metrics;
  const { jev, selection, jev_request: req } = record;
  const answers = jev.answers ?? {};
  const anons = Object.keys(selection.id_map).sort();
  const dims = DIMENSIONS.filter((d) => anons.some((a) => answers[`${a}_${d}`]));
  const decided = selection.selector === "jev";
  const off = bundle.meta.jev_mode === "off";
  const accuracy = (label: string) => record.stages.find((s) => s.label === label)?.accuracy ?? 0;
  const tokens = jev.usage?.input_tokens;

  return (
    <div className="card jev">
      <div className="card-head">
        <div>
          <h2>
            <span className="jev-mark">J</span> Jev final selection
          </h2>
          <p className="muted">
            {off
              ? "Jev disabled for this run"
              : `${jev.model ?? jev.requested_model ?? bundle.meta.jev_model}${
                  bundle.meta.jev_mode === "offline" ? " · deterministic stand-in, no API call" : ""
                }`}
          </p>
        </div>
      </div>

      <div className={`verdict-banner ${decided ? "decided" : "deferred"}`}>
        {decided ? (
          <>
            <strong>
              Jev chose {selection.chosen_id} <span className="muted">=</span> {selection.chosen_label}
            </strong>
            <span>All judgments met the confidence threshold; leaders were not tied.</span>
          </>
        ) : (
          <>
            <strong>
              {off ? "Agent selector" : "Jev deferred"} → {selection.chosen_label}
            </strong>
            <span>{jev.fallback_reason ?? "The agent selector made this choice."}</span>
          </>
        )}
      </div>

      {dims.length > 0 && (
        <div className="heat">
          <div className="heat-row head">
            <span>candidate</span>
            {dims.map((d) => (
              <span key={d} title={d}>
                {d.slice(0, 5)}
              </span>
            ))}
          </div>
          {anons.map((a) => {
            const label = selection.id_map[a];
            const win = a === selection.chosen_id;
            return (
              <div key={a} className={`heat-row ${win ? "win" : ""}`}>
                <span className="cand">
                  <b>{a}</b>
                  <small>
                    {label} · {pct(accuracy(label))}
                  </small>
                </span>
                {dims.map((d) => {
                  const ans = answers[`${a}_${d}`];
                  if (!ans) return <span key={d} />;
                  const p = ans.probabilities ?? {};
                  return (
                    <span
                      key={d}
                      className="heat-cell"
                      style={{ ["--s" as string]: String(ans.score / 2) }}
                      title={`${d}: score ${ans.score} · confidence ${ans.confidence}`}
                    >
                      <em>{ans.score.toFixed(1)}</em>
                      <span className="probs">
                        {["0", "1", "2"].map((k) => (
                          <i key={k} style={{ height: `${Math.max(6, (p[k] ?? 0) * 100)}%` }} />
                        ))}
                      </span>
                    </span>
                  );
                })}
              </div>
            );
          })}
          <p className="muted small">
            Scores (0–2) are compared in priority order, never averaged: {dims.join(" › ")}. Candidate labels and
            post-hoc accuracy are shown here only; Jev saw anonymous IDs.
          </p>
        </div>
      )}

      {req.bytes > 0 && (
        <div className="req">
          <div className="req-bar">
            <span>Request</span>
            <div className="bar">
              <i style={{ width: `${Math.min(100, (req.bytes / BUDGET) * 100)}%` }} />
            </div>
            <span className="mono">
              {(req.bytes / 1000).toFixed(1)} / {BUDGET / 1000} KB
            </span>
          </div>
          <div className="req-facts">
            <Fact k="questions" v={String(Object.keys(req.questions).length)} />
            <Fact
              k="source files"
              v={`${req.source_files.filter((f) => f.startsWith("images/")).length} images + ${req.source_files.filter((f) => !f.startsWith("images/")).length} text`}
            />
            <Fact k="tokens" v={tokens != null ? tokens.toLocaleString() : "n/a"} />
            <Fact k="est. cost" v={jev.estimated_usd != null ? `$${jev.estimated_usd.toFixed(6)}` : "$0"} />
          </div>
        </div>
      )}
      <p className="post-hoc">
        {metrics.regret === 0 ? "✓" : "✗"} Post-hoc check: the adopted final scores {pct(metrics.adopted)}; best final{" "}
        {pct(metrics.oracle)}.
      </p>
    </div>
  );
}

function Fact({ k, v }: { k: string; v: string }) {
  return (
    <div className="fact">
      <span>{k}</span>
      <b>{v}</b>
    </div>
  );
}
