import { useState } from "react";
import { pct, stageColor, stageOrBase, type TaskMetrics } from "../metrics";
import type { Bundle, Stage } from "../types";

export function StageInspector({
  bundle,
  metrics,
  inspected,
}: {
  bundle: Bundle;
  metrics: TaskMetrics;
  inspected: string;
}) {
  const [file, setFile] = useState<"script" | "all">("script");
  const rollout = stageOrBase(metrics.task, metrics.record, inspected);
  const stage = rollout && "dim" in rollout ? (rollout as Stage) : undefined;
  const warnings = stage?.warnings ?? (inspected.startsWith("base") ? metrics.task.base_warnings : []);
  const diff = stage?.diff ?? "";
  const shown = file === "all" ? diff : diffFor(diff, "scripts/label.py");

  return (
    <div className="card inspector">
      <div className="card-head">
        <div>
          <h2>
            Harness inspector · <span className="mono">{inspected}</span>
          </h2>
          <p className="muted">
            {stage ? (
              <>
                <span className="tag" style={{ ["--kc" as string]: stageColor(stage.dim) }}>
                  {stage.dim === "AGG" ? "COMPOSE" : stage.dim}
                </span>{" "}
                {pct(stage.accuracy)} images right · F1 {stage.f1.toFixed(2)}
              </>
            ) : (
              "Unmodified base harness"
            )}
          </p>
        </div>
        {diff && (
          <div className="segmented small">
            <button className={file === "script" ? "on" : ""} onClick={() => setFile("script")}>
              label.py
            </button>
            <button className={file === "all" ? "on" : ""} onClick={() => setFile("all")}>
              all files
            </button>
          </div>
        )}
      </div>

      {stage?.notes && <blockquote className="notes">{stage.notes}</blockquote>}

      <div className="fix-chips">
        {(stage?.fixes ?? []).map((f) => {
          const fx = bundle.fixes[f];
          return (
            <span key={f} className={`fix-chip ${fx?.good ? "good" : "bad"}`} title={fx?.lesson}>
              {fx?.good ? "✓" : "⚠"} {fx?.title ?? f}
            </span>
          );
        })}
        {stage && !stage.fixes.length && <span className="fix-chip none">no perception edits</span>}
      </div>

      <div className="warn-list">
        <span className="chip-title">trajectory warnings · {warnings.length}</span>
        {warnings.length ? (
          <ul>
            {summarize(warnings).map(([w, n]) => (
              <li key={w}>
                <code>{w}</code>
                {n > 1 && <b>×{n}</b>}
              </li>
            ))}
          </ul>
        ) : (
          <p className="muted small">The labeler reported nothing it could not handle.</p>
        )}
      </div>

      {shown ? (
        <pre className="diff">
          {shown.split("\n").map((line, i) => (
            <span
              key={i}
              className={
                line.startsWith("+++") || line.startsWith("---")
                  ? "d-file"
                  : line.startsWith("@@")
                    ? "d-hunk"
                    : line.startsWith("+")
                      ? "d-add"
                      : line.startsWith("-")
                        ? "d-del"
                        : ""
              }
            >
              {line || " "}
              {"\n"}
            </span>
          ))}
        </pre>
      ) : (
        stage && <p className="muted small">This harness is identical to the base.</p>
      )}
    </div>
  );
}

function diffFor(diff: string, name: string) {
  const parts = diff.split(/(?=^--- base\/)/m);
  return parts.filter((p) => p.startsWith(`--- base/${name}`)).join("");
}

function summarize(warnings: string[]): [string, number][] {
  const counts = new Map<string, number>();
  for (const w of warnings) {
    const key = w.replace(/\(\d+ elements\)/, "(…)").replace(/'\d+'/, "'n'");
    counts.set(key, (counts.get(key) ?? 0) + 1);
  }
  return [...counts.entries()];
}
