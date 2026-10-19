import { useEffect, useRef, useState } from "react";
import { pct, type TaskMetrics } from "../metrics";
import type { Bundle, Stage } from "../types";
import type { Step } from "./timeline";

/** A number that eases from `from` to `to` whenever `to` (or `run`) changes. */
export function CountUp({ to, from = 0, ms = 1400, run, suffix = "" }: { to: number; from?: number; ms?: number; run?: string | number; suffix?: string }) {
  const [value, setValue] = useState(from);
  useEffect(() => {
    let frame = 0;
    const start = performance.now();
    const tick = (now: number) => {
      const t = Math.min(1, (now - start) / ms);
      const eased = 1 - Math.pow(1 - t, 3);
      setValue(from + (to - from) * eased);
      if (t < 1) frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [to, from, ms, run]);
  return (
    <>
      {Math.round(value)}
      {suffix}
    </>
  );
}

const BURST = ["#3e63dd", "#1f9d6b", "#d97706", "#8e4ec6", "#0d9488", "#d6336c"];

export function Confetti({ run }: { run: string | number }) {
  return (
    <div className="lv-confetti" key={run} aria-hidden>
      {Array.from({ length: 22 }, (_, i) => {
        const angle = (i / 22) * Math.PI * 2 + (i % 3) * 0.2;
        const dist = 70 + (i % 5) * 22;
        return (
          <i
            key={i}
            style={
              {
                "--dx": `${Math.cos(angle) * dist}px`,
                "--dy": `${Math.sin(angle) * dist - 30}px`,
                "--r": `${(i * 47) % 360}deg`,
                background: BURST[i % BURST.length],
                animationDelay: `${(i % 4) * 30}ms`,
              } as React.CSSProperties
            }
          />
        );
      })}
    </div>
  );
}

/** Big before -> after numbers, a delta badge and one flipping dot per image. */
export function Scoreboard({
  metrics,
  after,
  revealed,
  run,
}: {
  metrics: TaskMetrics;
  after?: Stage;
  revealed: boolean;
  run: string | number;
}) {
  const base = metrics.task.base[0];
  const gain = Math.round(((after?.accuracy ?? 0) - metrics.base) * 100);
  return (
    <div className={`lv-score ${revealed ? "revealed" : ""}`}>
      <div className="lv-score-row">
        <div className="lv-score-cell before">
          <small>before</small>
          <b>{pct(metrics.base)}</b>
        </div>
        <span className="lv-score-arrow">→</span>
        <div className="lv-score-cell after">
          <small>after</small>
          <b>{revealed && after ? <CountUp to={after.accuracy * 100} from={metrics.base * 100} run={run} suffix="%" /> : "?"}</b>
        </div>
        {revealed && gain > 0 && (
          <span className="lv-gain" key={`gain-${run}`}>
            +{gain} pts
          </span>
        )}
      </div>
      <div className="lv-dots">
        {metrics.task.images.map((img, i) => {
          const ok = revealed && after ? after.per_image[img.name] : base.per_image[img.name];
          return (
            <i
              key={`${img.name}-${revealed ? run : "b"}`}
              className={`${ok ? "ok" : "ko"} ${revealed ? "flip" : ""}`}
              style={{ animationDelay: `${600 + i * 160}ms` }}
              title={img.name}
            />
          );
        })}
      </div>
    </div>
  );
}

function fixTitles(bundle: Bundle, fixes: string[]) {
  return fixes.map((f) => bundle.fixes[f]?.title.toLowerCase() ?? f).join(" + ");
}

/** One plain-language line per moment of the replay. */
export function captionFor(step: Step, m: TaskMetrics, bundle: Bundle): { kicker: string; text: string; tone: string } {
  const { record } = m;
  const chosen = record.stages.find((s) => s.label === record.selection.chosen_label);
  const firstWarning = m.task.base_warnings[0] ?? "";
  const missing = Object.entries(bundle.fixes).find(([, f]) => f.good && f.warning && firstWarning.includes(f.warning));
  switch (step.phase) {
    case "observe":
      return {
        kicker: "Before",
        text: `The starting harness gets ${pct(m.base)} of photos right${missing ? `: it can't ${missing[1].title.toLowerCase()}` : ""}.`,
        tone: "bad",
      };
    case "propose": {
      // Only what this probe adds to the harness it started from (the last composition).
      const startedWith = new Set(step.lastCompose?.fixes ?? []);
      const added = (step.stage?.fixes ?? []).filter((f) => !startedWith.has(f));
      const fresh = added.filter((f) => !m.knownBefore.includes(f) && bundle.fixes[f]?.tool);
      const reused = added.filter((f) => m.knownBefore.includes(f));
      const risky = fresh.filter((f) => !bundle.fixes[f]?.good);
      if (risky.length)
        return { kicker: "Explores", text: `Harness² also tries a risky shortcut: ${fixTitles(bundle, risky)}.`, tone: "amber" };
      if (fresh.length) return { kicker: "Builds a tool", text: `Harness² writes a new tool: ${fixTitles(bundle, fresh)}.`, tone: "violet" };
      if (reused.length) return { kicker: "Remembers", text: `Reusing what it learned before: ${fixTitles(bundle, reused)}.`, tone: "rose" };
      return { kicker: "Proposes", text: "Harness² tries a small change to its own instructions.", tone: "violet" };
    }
    case "probe":
      return {
        kicker: "Tests it",
        text: `Trying ${step.stage?.label} on the photos: ${pct(step.stage?.accuracy ?? 0)} right.`,
        tone: (step.stage?.accuracy ?? 0) > m.base ? "ok" : "bad",
      };
    case "compose":
      return { kicker: "Combines", text: `Merging the edits that helped: ${pct(step.stage?.accuracy ?? 0)} right.`, tone: "sky" };
    case "select":
      return record.selection.selector === "jev"
        ? { kicker: "Jev decides", text: `Jev picks the final harness (${pct(chosen?.accuracy ?? 0)}).`, tone: "indigo" }
        : { kicker: "Double-checks", text: `Jev wasn't sure, so each claim was checked; ${record.selection.chosen_label} wins (${pct(chosen?.accuracy ?? 0)}).`, tone: "amber" };
    case "reflect": {
      const f = step.fresh[0];
      return f
        ? { kicker: "Learns", text: `Lesson saved for next time: ${bundle.fixes[f].trigger} → ${bundle.fixes[f].action}.`, tone: "ochre" }
        : { kicker: "Learns", text: "Nothing new to save; past lessons covered this task.", tone: "ochre" };
    }
    case "deploy":
      return { kicker: "After", text: `Same photos, improved harness: ${pct(m.base)} → ${pct(chosen?.accuracy ?? 0)}. No weights changed.`, tone: "ok" };
    default:
      return { kicker: "Result", text: "Every task improved without touching a single weight.", tone: "ok" };
  }
}

export function Caption({ step, m, bundle, run }: { step: Step; m: TaskMetrics; bundle: Bundle; run: number }) {
  const { kicker, text, tone } = captionFor(step, m, bundle);
  return (
    <div className={`lv-caption tone-${tone}`} key={run}>
      <span>{kicker}</span>
      <p>{text}</p>
    </div>
  );
}

/** Pill counter that floats a "+n" whenever it grows. */
export function Pillar({ value, next, label, hint, tone }: { value: number; next?: number; label: string; hint: string; tone: string }) {
  const previous = useRef(value);
  const [delta, setDelta] = useState(0);
  useEffect(() => {
    setDelta(value - previous.current);
    previous.current = value;
  }, [value]);
  return (
    <span className={`lv-pillar tone-${tone} ${delta > 0 ? "grew" : ""}`} title={hint} key={delta > 0 ? `${label}-${value}` : label}>
      <b key={value}>{value}</b>
      {next != null && next !== value && <i>→ {next}</i>}
      <span>{label}</span>
      {delta > 0 && (
        <em className="lv-plus1" key={`${label}-${value}`}>
          +{delta}
        </em>
      )}
    </span>
  );
}

export function Summary({ bundle, metrics }: { bundle: Bundle; metrics: TaskMetrics[] }) {
  const finals = metrics.map((m) => m.record.stages.find((s) => s.label === m.record.selection.chosen_label));
  const before = metrics.reduce((n, m) => n + m.base, 0) / metrics.length;
  const after = finals.reduce((n, s) => n + (s?.accuracy ?? 0), 0) / metrics.length;
  const last = finals[finals.length - 1];
  const lessons = metrics[metrics.length - 1].libraryBefore + (metrics[metrics.length - 1].record.experience?.lesson ? 1 : 0);
  return (
    <div className="lv-summary">
      <div className="lv-summary-card">
        <p className="lv-summary-kicker">{bundle.meta.source === "aircraft"
            ? `${metrics.length * metrics[0].task.images.length} aircraft photos (FGVC-Aircraft)`
            : bundle.meta.source === "real"
              ? `${metrics.length * metrics[0].task.images.length} real street photos`
              : "Task stream"} · Jev weights frozen</p>
        <h2>
          <span className="from">
            <CountUp to={before * 100} ms={900} run="b" suffix="%" />
          </span>
          <span className="arrow">→</span>
          <span className="to">
            <CountUp to={after * 100} from={before * 100} ms={2000} run="a" suffix="%" />
          </span>
        </h2>
        <p className="lv-summary-sub">average accuracy, starting harness → harness Harness² built</p>
        <div className="lv-summary-rows">
          {metrics.map((m, i) => (
            <div className="lv-summary-row" key={m.task.id} style={{ animationDelay: `${300 + i * 180}ms` }}>
              <span className="name">{m.task.title}</span>
              <span className="bars">
                <i className="before" style={{ width: `${Math.max(2, m.base * 100)}%` }} />
                <i className="after" style={{ width: `${Math.max(2, (finals[i]?.accuracy ?? 0) * 100)}%`, animationDelay: `${700 + i * 180}ms` }} />
              </span>
              <span className="nums">
                {pct(m.base)} → <b>{pct(finals[i]?.accuracy ?? 0)}</b>
              </span>
            </div>
          ))}
        </div>
        <div className="lv-summary-pills">
          <span>
            <b>0</b> weights touched
          </span>
          <span>
            <b>{last?.prompt_rules.length ?? 0}</b> prompt rules written
          </span>
          <span>
            <b>{last?.tools.length ?? 0}</b> tools built
          </span>
          <span>
            <b>{lessons}</b> lessons distilled
          </span>
        </div>
      </div>
      <Confetti run="summary" />
    </div>
  );
}
