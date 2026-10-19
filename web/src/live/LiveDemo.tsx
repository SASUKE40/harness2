import { useEffect, useMemo, useReducer, useRef, useState } from "react";
import { imageSize, imageSrc, stageOrBase, type TaskMetrics } from "../metrics";
import type { Bundle, Mode, Stage } from "../types";
import { Frame } from "./Frame";
import { Caption, Confetti, Pillar, Scoreboard, Summary } from "./Motion";
import { buildTimeline, type Step } from "./timeline";
import type { Dataset } from "../App";
import "./live.css";

const FRAME_MS = 1150;
const TICK = 100;
const ICONS = ["⌕", "◧", "◐", "≋", "⊘", "◇"];

type State = { i: number; t: number; clock: number };
type Action = { type: "tick"; dt: number; steps: Step[] } | { type: "jump"; i: number } | { type: "reset" };

function reducer(s: State, a: Action): State {
  if (a.type === "reset") return { i: 0, t: 0, clock: 0 };
  if (a.type === "jump") return { ...s, i: a.i, t: 0 };
  const t = s.t + a.dt;
  if (t < a.steps[s.i].ms) return { ...s, t, clock: s.clock + a.dt };
  return { i: (s.i + 1) % a.steps.length, t: 0, clock: s.clock + a.dt };
}

export function LiveDemo({
  bundle,
  metrics,
  mode,
  setMode,
  onDashboard,
  startStep = 0,
  dataset,
  onDataset,
}: {
  bundle: Bundle;
  metrics: TaskMetrics[];
  mode: Mode;
  setMode: (m: Mode) => void;
  onDashboard: () => void;
  startStep?: number;
  dataset: Dataset;
  onDataset: (d: Dataset) => void;
}) {
  const steps = useMemo(() => buildTimeline(metrics), [metrics]);
  const [state, dispatch] = useReducer(reducer, {
    i: Math.max(0, Math.min(startStep, steps.length - 1)),
    t: 0,
    clock: 0,
  });
  const [playing, setPlaying] = useState(true);
  const [speed, setSpeed] = useState(1);

  // Restart only when the replay itself changes (mode or data), not on re-renders.
  const replayKey = `${mode}-${bundle.meta.generated_at}-${steps.length}`;
  const shown = useRef(replayKey);
  useEffect(() => {
    if (shown.current === replayKey) return;
    shown.current = replayKey;
    dispatch({ type: "reset" });
  }, [replayKey]);
  useEffect(() => {
    if (!playing) return;
    const id = window.setInterval(() => dispatch({ type: "tick", dt: TICK * speed, steps }), TICK);
    return () => window.clearInterval(id);
  }, [playing, speed, steps]);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === " ") {
        e.preventDefault();
        setPlaying((p) => !p);
      } else if (e.key === "ArrowRight") dispatch({ type: "jump", i: (state.i + 1) % steps.length });
      else if (e.key === "ArrowLeft") dispatch({ type: "jump", i: (state.i - 1 + steps.length) % steps.length });
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [state.i, steps.length]);

  const step = steps[Math.min(state.i, steps.length - 1)];
  const m = metrics[step.ti];
  const { task, record } = m;
  const frame = Math.floor(state.clock / FRAME_MS);
  const deploying = step.phase === "deploy" || step.phase === "summary";
  const adoptedStage = record.stages.find((s) => s.label === record.selection.chosen_label);
  const baseRun = task.base[0];
  // While the improved harness is shown, cycle the photos it fixed so the change is visible.
  const fixedImages = task.images.filter((img) => !baseRun.per_image[img.name] && adoptedStage?.per_image[img.name]);
  const reel = deploying && fixedImages.length ? fixedImages : task.images;
  const image = reel[frame % reel.length];
  const imgIndex = task.images.indexOf(image);
  const deployed = stageOrBase(task, record, step.deployed);
  const before = baseRun.detections?.[image.name] ?? [];
  const detections = (deploying ? adoptedStage : deployed)?.detections?.[image.name] ?? [];
  const visible = detections.filter((d) => d.status !== "missed");
  const changed = visible.findIndex((d) => d.status === "hit" && !before.some((b) => b.label === d.label && b.status === "hit"));
  const focus = deploying && changed >= 0 ? changed : visible.length ? frame % visible.length : 0;
  const target = visible[focus];
  const gain = (adoptedStage?.accuracy ?? 0) - m.base;

  const judged = step.phase === "select" || step.phase === "reflect" || deploying;
  const answers = Object.values(record.jev.answers ?? {});
  const conf = answers.length ? Math.min(...answers.map((a) => a.confidence)) : null;
  const threshold = record.selector_config?.min_confidence ?? 0.6;
  const decided = record.selection.selector === "jev";
  const jevOff = bundle.meta.jev_mode === "off";
  const latest = step.store[step.store.length - 1];

  const total = steps.reduce((n, s) => n + s.ms, 0);
  const done = steps.slice(0, state.i).reduce((n, s) => n + s.ms, 0) + state.t;
  const rec = Math.floor(state.clock / 1000);
  const hms = [Math.floor(rec / 3600), Math.floor(rec / 60) % 60, rec % 60].map((v) => String(v).padStart(2, "0")).join(":");

  const boxConf = target?.confidence;
  const shownConf = boxConf ?? (judged ? conf : null);
  const live = deployed && "tools" in deployed ? (deployed as Stage) : undefined;
  const building = step.stage ?? (judged ? undefined : step.lastCompose);
  const counts = {
    rules: live?.prompt_rules.length ?? 0,
    tools: live?.tools.length ?? 0,
    nextRules: building?.prompt_rules.length,
    nextTools: building?.tools.length,
  };
  const composedSoFar = judged
    ? record.finals
    : record.finals.slice(0, step.lastCompose ? record.finals.indexOf(step.lastCompose.label) + 1 : 0);
  const proposing = step.phase === "propose" || step.phase === "probe";
  const activeProbe = proposing ? step.stage : step.lastProbe;
  const composed = judged ? record.stages.find((s) => s.label === record.selection.chosen_label) : step.lastCompose;
  const fixLine = (f: string) => {
    const fx = bundle.fixes[f];
    const glyph = !fx?.good ? "⚠" : m.knownBefore.includes(f) ? "↻" : "+";
    return (
      <div key={f} className={`lv-fixline ${fx?.good ? "" : "bad"} ${glyph === "↻" ? "lib" : ""}`}>
        {glyph} {fx?.trigger} → {fx?.action}
      </div>
    );
  };

  return (
    <div className="lv">
      <header className="lv-head">
        <h1 className="lv-title">
          <span className="lv-mark" aria-hidden>
            <i />
            <i />
            <i />
          </span>
          Harness<sup>2</sup>
          <span className="lv-plus">×</span>
          Jev
        </h1>
        <div className="lv-pillars">
          <Pillar value={0} label="weights touched" hint="Zero weights touched" tone="ink" />
          <Pillar
            value={counts.rules}
            next={counts.nextRules}
            label="prompt rules"
            hint="Rewrites its own prompts"
            tone="violet"
          />
          <Pillar
            value={counts.tools}
            next={counts.nextTools}
            label="tools built"
            hint="Builds its own tools"
            tone="teal"
          />
          <Pillar value={step.lessons} label="lessons" hint="Distills its own experience" tone="ochre" />
        </div>
        <span className="lv-phase">
          T{step.ti + 1}/{metrics.length} · {task.title} · <b>{step.phase}</b>
        </span>
      </header>

      <main className="lv-main">
        {step.phase === "summary" && <Summary bundle={bundle} metrics={metrics} />}
        <fieldset className="lv-sec lv-live">
          <legend>Live</legend>
          <div className="lv-live-top">
            <fieldset className="lv-box lv-input">
              <legend>Input</legend>
              <div className="lv-frame">
                <div className={`lv-screen ${deploying ? "is-after" : ""}`}>
                  {deploying ? (
                    <>
                      <Frame image={image} detections={before} focus={-1} adapted={false} anchor={[...before, ...detections]} />
                      <div className="lv-after" key={`after-${state.i}`}>
                        <Frame image={image} detections={detections} focus={focus} adapted anchor={[...before, ...detections]} />
                      </div>
                      <div className="lv-wipe" key={`wipe-${state.i}`}>
                        <span>after</span>
                      </div>
                      <span className="lv-side before">before</span>
                    </>
                  ) : (
                    <Frame image={image} detections={detections} focus={focus} adapted={step.adapted} />
                  )}
                  <Scoreboard metrics={m} after={adoptedStage} revealed={deploying} run={state.i} />
                  {step.phase === "deploy" && gain >= 0.3 && <Confetti run={state.i} />}
                  <Caption step={step} m={m} bundle={bundle} run={state.i} />
                  <i className="lv-corner tl" />
                  <i className="lv-corner tr" />
                  <i className="lv-corner bl" />
                  <i className="lv-corner br" />
                  <div className="lv-scan" />
                </div>
                <div className="lv-hud">
                  <span>
                    <b className="lv-rec" /> REC {hms}
                  </span>
                  <span className={step.adapted ? "lv-boost" : "lv-baseline"}>
                    ∘ {step.adapted ? `adapted · ${step.deployed}` : "base harness"}
                  </span>
                  <span>
                    img {imgIndex + 1}/{task.images.length}
                  </span>
                </div>
              </div>
            </fieldset>
            <div className="lv-track" aria-hidden>
              {task.images.map((img, i) => (
                <i key={img.name} className={i === imgIndex ? "on" : i < imgIndex ? "past" : ""} />
              ))}
            </div>
          </div>

          <div className="lv-live-bottom">
            <fieldset className={`lv-box lv-jev ${step.phase === "select" ? "active" : ""}`}>
              <legend>Jev</legend>
              <div className="lv-kv">
                <span>
                  <em>class</em> <b>{target ? target.label : "—"}</b>
                  {target && target.status !== "hit" && <i className="lv-x"> ✗ {target.expected ?? "not a sign"}</i>}
                </span>
                <span>
                  <em>bbox</em>{" "}
                  <b className="lv-bbox">[{target ? target.bbox.map((v) => Math.round((v * 640) / imageSize(image)[0])).join(",") : "—"}]</b>
                </span>
              </div>
              <div className="lv-conf">
                <em>conf</em>
                <span className="lv-meter">
                  <i style={{ width: `${shownConf != null ? shownConf * 100 : 0}%` }} />
                  <b style={{ left: `${threshold * 100}%` }} />
                </span>
                {boxConf != null ? (
                  <>
                    <b className="lv-num">{boxConf.toFixed(2)}</b>
                    <span className={boxConf >= threshold ? "lv-ok" : "lv-warn"}>
                      {boxConf >= threshold ? "✓" : "✗"} {boxConf >= threshold ? "≥" : "<"} {threshold.toFixed(2)}
                    </span>
                  </>
                ) : judged && conf != null ? (
                  <>
                    <b className="lv-num">{conf.toFixed(2)}</b>
                    <span className={decided ? "lv-ok" : "lv-warn"}>
                      {decided ? `✓ ≥ ${threshold.toFixed(2)}` : "⇄ tie"}
                    </span>
                  </>
                ) : (
                  <span className="lv-dim">{jevOff ? "jev off" : "awaiting finals…"}</span>
                )}
              </div>
              <div className="lv-lock">weights frozen</div>
              <div className="lv-versions">
                <span>[prompt +{counts.rules} rules]</span>
                <span>[tools {counts.tools}]</span>
                <span>[lessons {step.lessons}]</span>
              </div>
              <div className={`lv-latest ${step.phase === "reflect" ? "flash" : ""}`}>
                {latest ? (
                  <>
                    + {bundle.fixes[latest]?.trigger} → {bundle.fixes[latest]?.action}
                  </>
                ) : (
                  <span className="lv-dim">no lessons yet</span>
                )}
              </div>
            </fieldset>

            <div className="lv-outs">
              <span className="lv-tplus">↻ t+1</span>
              <fieldset className={`lv-box lv-out ${judged && decided ? "active" : ""}`}>
                <legend>Output</legend>
                <span>{judged && decided ? `emit(${record.selection.chosen_label})` : "emit(result)"} ·</span>
              </fieldset>
              <fieldset className={`lv-box lv-fallback ${judged && !decided ? "active" : ""}`}>
                <legend>Fallback</legend>
                <span>⇄ {judged && !decided ? `agent → ${record.selection.chosen_label}` : "review"}</span>
              </fieldset>
              <span className="lv-readonly">
                read-only <b>▲</b>
              </span>
            </div>
          </div>
        </fieldset>

        <div className="lv-evo" aria-hidden>
          <i className="lv-evo-line" />
          <span className="lv-evo-label">runtime → evolution</span>
          <i className="lv-evo-line" />
        </div>

        <fieldset className="lv-sec lv-bg">
          <legend>Background</legend>
          <div className="lv-bg-grid">
            <fieldset className={`lv-box lv-traj ${step.phase === "observe" ? "active" : ""}`}>
              <legend><b>1</b> Trajectories</legend>
              <div className="lv-glyphs">
                {step.glyphs.map((g, i) => (
                  <span key={i} className={g.tone}>
                    {g.g}
                  </span>
                ))}
              </div>
            </fieldset>
            <fieldset
              className={`lv-box lv-propose ${step.phase === "propose" ? "active" : ""} ${proposing ? "" : "past"}`}
            >
              <legend><b>2</b> Propose</legend>
              {activeProbe ? (
                <div className="lv-reveal" key={activeProbe.label}>
                  <div className="lv-hyp">
                    <b>H{step.hyp}</b>{" "}
                    {activeProbe.fixes.some((f) => bundle.fixes[f]?.trigger !== "partial_output") ? "tool" : "guard"}
                    <span className="lv-dim"> · {activeProbe.label}</span>
                  </div>
                  {activeProbe.fixes.length ? activeProbe.fixes.map(fixLine) : <div className="lv-dim">no edit</div>}
                  {activeProbe.notes && <div className="lv-notes">{activeProbe.notes}</div>}
                </div>
              ) : (
                <div className="lv-dim">{step.phase === "observe" ? "reading trajectories…" : "idle"}</div>
              )}
            </fieldset>
            <fieldset className={`lv-box lv-probe ${step.phase === "probe" ? "active" : ""}`}>
              <legend><b>3</b> Probe</legend>
              <StageLine stage={step.lastProbe} />
              {step.lastProbe && (
                <div
                  className="lv-thumbs"
                  style={{ gridTemplateColumns: `repeat(${Math.max(3, Math.ceil(task.images.length / 2))}, minmax(0, 1fr))` }}
                >
                  {task.images.map((img, i) => {
                    const ok = step.lastProbe?.per_image[img.name];
                    return (
                      <span
                        key={`${img.name}-${step.lastProbe?.label}`}
                        className={`lv-thumb ${ok ? "ok" : "ko"}`}
                        style={{ animationDelay: `${i * 70}ms` }}
                      >
                        <img src={imageSrc(img)} alt={img.name} />
                        <b>{ok ? "✓" : "✗"}</b>
                      </span>
                    );
                  })}
                </div>
              )}
            </fieldset>
            <fieldset className={`lv-box lv-compose ${step.phase === "compose" ? "active" : ""}`}>
              <legend><b>4</b> Compose</legend>
              <StageLine stage={composed} chosen={judged ? record.selection.chosen_label : ""} />
              <div className="lv-rounds">
                {record.finals.map((f, i) => {
                  const final = record.stages.find((s) => s.label === f);
                  const shown = composedSoFar.includes(f);
                  const win = judged && f === record.selection.chosen_label;
                  return (
                    <div key={f} className={`lv-round ${shown ? "" : "pending"} ${win ? "win" : ""}`}>
                      <b>{shown && final ? `${Math.round(final.accuracy * 100)}%` : "·"}</b>
                      <span className="lv-round-bar">
                        <i style={{ height: `${shown && final ? Math.max(4, final.accuracy * 100) : 0}%` }} />
                      </span>
                      <small>{mode === "seq" ? `round ${i + 1}` : f}</small>
                    </div>
                  );
                })}
              </div>
            </fieldset>
            <fieldset className={`lv-box lv-store ${step.phase === "reflect" ? "active" : ""}`}>
              <legend>
                <b>5</b> Experience store
              </legend>
              <div className="lv-ver">v{step.lessons}</div>
              {step.store.length ? (
                step.store.map((f, i) => (
                  <div key={f} className={`lv-entry ${step.fresh.includes(f) ? "fresh" : ""}`}>
                    <div>
                      <span className="lv-ico">{ICONS[i % ICONS.length]}</span> {bundle.fixes[f]?.trigger}
                    </div>
                    <div>→ {bundle.fixes[f]?.action}</div>
                  </div>
                ))
              ) : (
                <div className="lv-dim">empty: first task in this domain</div>
              )}
            </fieldset>
          </div>
        </fieldset>
      </main>

      <footer className="lv-status">
        <div className="lv-progress">
          <i style={{ width: `${(done / total) * 100}%` }} />
        </div>
        <div className="lv-status-row">
          <div className="lv-stats">
            <span>
              round <b>[{step.round}/{step.rounds}]</b>
            </span>
            <span>
              conf <b>[{judged && conf != null ? conf.toFixed(2) : "—"}]</b>
            </span>
            <span>
              fallback <b>[{step.deferred}/{step.decided + step.deferred}]</b>
            </span>
            <span>
              store <b>[v{step.lessons}]</b>
            </span>
          </div>
          <span className="lv-note">
            {bundle.meta.dataset ? (
              <a href={bundle.meta.dataset.url} target="_blank" rel="noreferrer">
                photos: {bundle.meta.dataset.name} ({bundle.meta.dataset.license})
              </a>
            ) : (
              "synthetic scenes"
            )}{" "}
            · {bundle.meta.executor === "jev" ? "Jev labels the images" : "harness labeler"} · selector:{" "}
            {bundle.meta.jev_mode === "offline" ? "Jev offline stand-in" : `Jev ${bundle.meta.jev_mode}`} · scripted
            editor · ✗ judged post-hoc
          </span>
          <div className="lv-controls">
            {(["aircraft", "real", "synthetic"] as const).map((d) => (
              <button key={d} className={d === dataset ? "on" : ""} onClick={() => d !== dataset && onDataset(d)}>
                {{ aircraft: "aircraft", real: "street signs", synthetic: "synthetic" }[d]}
              </button>
            ))}
            {bundle.meta.modes.map((md) => (
              <button key={md} className={md === mode ? "on" : ""} onClick={() => setMode(md)}>
                {md === "agg" ? "parallel" : "sequential"}
              </button>
            ))}
            <button onClick={() => setSpeed((s) => (s >= 4 ? 1 : s * 2))}>{speed}×</button>
            <button onClick={() => setPlaying((p) => !p)}>{playing ? "❚❚" : "▶"}</button>
            <button onClick={onDashboard}>analytics ↗</button>
          </div>
        </div>
      </footer>
    </div>
  );
}

function StageLine({ stage, chosen = "" }: { stage?: Stage; chosen?: string }) {
  if (!stage) return <div className="lv-dim">—</div>;
  const ok = stage.accuracy >= 0.999;
  return (
    <div className="lv-stage">
      <div>
        <b>{stage.label}</b> ▸ {Math.round(stage.accuracy * 100)}% <span className={ok ? "lv-ok" : "lv-bad"}>{ok ? "✓" : "✗"}</span>
        {chosen && chosen === stage.label && <span className="lv-star"> ★</span>}
      </div>
      <div className="lv-mini">
        <i style={{ width: `${stage.accuracy * 100}%` }} className={ok ? "ok" : "bad"} />
      </div>
      <div className="lv-dim">{stage.fixes.filter((f) => f !== "complete").length} harness edits</div>
    </div>
  );
}
