import { accuracyColor, isBadStage, pct, stageColor, type TaskMetrics } from "../metrics";
import type { Bundle, Mode, Rollout, Stage } from "../types";

interface GNode {
  id: string;
  x: number;
  y: number;
  kind: "BASE" | "INV" | "DOM" | "AGG" | "COMPOSE" | "SEL" | "LIB";
  rollout?: Rollout | Stage;
}

const spread = (n: number, lo = 16, hi = 84) =>
  n === 1 ? [50] : Array.from({ length: n }, (_, i) => lo + ((hi - lo) * i) / (n - 1));

function layout(mode: Mode, m: TaskMetrics) {
  const { task, record } = m;
  const stage = (label: string) => record.stages.find((s) => s.label === label);
  const k = record.finals.length;
  const nodes: GNode[] = [];
  const edges: [string, string][] = [];
  const heads: { x: number; text: string }[] = [];
  const mid = mode === "seq" ? 38 : 50;
  const baseYs = spread(task.base.length, mid - 12, mid + 12);
  if (mode === "agg") {
    task.base.forEach((b, i) => nodes.push({ id: b.label, x: 7, y: baseYs[i], kind: "BASE", rollout: b }));
    const probes = record.stages.filter((s) => s.dim === "INV" || s.dim === "DOM");
    const py = spread(probes.length, 12, 88);
    probes.forEach((s, i) => nodes.push({ id: s.label, x: 31, y: py[i], kind: s.dim, rollout: s }));
    const ay = spread(k, 30, 70);
    record.finals.forEach((f, i) => nodes.push({ id: f, x: 56, y: ay[i], kind: "AGG", rollout: stage(f) }));
    task.base.forEach((b) => probes.forEach((p) => edges.push([b.label, p.label])));
    probes.forEach((p) => record.finals.forEach((f) => edges.push([p.label, f])));
    heads.push(
      { x: 7, text: "Base executions" },
      { x: 31, text: `Propose & probe · ${probes.length}` },
      { x: 56, text: `Compose · ${k} independent` },
    );
  } else {
    const cols = 1 + 2 * k;
    const xs = Array.from({ length: cols }, (_, i) => 6 + (i * 58) / (cols - 1));
    task.base.forEach((b, i) => nodes.push({ id: b.label, x: xs[0], y: baseYs[i], kind: "BASE", rollout: b }));
    heads.push({ x: xs[0], text: "Base executions" });
    for (let t = 1; t <= k; t++) {
      const px = xs[2 * t - 1];
      const cx = xs[2 * t];
      const inv = stage(`inv${t}`);
      const dom = stage(`dom${t}`);
      if (inv) nodes.push({ id: inv.label, x: px, y: 14, kind: "INV", rollout: inv });
      if (dom) nodes.push({ id: dom.label, x: px, y: 62, kind: "DOM", rollout: dom });
      nodes.push({ id: `c${t}`, x: cx, y: mid, kind: "COMPOSE", rollout: stage(`c${t}`) });
      const from = t === 1 ? task.base.map((b) => b.label) : [`c${t - 1}`];
      from.forEach((f) => [inv, dom].forEach((p) => p && edges.push([f, p.label])));
      [inv, dom].forEach((p) => p && edges.push([p.label, `c${t}`]));
      if (t > 1) edges.push([`c${t - 1}`, `c${t}`]);
      heads.push({ x: (px + cx) / 2, text: `Round ${t}` });
    }
  }
  nodes.push({ id: "__sel", x: 79, y: mid, kind: "SEL" });
  nodes.push({ id: "__lib", x: 94, y: mid, kind: "LIB" });
  record.finals.forEach((f) => edges.push([f, "__sel"]));
  edges.push(["__sel", "__lib"]);
  heads.push({ x: 79, text: "Select final" }, { x: 94, text: "Reflect" });
  return { nodes, edges, heads };
}

function lineage(mode: Mode, m: TaskMetrics) {
  const { record } = m;
  const chosen = record.stages.find((s) => s.label === record.selection.chosen_label);
  const on = new Set<string>([record.selection.chosen_label, "__sel", "__lib"]);
  if (!chosen) return on;
  if (mode === "agg") {
    record.stages
      .filter((s) => (s.dim === "INV" || s.dim === "DOM") && s.fixes.some((f) => chosen.fixes.includes(f)))
      .forEach((s) => on.add(s.label));
    m.task.base.forEach((b) => on.add(b.label));
  } else {
    const t = Number(chosen.label.slice(1));
    for (let i = 1; i <= t; i++) {
      on.add(`c${i}`);
      [`inv${i}`, `dom${i}`].forEach((p) => {
        const s = record.stages.find((x) => x.label === p);
        if (s && s.fixes.every((f) => chosen.fixes.includes(f))) on.add(p);
      });
    }
    m.task.base.forEach((b) => on.add(b.label));
  }
  return on;
}

export function RecursionGraph({
  bundle,
  mode,
  metrics,
  inspected,
  onInspect,
}: {
  bundle: Bundle;
  mode: Mode;
  metrics: TaskMetrics;
  inspected: string;
  onInspect: (label: string) => void;
}) {
  const { nodes, edges, heads } = layout(mode, metrics);
  const live = lineage(mode, metrics);
  const pos = Object.fromEntries(nodes.map((n) => [n.id, n]));
  const { record } = metrics;
  const sel = record.selection;
  const lessons = metrics.libraryBefore + (record.experience?.lesson ? 1 : 0);
  const k = record.finals.length;

  return (
    <section className="card graph-card">
      <div className="card-head">
        <div>
          <h2>
            Task-level recursion · <span className="muted-strong">{metrics.task.title}</span>
          </h2>
          <p className="muted">
            {mode === "agg"
              ? "Parallel: 2k probes from the same base evidence, then k independent compositions"
              : "Sequential: k nested Propose–Probe–Compose rounds, each building on the last"}
            {" · "}click any node to inspect its harness and labels
          </p>
        </div>
        <div className="legend">
          <span className="lg k-INV"><i />Invariant edit</span>
          <span className="lg k-DOM"><i />Domain edit</span>
          <span className="lg k-COMPOSE"><i />Composition</span>
          <span className="lg k-bad"><i />Wrong rule</span>
        </div>
      </div>
      <div className="graph" style={{ height: Math.max(420, 80 + 68 * (mode === "agg" ? 2 * k : 2)) }}>
        {heads.map((h) => (
          <span key={h.text} className="ghead" style={{ left: `${h.x}%` }}>
            {h.text}
          </span>
        ))}
        <div className="graph-inner">
        <svg className="edges" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden>
          {edges.map(([a, b]) => {
            const p = pos[a];
            const q = pos[b];
            if (!p || !q) return null;
            const dx = (q.x - p.x) / 2;
            const hot = live.has(a) && live.has(b);
            const toSel = b === "__sel" && a === sel.chosen_label;
            const skips = mode === "seq" && b === "__sel" && a !== record.finals[record.finals.length - 1];
            const d = skips
              ? `M${p.x},${p.y} C${p.x + 4},${p.y + 66} ${q.x - 4},${q.y + 66} ${q.x},${q.y}`
              : `M${p.x},${p.y} C${p.x + dx},${p.y} ${q.x - dx},${q.y} ${q.x},${q.y}`;
            return (
              <path
                key={`${a}-${b}`}
                d={d}
                className={`edge ${hot ? "hot" : ""} ${toSel ? "win" : ""}`}
                vectorEffect="non-scaling-stroke"
              />
            );
          })}
        </svg>
        {nodes.map((n, i) => {
          if (n.kind === "SEL")
            return (
              <div
                key={n.id}
                className={`gnode sel-node ${sel.selector === "jev" ? "is-jev" : "is-agent"}`}
                style={{ left: `${n.x}%`, top: `${n.y}%`, animationDelay: `${i * 40}ms` }}
              >
                <span className="gtag">{sel.selector === "jev" ? "Jev" : "Agent"}</span>
                <strong>{sel.chosen_id} → {sel.chosen_label}</strong>
                <small>
                  {sel.selector === "jev"
                    ? "typed scores, ordered criteria"
                    : bundle.meta.jev_mode === "off"
                      ? "agent selector"
                      : "Jev deferred"}
                </small>
              </div>
            );
          if (n.kind === "LIB")
            return (
              <div key={n.id} className="gnode lib-node" style={{ left: `${n.x}%`, top: `${n.y}%`, animationDelay: `${i * 40}ms` }}>
                <span className="gtag">Library</span>
                <strong>{lessons}</strong>
                <small>{record.experience?.lesson ? "+1 lesson" : "no new lesson"}</small>
              </div>
            );
          const r = n.rollout;
          const stage = r && "dim" in r ? (r as Stage) : undefined;
          const bad = stage ? isBadStage(bundle, stage) : false;
          const chosen = n.id === sel.chosen_label;
          return (
            <button
              key={n.id}
              className={`gnode ${chosen ? "chosen" : ""} ${inspected === n.id ? "inspected" : ""} ${bad ? "bad" : ""} ${live.has(n.id) ? "" : "dim"}`}
              style={{ left: `${n.x}%`, top: `${n.y}%`, ["--kc" as string]: stageColor(n.kind), animationDelay: `${i * 40}ms` }}
              onClick={() => onInspect(n.id)}
            >
              <div className="gnode-row">
                <div>
                  <span className="gtag">{n.kind === "AGG" ? "COMPOSE" : n.kind}</span>
                  <span className="glabel">{n.id}</span>
                </div>
                <Ring value={r?.accuracy ?? 0} />
              </div>
              {stage && stage.fixes.length > 0 && (
                <div className="gfix">
                  {stage.fixes.map((f) => (
                    <i key={f} className={bundle.fixes[f]?.good ? "" : "bad"} title={bundle.fixes[f]?.title} />
                  ))}
                </div>
              )}
              {chosen && <span className="crown">adopted</span>}
            </button>
          );
        })}
        </div>
      </div>
    </section>
  );
}

function Ring({ value }: { value: number }) {
  const c = 2 * Math.PI * 13;
  return (
    <svg className="ring" viewBox="0 0 34 34" width={34} height={34}>
      <circle cx="17" cy="17" r="13" className="ring-bg" />
      <circle
        cx="17"
        cy="17"
        r="13"
        className="ring-fg"
        stroke={accuracyColor(value)}
        strokeDasharray={`${c * value} ${c}`}
        transform="rotate(-90 17 17)"
      />
      <text x="17" y="20.5" textAnchor="middle">
        {pct(value)}
      </text>
    </svg>
  );
}
