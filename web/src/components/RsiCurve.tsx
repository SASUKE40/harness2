import { useState } from "react";
import { pct, type TaskMetrics } from "../metrics";

const W = 760;
const H = 320;
const PAD = { l: 48, r: 24, t: 24, b: 78 };

const SERIES = [
  { key: "base", label: "Base harness", cls: "s-base" },
  { key: "probes", label: "Mean probe", cls: "s-probe" },
  { key: "firstProbe", label: "First proposal", cls: "s-first" },
  { key: "adopted", label: "Adopted by selector", cls: "s-adopted" },
] as const;

export function RsiCurve({
  metrics,
  selected,
  onSelect,
}: {
  metrics: TaskMetrics[];
  selected: number;
  onSelect: (i: number) => void;
}) {
  const [hover, setHover] = useState<number | null>(null);
  const n = metrics.length;
  const innerW = W - PAD.l - PAD.r;
  const plotH = H - PAD.t - PAD.b;
  const x = (i: number) => PAD.l + (n === 1 ? innerW / 2 : (i / (n - 1)) * innerW);
  const y = (v: number) => PAD.t + (1 - v) * plotH;
  const maxLib = Math.max(1, ...metrics.map((m) => m.libraryBefore + 1));
  const focus = hover ?? selected;
  const m = metrics[focus];

  const path = (key: (typeof SERIES)[number]["key"]) =>
    metrics.map((mm, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(mm[key]).toFixed(1)}`).join(" ");
  const area =
    `${path("adopted")} L${x(n - 1)},${y(0)} L${x(0)},${y(0)} Z`;

  return (
    <div className="card curve">
      <div className="card-head">
        <div>
          <h2>Recursive self-improvement curve</h2>
          <p className="muted">Post-hoc label accuracy across the task stream (same domain, library threaded forward)</p>
        </div>
        <div className="legend">
          {SERIES.map((s) => (
            <span key={s.key} className={`lg ${s.cls}`}>
              <i />
              {s.label}
            </span>
          ))}
          <span className="lg s-lib">
            <i />
            Library lessons
          </span>
        </div>
      </div>
      <div className="curve-body">
        <svg viewBox={`0 0 ${W} ${H}`} className="curve-svg" onMouseLeave={() => setHover(null)}>
          <defs>
            <linearGradient id="adoptedStroke" gradientUnits="userSpaceOnUse" x1={PAD.l} x2={W - PAD.r} y1={0} y2={0}>
              <stop offset="0" stopColor="#8b5cf6" />
              <stop offset="0.5" stopColor="#22d3ee" />
              <stop offset="1" stopColor="#34d399" />
            </linearGradient>
            <linearGradient id="adoptedFill" x1="0" x2="0" y1="0" y2="1">
              <stop offset="0" stopColor="#22d3ee" stopOpacity="0.2" />
              <stop offset="1" stopColor="#22d3ee" stopOpacity="0" />
            </linearGradient>
            <filter id="glow" filterUnits="userSpaceOnUse" x={0} y={0} width={W} height={H}>
              <feGaussianBlur stdDeviation="4" result="b" />
              <feMerge>
                <feMergeNode in="b" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>
          {[0, 0.25, 0.5, 0.75, 1].map((v) => (
            <g key={v}>
              <line x1={PAD.l} x2={W - PAD.r} y1={y(v)} y2={y(v)} className="gridline" />
              <text x={PAD.l - 10} y={y(v) + 4} className="axis" textAnchor="end">
                {v * 100}%
              </text>
            </g>
          ))}
          <rect
            x={x(focus) - 34}
            y={PAD.t - 8}
            width={68}
            height={H - PAD.t - 10}
            rx={14}
            className="focus-band"
          />
          {metrics.map((mm, i) => {
            const bh = ((mm.libraryBefore + (mm.record.experience?.lesson ? 1 : 0)) / maxLib) * 34;
            return (
              <g key={mm.task.id}>
                <rect x={x(i) - 9} y={H - 22 - bh} width={18} height={bh} rx={4} className="lib-bar" />
                <rect x={x(i) - 9} y={H - 22 - (mm.libraryBefore / maxLib) * 34} width={18} height={(mm.libraryBefore / maxLib) * 34} rx={4} className="lib-bar before" />
                <text x={x(i)} y={H - 6} className="axis" textAnchor="middle">
                  T{i + 1} · {mm.task.title.split(" ")[0]}
                </text>
              </g>
            );
          })}
          <path d={area} fill="url(#adoptedFill)" className="area" />
          {SERIES.map((s) => (
            <path key={s.key} d={path(s.key)} className={`line ${s.cls}`} pathLength={1} filter={s.key === "adopted" ? "url(#glow)" : undefined} />
          ))}
          {metrics.map((mm, i) => (
            <g key={mm.task.id} className="pts">
              <circle cx={x(i)} cy={y(mm.base)} r={4} className="pt s-base" />
              <circle cx={x(i)} cy={y(mm.firstProbe)} r={4.5} className="pt s-first" />
              <circle cx={x(i)} cy={y(mm.adopted)} r={11} className={`pt-sel ${mm.selector === "jev" ? "jev" : "agent"}`} />
              <text x={x(i)} y={y(mm.adopted) + 3.5} textAnchor="middle" className="pt-sel-txt">
                {mm.selector === "jev" ? "J" : "A"}
              </text>
            </g>
          ))}
          {metrics.map((mm, i) => (
            <rect
              key={mm.task.id}
              x={x(i) - 40}
              y={0}
              width={80}
              height={H}
              fill="transparent"
              style={{ cursor: "pointer" }}
              onMouseEnter={() => setHover(i)}
              onClick={() => onSelect(i)}
            />
          ))}
        </svg>
        <aside className="curve-tip">
          <span className="tip-step">Task {focus + 1}</span>
          <strong>{m.task.title}</strong>
          <dl>
            <dt>Base harness</dt>
            <dd>{pct(m.base)}</dd>
            <dt>First proposal</dt>
            <dd>{pct(m.firstProbe)}</dd>
            <dt>Mean probe</dt>
            <dd>{pct(m.probes)}</dd>
            <dt>Adopted</dt>
            <dd className="hl">{pct(m.adopted)}</dd>
            <dt>Best final</dt>
            <dd>{pct(m.oracle)}</dd>
            <dt>Library before</dt>
            <dd>{m.libraryBefore}</dd>
          </dl>
          <p className="muted small">
            <b className={m.selector === "jev" ? "c-jev" : "c-agent"}>{m.selector === "jev" ? "J" : "A"}</b>{" "}
            {m.selector === "jev" ? "Jev chose the final harness" : "Jev deferred; agent selector chose"}
          </p>
        </aside>
      </div>
    </div>
  );
}
