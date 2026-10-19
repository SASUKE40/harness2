import { mean, pct, type TaskMetrics } from "../metrics";
import type { Bundle } from "../types";

export function Kpis({ bundle, metrics }: { bundle: Bundle; metrics: TaskMetrics[] }) {
  const base = mean(metrics.map((m) => m.base));
  const adopted = mean(metrics.map((m) => m.adopted));
  const first = metrics[0];
  const last = metrics[metrics.length - 1];
  const decided = metrics.filter((m) => m.selector === "jev").length;
  const regret = mean(metrics.map((m) => m.regret));
  const bad = metrics.reduce((n, m) => n + m.badProbes, 0);
  const badAdopted = metrics.filter((m) =>
    (m.record.stages.find((s) => s.label === m.record.selection.chosen_label)?.fixes ?? []).some(
      (f) => !bundle.fixes[f]?.good,
    ),
  ).length;
  const lessons = last.libraryBefore + (last.record.experience?.lesson ? 1 : 0);
  const learned = new Set(metrics.flatMap((m) => m.learned)).size;

  return (
    <div className="kpis">
      <Kpi
        tone="ok"
        label="Image accuracy"
        value={
          <>
            <span className="from">{pct(base)}</span>
            <span className="arrow">→</span>
            {pct(adopted)}
          </>
        }
        foot={`+${Math.round((adopted - base) * 100)} pts over the base harness, mean of ${metrics.length} tasks`}
      />
      <Kpi
        tone="lib"
        label="First proposal accuracy"
        value={
          <>
            <span className="from">{pct(first.firstProbe)}</span>
            <span className="arrow">→</span>
            {pct(last.firstProbe)}
          </>
        }
        foot="Task 1 vs last task: library lessons make the first probe smarter"
      />
      <Kpi
        tone="jev"
        label="Jev selection"
        value={
          <>
            {decided}
            <small>/{metrics.length} decided</small>
          </>
        }
        foot={`${metrics.length - decided} deferred to the agent · ${Math.round(regret * 100)} pts regret vs best final`}
      />
      <Kpi
        tone="bad"
        label="Wrong rules filtered"
        value={
          <>
            {bad - badAdopted}
            <small>/{bad} probes</small>
          </>
        }
        foot="Plausible but wrong edits probed, then rejected at selection"
      />
      <Kpi
        tone="inv"
        label="Library"
        value={
          <>
            {lessons}
            <small> lessons</small>
          </>
        }
        foot={`${learned} harness edits learned · ${bundle.meta.executions} executions`}
      />
    </div>
  );
}

function Kpi({ label, value, foot, tone }: { label: string; value: React.ReactNode; foot: string; tone: string }) {
  return (
    <div className={`kpi tone-${tone}`}>
      <span className="kpi-label">{label}</span>
      <span className="kpi-value">{value}</span>
      <span className="kpi-foot">{foot}</span>
    </div>
  );
}
