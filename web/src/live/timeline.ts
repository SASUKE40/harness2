import type { TaskMetrics } from "../metrics";
import type { Stage } from "../types";

export type Phase = "observe" | "propose" | "probe" | "compose" | "select" | "reflect" | "deploy" | "summary";

export interface Glyph {
  g: "✓" | "✗" | "⇄";
  tone: "ok" | "bad" | "mid";
}

export interface Step {
  ti: number;
  phase: Phase;
  ms: number;
  stage?: Stage;
  hyp: number;
  round: number;
  rounds: number;
  deployed: string;
  adapted: boolean;
  glyphs: Glyph[];
  store: string[];
  fresh: string[];
  lessons: number;
  adoptions: number;
  lastProbe?: Stage;
  lastCompose?: Stage;
  decided: number;
  deferred: number;
}

const DURATION: Record<Phase, number> = {
  observe: 4200,
  propose: 1500,
  probe: 1500,
  compose: 1800,
  select: 3200,
  reflect: 2400,
  deploy: 7000,
  summary: 9000,
};

function glyph(accuracy: number, base: number): Glyph {
  if (accuracy >= 0.999) return { g: "✓", tone: "ok" };
  if (accuracy > base) return { g: "⇄", tone: "mid" };
  return { g: "✗", tone: "bad" };
}

/** One scripted walk through the recorded stream: every value shown comes from the run. */
export function buildTimeline(metrics: TaskMetrics[]): Step[] {
  const steps: Step[] = [];
  const store: string[] = [];
  let decided = 0;
  let deferred = 0;
  metrics.forEach((m, ti) => {
    const { task, record } = m;
    const rounds = record.finals.length;
    const base = task.base[0];
    const glyphs: Glyph[] = Object.values(base.per_image).map((ok) => (ok ? { g: "✓", tone: "ok" } : { g: "✗", tone: "bad" }));
    let lastProbe: Stage | undefined;
    let lastCompose: Stage | undefined;
    let hyp = 0;
    const push = (phase: Phase, extra: Partial<Step> = {}) =>
      steps.push({
        ti,
        phase,
        ms: DURATION[phase],
        hyp,
        round: 1,
        rounds,
        deployed: base.label,
        adapted: false,
        glyphs: [...glyphs],
        store: [...store],
        fresh: [],
        lessons: m.libraryBefore,
        adoptions: ti,
        lastProbe,
        lastCompose,
        decided,
        deferred,
        ...extra,
      });

    push("observe");
    for (const stage of record.stages) {
      const round = stage.dim === "AGG" ? rounds : Number(stage.label.replace(/\D/g, "")) || 1;
      if (stage.dim === "INV" || stage.dim === "DOM") {
        hyp += 1;
        push("propose", { stage, round });
        lastProbe = stage;
        glyphs.push(glyph(stage.accuracy, m.base));
        push("probe", { stage, round, lastProbe });
      } else {
        lastCompose = stage;
        glyphs.push(glyph(stage.accuracy, m.base));
        push("compose", { stage, round, lastCompose });
      }
    }
    if (record.selection.selector === "jev") decided += 1;
    else deferred += 1;
    push("select", { round: rounds, decided, deferred });
    const fresh = m.learned.filter((f) => !store.includes(f));
    store.push(...fresh);
    const lessons = m.libraryBefore + (record.experience?.lesson ? 1 : 0);
    push("reflect", { round: rounds, fresh, store: [...store], lessons, decided, deferred });
    push("deploy", {
      round: rounds,
      store: [...store],
      lessons,
      deployed: record.selection.chosen_label,
      adapted: true,
      adoptions: ti + 1,
      decided,
      deferred,
    });
  });
  const last = steps[steps.length - 1];
  if (last) steps.push({ ...last, phase: "summary", ms: DURATION.summary, fresh: [] });
  return steps;
}
