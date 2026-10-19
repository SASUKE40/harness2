import type { Bundle, ImageRecord, Mode, ModeRecord, Rollout, Stage, Task } from "./types";

export const DIMENSIONS = ["delivery", "conformance", "fidelity", "consistency", "usability"];

export const mean = (xs: number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0);

export const pct = (x: number, digits = 0) => `${(x * 100).toFixed(digits)}%`;

export function chosenStage(record: ModeRecord): Stage | undefined {
  return record.stages.find((s) => s.label === record.selection.chosen_label);
}

export function stageOrBase(task: Task, record: ModeRecord | undefined, label: string): Rollout | undefined {
  return record?.stages.find((s) => s.label === label) ?? task.base.find((b) => b.label === label);
}

export function isBadStage(bundle: Bundle, stage: Stage) {
  return stage.fixes.some((f) => bundle.fixes[f] && !bundle.fixes[f].good);
}

export interface TaskMetrics {
  task: Task;
  record: ModeRecord;
  base: number;
  firstProbe: number;
  probes: number;
  adopted: number;
  oracle: number;
  regret: number;
  badProbes: number;
  needed: string[];
  transferred: string[];
  learned: string[];
  knownBefore: string[];
  selector: string;
  deferred: boolean;
  libraryBefore: number;
}

export function taskMetrics(bundle: Bundle, mode: Mode): TaskMetrics[] {
  const known = new Set<string>();
  const out: TaskMetrics[] = [];
  for (const task of bundle.tasks) {
    const record = task.modes[mode];
    if (!record) continue;
    const probes = record.stages.filter((s) => s.dim === "INV" || s.dim === "DOM");
    const finals = record.stages.filter((s) => record.finals.includes(s.label));
    const chosen = chosenStage(record);
    const needed = Object.entries(bundle.fixes)
      .filter(([, f]) => f.good && f.warning && task.base_warnings.some((w) => w.includes(f.warning)))
      .map(([name]) => name);
    const adoptedGood = (chosen?.fixes ?? []).filter((f) => bundle.fixes[f]?.good && bundle.fixes[f]?.warning);
    const knownBefore = [...known];
    const transferred = needed.filter((f) => known.has(f));
    const learned = adoptedGood.filter((f) => !known.has(f));
    if (record.experience?.lesson) adoptedGood.forEach((f) => known.add(f));
    const oracle = Math.max(...finals.map((s) => s.accuracy));
    const adopted = chosen?.accuracy ?? 0;
    out.push({
      task,
      record,
      base: mean(task.base.map((b) => b.accuracy)),
      firstProbe: probes[0]?.accuracy ?? 0,
      probes: mean(probes.map((s) => s.accuracy)),
      adopted,
      oracle,
      regret: oracle - adopted,
      badProbes: probes.filter((s) => isBadStage(bundle, s)).length,
      needed,
      transferred,
      learned,
      knownBefore,
      selector: record.selection.selector,
      deferred: bundle.meta.jev_mode !== "off" && record.selection.selector !== "jev",
      libraryBefore: task.library_before[mode] ?? 0,
    });
  }
  return out;
}

export function svgUri(svg: string) {
  return `data:image/svg+xml;utf8,${encodeURIComponent(svg)}`;
}

export function imageSrc(img: ImageRecord) {
  return img.src ?? svgUri(img.svg ?? "");
}

/** Pixel size of the coordinate space boxes are given in. */
export function imageSize(img: ImageRecord): [number, number] {
  return img.svg ? [160, 120] : [img.width ?? 416, img.height ?? 416];
}

export function stageColor(dim: string) {
  return (
    {
      BASE: "var(--c-base)",
      INV: "var(--c-inv)",
      DOM: "var(--c-dom)",
      AGG: "var(--c-compose)",
      COMPOSE: "var(--c-compose)",
    } as Record<string, string>
  )[dim] ?? "var(--c-base)";
}

export function accuracyColor(x: number) {
  if (x >= 0.999) return "var(--ok)";
  if (x >= 0.6) return "var(--warn)";
  return "var(--bad)";
}
