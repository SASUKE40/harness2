import { useEffect, useMemo, useRef, useState } from "react";
import { Header } from "./components/Header";
import { ImageGrid } from "./components/ImageGrid";
import { JevPanel } from "./components/JevPanel";
import { Kpis } from "./components/Kpis";
import { KnowledgeMatrix } from "./components/KnowledgeMatrix";
import { LessonCard } from "./components/LessonCard";
import { RecursionGraph } from "./components/RecursionGraph";
import { RsiCurve } from "./components/RsiCurve";
import { StageInspector } from "./components/StageInspector";
import { TaskRail } from "./components/TaskRail";
import { LiveDemo } from "./live/LiveDemo";
import { taskMetrics } from "./metrics";
import type { Bundle, Mode } from "./types";

const initial = new URLSearchParams(window.location.search);

export type Dataset = "aircraft" | "real" | "synthetic";
const DATASETS: Dataset[] = ["aircraft", "real", "synthetic"];
const FILES: Record<Dataset, string> = {
  aircraft: "/rsi-aircraft.json",
  real: "/rsi-real.json",
  synthetic: "/rsi.json",
};

export default function App() {
  const [bundle, setBundle] = useState<Bundle | null>(null);
  const [error, setError] = useState("");
  const [mode, setMode] = useState<Mode>(initial.get("mode") === "seq" ? "seq" : "agg");
  const [taskIndex, setTaskIndex] = useState(Math.max(0, Number(initial.get("task") ?? 1) - 1));
  const [inspected, setInspected] = useState<string | null>(initial.get("node"));
  const [playing, setPlaying] = useState(false);
  const [page, setPage] = useState<"live" | "dashboard">(initial.get("view") === "dashboard" ? "dashboard" : "live");

  const requested = initial.get("data");
  const [dataset, setDataset] = useState<Dataset>(
    requested === "synthetic" || requested === "real" ? requested : "aircraft",
  );

  useEffect(() => {
    const load = (name: string) =>
      fetch(FILES[name as Dataset]).then((r) =>
        r.ok && r.headers.get("content-type")?.includes("json")
          ? r.json()
          : Promise.reject(new Error(`HTTP ${r.status}`)),
      );
    // Fall back along aircraft -> real -> synthetic when a bundle has not been generated.
    const order = DATASETS.slice(DATASETS.indexOf(dataset));
    const attempt = (i: number): Promise<Bundle> =>
      load(order[i]).then(
        (b) => {
          if (order[i] !== dataset) setDataset(order[i]);
          return b;
        },
        (err) => (i + 1 < order.length ? attempt(i + 1) : Promise.reject(err)),
      );
    attempt(0)
      .then((b: Bundle) => {
        setBundle(b);
        if (!b.meta.modes.includes(mode)) setMode(b.meta.modes[0]);
        setTaskIndex((i) => Math.min(i, b.tasks.length - 1));
      })
      .catch((e: Error) => setError(e.message));
  }, []);

  const view = useRef(`${mode}-${taskIndex}`);
  useEffect(() => {
    if (view.current === `${mode}-${taskIndex}`) return;
    view.current = `${mode}-${taskIndex}`;
    setInspected(null);
  }, [taskIndex, mode]);

  useEffect(() => {
    const params = new URLSearchParams(page === "live" ? { mode } : { view: page, mode, task: String(taskIndex + 1) });
    if (dataset !== "aircraft") params.set("data", dataset);
    if (inspected && page === "dashboard") params.set("node", inspected);
    window.history.replaceState(null, "", `?${params}`);
  }, [mode, taskIndex, inspected, page, dataset]);

  const switchDataset = (next: Dataset) => {
    const params = new URLSearchParams(window.location.search);
    params.set("data", next);
    window.location.search = params.toString();
  };

  useEffect(() => {
    if (!playing || !bundle) return;
    const id = window.setInterval(() => {
      setTaskIndex((i) => {
        if (i >= bundle.tasks.length - 1) {
          setPlaying(false);
          return i;
        }
        return i + 1;
      });
    }, 2600);
    return () => window.clearInterval(id);
  }, [playing, bundle]);

  const metrics = useMemo(() => (bundle ? taskMetrics(bundle, mode) : []), [bundle, mode]);

  if (error) return <EmptyState error={error} />;
  if (!bundle || !metrics.length) return <div className="loading">Loading recursion record…</div>;
  if (page === "live")
    return (
      <LiveDemo
        bundle={bundle}
        metrics={metrics}
        mode={mode}
        setMode={setMode}
        onDashboard={() => setPage("dashboard")}
        dataset={dataset}
        onDataset={switchDataset}
        startStep={Number(initial.get("step") ?? 0) || 0}
      />
    );

  const current = metrics[Math.min(taskIndex, metrics.length - 1)];
  const record = current.record;
  const label = inspected ?? record.selection.chosen_label;

  const replay = () => {
    if (playing) return setPlaying(false);
    setTaskIndex(0);
    setPlaying(true);
  };

  return (
    <div className="app">
      <div className="bg-glow" aria-hidden />
      <Header
        bundle={bundle}
        mode={mode}
        setMode={setMode}
        playing={playing}
        onReplay={replay}
        onLive={() => setPage("live")}
      />
      <main>
        <section className="hero">
          <div>
            <p className="eyebrow">Recursive agent harnessing · open-world image labeling</p>
            <h1>
              The harness gets better at <span className="grad">seeing</span>, task after task.
            </h1>
            <p className="lede">
              Harness² refines the labeling harness on each task with Propose–Probe–Compose, Jev picks the
              final harness from its anonymized outputs, and adopted lessons flow into a library that makes
              the next task's first proposal smarter.
            </p>
          </div>
          <Kpis bundle={bundle} metrics={metrics} />
        </section>

        <section className="grid two">
          <RsiCurve metrics={metrics} selected={taskIndex} onSelect={setTaskIndex} />
          <KnowledgeMatrix bundle={bundle} metrics={metrics} selected={taskIndex} onSelect={setTaskIndex} />
        </section>

        <TaskRail metrics={metrics} selected={taskIndex} onSelect={setTaskIndex} />

        <RecursionGraph
          key={`${mode}-${current.task.id}`}
          bundle={bundle}
          mode={mode}
          metrics={current}
          inspected={label}
          onInspect={setInspected}
        />

        <section className="grid split">
          <ImageGrid metrics={current} inspected={label} />
          <JevPanel bundle={bundle} metrics={current} />
        </section>

        <section className="grid split">
          <StageInspector bundle={bundle} metrics={current} inspected={label} />
          <LessonCard bundle={bundle} metrics={current} />
        </section>
      </main>
      <footer>
        <p>
          <strong>About this run.</strong> {bundle.meta.editor}; executions run the real labeler scripts in each
          harness through the Harness² recursion, selection, reflection and library code.{" "}
          {bundle.meta.jev_mode === "offline"
            ? "Jev is an offline stand-in that follows Jev's response contract inside the real Jev request, validation and audit path; no API call is made. Run with --jev live for TypeSafe's Jev."
            : bundle.meta.jev_mode === "live"
              ? `Selections were made by TypeSafe's Jev (${bundle.meta.jev_model}).`
              : "Jev was disabled; the agent selector chose every final."}{" "}
          Accuracy is computed afterwards against reference labels that never enter any editor, selector or Jev
          context.
        </p>
        <p className="muted">
          Generated {new Date(bundle.meta.generated_at).toLocaleString()} · k={bundle.meta.k} ·{" "}
          {bundle.meta.executions} executions · domain <code>{bundle.meta.domain}</code>
        </p>
      </footer>
    </div>
  );
}

function EmptyState({ error }: { error: string }) {
  return (
    <div className="empty">
      <h1>No recursion record found</h1>
      <p>
        Could not load <code>/rsi.json</code> ({error}). Generate it from the repository root:
      </p>
      <pre>harness2-image-demo --output runs/image-demo --k 2 --export web/public</pre>
      <p className="muted">Then reload this page.</p>
    </div>
  );
}
