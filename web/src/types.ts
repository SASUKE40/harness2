export type Mode = "agg" | "seq";

export interface FixInfo {
  title: string;
  kind: "INV" | "DOM";
  good: boolean;
  lesson: string;
  alternative: string;
  warning: string;
  trigger: string;
  action: string;
  tool: string;
}

export interface Box {
  label: string;
  bbox: [number, number, number, number];
}

export interface Detection extends Box {
  status: "hit" | "wrong" | "extra" | "missed";
  expected?: string;
  confidence?: number;
}

export interface Rollout {
  label: string;
  accuracy: number;
  f1: number;
  per_image: Record<string, boolean>;
  labels: Record<string, string[]>;
  detections: Record<string, Detection[]>;
}

export interface Stage extends Rollout {
  dim: "INV" | "DOM" | "AGG" | "COMPOSE";
  notes: string;
  fell_back: boolean;
  round: number;
  fixes: string[];
  tools: string[];
  prompt_rules: string[];
  diff: string;
  warnings: string[];
}

export interface JevAnswer {
  type: string;
  score: number;
  confidence: number;
  probabilities: Record<string, number>;
}

export interface JevDetails {
  requested_model?: string;
  model?: string | null;
  usage?: { input_tokens?: number; output_tokens?: number } | null;
  answers?: Record<string, JevAnswer>;
  ranks?: Record<string, number[]>;
  fallback_reason?: string;
  elapsed_s?: number;
  estimated_usd?: number;
}

export interface Experience {
  cue?: string;
  lesson?: string;
  evidence?: string;
  confidence?: string;
  edits?: string;
}

export interface ModeRecord {
  finals: string[];
  stages: Stage[];
  selection: {
    selector: "jev" | "proposer" | "trivial";
    chosen_id: string;
    chosen_label: string;
    id_map: Record<string, string>;
    why: string;
    usd: number;
    ok: boolean;
    fallback: boolean;
  };
  jev: JevDetails;
  selector_config: { backend?: string; model?: string; min_confidence?: number };
  jev_request: {
    bytes: number;
    questions: Record<string, { criteria: string[] }>;
    source_files: string[];
  };
  experience: Experience;
  ledger: { R: number; P: number; usd: number };
}

export interface ImageRecord {
  name: string;
  svg?: string;
  src?: string;
  width?: number;
  height?: number;
  truth: string[];
  boxes: Box[];
}

export interface Task {
  id: string;
  title: string;
  status: string;
  images: ImageRecord[];
  base: Rollout[];
  base_warnings: string[];
  library_before: Record<Mode, number>;
  modes: Partial<Record<Mode, ModeRecord>>;
}

export interface Bundle {
  meta: {
    title: string;
    generated_at: string;
    k: number;
    modes: Mode[];
    domain: string;
    jev_mode: "offline" | "live" | "off";
    jev_model: string;
    executor?: "labeler" | "jev";
    source?: "synthetic" | "real" | "aircraft";
    dataset?: { name: string; url: string; license: string } | null;
    executions: number;
    editor: string;
  };
  taxonomy: string;
  fixes: Record<string, FixInfo>;
  tasks: Task[];
}
