# Setup guide

Start with the [quick start](../README.md#install). This guide covers benchmark
dependencies, model credentials, and the optional Codex executor. For individual
benchmarks, see the [benchmark guides](BENCHMARKS.md).

## Requirements

Use Linux and Python 3.12 for benchmark runs; the Python package also supports
3.13. On Debian/Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y git curl unzip ca-certificates pandoc poppler-utils
```

Pandoc and Poppler are needed by LAB. Also install:

- [Node.js](https://nodejs.org/en/download) 20 or newer.
- [uv](https://docs.astral.sh/uv/getting-started/installation/).
- [Google Cloud CLI](https://cloud.google.com/sdk/docs/install).
- [Docker Engine](https://docs.docker.com/engine/install/) for WorkBuddy-Bench,
  with permission to run `docker info`.

Benchmark setup installs its own pinned Bun runtime under `dependencies/`.
Keep the main Harness² environment active when running the commands below.

## Models and credentials

Copy `.env.example` to `.env` if you have not already done so. Set your Vertex AI
project and model IDs, then authenticate:

```bash
gcloud auth application-default login
```

Shell environment variables take precedence over `.env`.
`GOOGLE_CLOUD_PROJECT` identifies your project; `GOOGLE_CLOUD_LOCATION` defaults
to `global`. Your account must have access to each configured model.

| Setting | Meaning |
|---|---|
| `HARNESS2_SOLVER_MODEL` | Executor model in OpenCode's `provider/model` format |
| `HARNESS2_IMPROVER_MODEL` | Improver model in the same format; defaults to the executor |
| `HARNESS2_JUDGE_MODEL` | Bare judge model ID; the adapter supplies the Vertex prefix |
| `HARNESS2_RUNS_DIR` | Output root; defaults to `runs/` |

The provided configuration uses Vertex Gemini executors and judges, and supports
either Vertex Claude or Gemini as the improver. LAB also supports the upstream
judge's direct providers when their credentials are configured. For WorkBuddy,
setup generates the Gemini solver and judge routes from these settings; rerun
setup after changing either model.

### Authentication alternatives

On a machine with a configured attached service account, Application Default
Credentials can use that account. Other options include:

- `gcloud auth application-default login --no-browser`, which prints a command
  to run on another machine that has a browser.
- `GOOGLE_APPLICATION_CREDENTIALS` pointing at a service account key file.
- `gcloud auth application-default login --impersonate-service-account=<account>`.

Keep `gcloud` on `PATH` for these Vertex configurations, including the key-file
route: the installation check runs
`gcloud auth application-default print-access-token`.

## Jev selection

[Jev](https://docs.typesafe.ai/introduction) returns typed decisions rather than
generated code. Harness² can use it to select among the final outputs of a
`k > 1` run. The executor, harness editor, composer, reflection, and benchmark
judges retain their existing models.

Install the optional SDK and set an API key from the
[TypeSafe console](https://console.typesafe.ai/):

```bash
uv pip install -e '.[jev]'
export TYPESAFE_API_KEY="your-api-key"
export HARNESS2_SELECTOR=jev
```

These settings can also go in your local `.env`. To try the integration with
deterministic local executors and editors, run:

```bash
harness2-demo --output runs/demo-jev --k 2
```

Only Jev makes paid API calls in this demo. Both final outputs may be equally
good, in which case Jev defers to the deterministic demo selector. To use Jev in
a configured benchmark, choose a fresh run directory:

```bash
HARNESS2_RUNS_DIR=runs/lab-jev harness2 --bench lab --k 2 --mode parallel \
  --only-domains antitrust-competition --max-tasks 1
```

Use the same `HARNESS2_RUNS_DIR` for judging and reporting. Selection settings
are recorded in the experiment identity; changing them requires a new output
directory. The ordinary agent selector remains the default when
`HARNESS2_SELECTOR` is unset. At `k=1`, the sole final is delivered without any
selector API call.

| Setting | Default | Meaning |
|---|---|---|
| `HARNESS2_SELECTOR` | `proposer` | `jev` enables Jev before the existing selector |
| `TYPESAFE_API_KEY` | required for Jev | API credential; excluded from run records |
| `HARNESS2_JEV_MODEL` | `jev-1.14.0` | Pinned model; use a versioned ID for reproducible experiments |
| `HARNESS2_JEV_MIN_CONFIDENCE` | `0.6` | Minimum confidence for every candidate's individual judgments |
| `HARNESS2_JEV_MAX_REQUEST_BYTES` | `28000` | Maximum complete JSON request size before deferring to the agent |

The request contains the task, complete sanitized text source files, and
anonymized candidate outputs. Jev answers one Score question per candidate and
dimension in a single call. Code compares the resulting tuples in this order:
delivery, format conformance, source fidelity, internal consistency, and
usability. Format conformance is omitted for benchmarks whose selection policy
does not require full conformance. Higher-priority dimensions take precedence
over all lower-priority dimensions.

This is an experimental selection policy, not a benchmark judge. The default
confidence threshold needs validation on your tasks; confidence describes the
answer distribution, not a guarantee of correctness. See TypeSafe's
[Score](https://docs.typesafe.ai/primitives/score) and
[confidence](https://docs.typesafe.ai/confidence) documentation.

Low confidence, tied leaders, invalid responses, and service failures defer to
the existing agent selector. So do binary sources, candidate attachments,
session-transcript evidence, and requests exceeding the local byte budget.
Evidence is never truncated to fit Jev. The API's actual token limits still
apply; byte size is only a conservative local guard. This initial integration
is most useful for compact text deliverables. The agent selector continues to
handle work requiring visual inspection or tools.

`selection.json` records the selector that actually chose the output, Jev's
answers, resolved model, token usage, elapsed time, and any deferral reason.
The full request and response are saved beside the selection context, for
example `selection/agg2.jev.json`, so the fallback selector cannot read Jev's
judgments. Known model versions include an estimated Jev charge in the
selection's `usd`, using the published $0.042 per million input tokens for Jev
1.13/1.14. Output tokens are free; unsuccessful/retried requests and unknown
model prices are not included in that estimate. Verify actual charges in the
console. [TypeSafe model reference](https://docs.typesafe.ai/models).

## Benchmark installation

Install only the benchmark you intend to run:

```bash
harness2-setup lab
harness2-check --bench lab
```

Use `jb` for JobBench or `wb` for WorkBuddy-Bench. See the
[benchmark guides](BENCHMARKS.md) for subset selection and licensing.

The runner overlays and compatibility patches are included in
[`third_party/`](../third_party/README.md). Setup clones supported upstream
revisions under `dependencies/`, applies the patches, creates the benchmark
environments, and downloads task data.

Each benchmark environment lives inside its upstream checkout:

- `dependencies/harvey-labs/.venv`
- `dependencies/job-bench-eval/.venv`
- `dependencies/workbuddy-bench/.venv`

Harness² selects these interpreters automatically; do not activate them for the
main commands.

### Installation checks

`harness2-setup <bench> --dry-run` previews setup without making changes.
`harness2-check` checks installed dependencies, credentials, and configuration
without calling a model. Refinement runs perform the same checks, including
judge configuration, before starting.

These checks do not test model availability. Access to an executor, improver, or
judge model is tested when that model is first called. Refinement and judging
return a nonzero exit status when incomplete.

## Codex executor

The Codex executor connects to a Responses API endpoint. For Gemini on Vertex
AI, the included `harness2-bridge` provides this interface through a local
LiteLLM server, using the same Application Default Credentials and
`GOOGLE_CLOUD_PROJECT`.

Install the optional dependencies and start the bridge in a separate session:

```bash
uv pip install -e '.[render,bridge]'
harness2-bridge                 # for WorkBuddy: harness2-bridge --host 0.0.0.0
```

The bridge prints the `HARNESS2_CODEX_*` settings to add to `.env`. Its port and
generated access key are stored under `dependencies/codex/` and reused on
restart. Pass `--substrate cx` to setup, checks, refinement, judging, and
reporting:

```bash
harness2-setup lab --substrate cx
harness2-check --bench lab --substrate cx
harness2 --bench lab --substrate cx --k 1 --mode parallel \
  --only-domains antitrust-competition --max-tasks 1
```

WorkBuddy executes Codex inside Docker containers, so the bridge must listen on
`--host 0.0.0.0`; it also prints `HARNESS2_CODEX_CONTAINER_URL`. This binds the
bridge to all network interfaces; restrict access to the intended clients.

To use another Responses API endpoint, set `HARNESS2_CODEX_BASE_URL`,
`HARNESS2_CODEX_API_KEY`, and `HARNESS2_CODEX_MODEL` directly. Setup records the
endpoint and model in `dependencies/codex/home/config.toml`; rerun setup after
changing either. `harness2-check` reports stale configuration until then.

Codex runs with its sandbox disabled. Review the
[executor network restrictions](EXPERIMENTS.md#web-access) before evaluation.

## Network access during setup

Setup downloads dependencies from the following hosts:

| Host | Used for |
|---|---|
| `github.com` | Pinned upstream checkouts |
| `registry.npmjs.org` | Pinned Bun runtime and Codex CLI, plus dependencies resolved by `bun install` |
| `huggingface.co` and its CDN | JobBench data and WorkBuddy subset archives |
| A Docker registry | WorkBuddy container image |
| `astral.sh` | CPython 3.12 when the host has no system Python 3.12 |

Use `harness2-setup <bench> --dry-run` to review setup commands when configuring
network access. Experiments also access the model endpoints configured in
`.env`. Setup downloads and executor web access have different requirements;
see [web access during evaluation](EXPERIMENTS.md#web-access).

## Wheel installations

A wheel ships the `harness2` package only. The task partitioner,
`python scripts/split_tasks.py`, requires this source tree. For a wheel
installation, work from a dedicated directory: `.env`, `dependencies/`, and
`runs/` are resolved there.
