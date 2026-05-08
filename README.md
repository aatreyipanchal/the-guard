# The Guard

An eval framework for GrabOn-style LLM outputs that catches regressions across model changes, prompt changes, and scoring/tooling changes before deployment.

I chose this assignment because it forces the full loop, not just prompt demos: dataset design, scoring, statistical comparison, reporting, CI gating, and historical tracking. The goal here was to build something that can answer "did quality really drop?" with evidence, not opinion.

## What I Built

- A scriptable eval pipeline with three task families:
  - `deal_copy`
  - `insurance_intent`
  - `credit_narrative`
- Multiple provider wrappers:
  - OpenAI `gpt-4o-mini`
  - Gemini `gemini-2.5-flash-lite`
  - Groq `llama-3.1-8b-instant`
- Numerical scorers with pass/fail thresholds
- Statistical baseline comparison
- GO / NO-GO / INCONCLUSIVE gate
- Prompt versioning with hash, git commit, branch, and task-level diffs
- Historical run tracking
- Query CLI plus a simple Streamlit dashboard
- GitHub Actions workflow that runs eval on PRs touching eval-relevant files

## Architecture

```text
                    +----------------------+
                    |     tests.json       |
                    |  140 source cases    |
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    | tests/golden_suite.py|
                    | prompt builders      |
                    | case metadata        |
                    +----------+-----------+
                               |
                               v
 +-------------+    +----------------------+    +----------------------+
 | providers/* | -> |      agent.py        | -> |    scorer/scorer.py  |
 | openai      |    | PLAN / ACT / OBSERVE |    | numerical scorers    |
 | gemini      |    | retries / budget     |    | pass/fail thresholds |
 | groq        |    +----------+-----------+    +----------+-----------+
 +-------------+               |                           |
                               +-------------+-------------+
                                             |
                                             v
                               +---------------------------+
                               |   detector/detector.py    |
                               | baseline comparison       |
                               | GO / NO-GO / INCONCLUSIVE |
                               +-------------+-------------+
                                             |
                      +----------------------+----------------------+
                      |                                             |
                      v                                             v
        +-----------------------------+              +-----------------------------+
        | versioning/prompt_versioning|              | reports/generator.py        |
        | hash / git / prompt diff    |              | raw JSON + markdown report  |
        +-----------------------------+              +-----------------------------+
                      |
                      v
        +-----------------------------+
        | history/tracker.py          |
        | eval_history.json           |
        +-------------+---------------+
                      |
                      v
        +-----------------------------+
        | dashboard/query.py          |
        | dashboard/app.py            |
        | CLI + Streamlit dashboard   |
        +-----------------------------+
```

## Per-Module Design Decisions And Tradeoffs

### `run_eval.py`

- Single entrypoint so the eval can run locally and in CI the same way.
- Keeps orchestration explicit instead of hiding it behind a framework.
- Tradeoff: the file is still central and somewhat large.

### `agent.py`

- Added a clear PLAN / ACT / OBSERVE / DECIDE loop so provider execution, retries, and budget handling are isolated from reporting.
- Provider calls are sequential for simplicity and determinism.
- Tradeoff: simpler control flow, slower than a bounded-concurrency runner.

### `providers/*`

- Separate wrappers per provider with common `ProviderResponse`.
- OpenAI now uses task-specific token caps to reduce latency.
- Gemini response serialization is defensive across SDK shapes.
- Tradeoff: wrappers are straightforward, but capabilities are normalized to the lowest common structure.

### `tests/golden_suite.py`

- `tests.json` is the source of truth; prompts are built from structured fixtures at runtime.
- This makes prompt versioning meaningful because the actual prompt text is generated and stored.
- Tradeoff: prompt changes can come from fixture or builder edits, so both need to be tracked.

### `scorer/scorer.py`

- Numerical scores are first-class; pass/fail is derived from score thresholds.
- Includes task-appropriate scoring instead of one generic similarity metric.
- Tradeoff: simple heuristics are easier to audit, but less expressive than richer judges for every task.

### `stats/engine.py`

- Uses paired bootstrap for accuracy, McNemar for pass/fail, Welch’s t-test for latency/cost.
- This is enough to defend claims of regression beyond raw averages.
- Tradeoff: still lightweight; not a full experiment analysis framework.

### `detector/detector.py`

- Stores baselines per provider and emits a direct deployment decision.
- Prompt diffs are attached only for regressed task types.
- Tradeoff: baseline files are local artifacts unless externalized or cached in CI.

### `versioning/prompt_versioning.py`

- Stores prompt bundles by task and case, not just one concatenated hash.
- Falls back to reading `.git/HEAD` when `git` is installed but missing from `PATH`.
- Tradeoff: prompt history can become large because full prompt text is persisted.

### `history/tracker.py` and `dashboard/*`

- Stores enough dimensions to answer historical questions by task, model, prompt version, and language.
- Includes a Streamlit dashboard for quick inspection and a CLI for scriptable queries.
- Tradeoff: history is JSON, not SQLite; fine for assessment size, weaker for long-term scaling.

### `.github/workflows/eval.yml`

- Runs eval when eval-relevant files change and blocks PRs on `NO-GO`.
- Captures the actual eval exit code instead of trying to infer outcome from logs.
- Tradeoff: `INCONCLUSIVE` also blocks, which is safer but can slow iteration if baselines are missing.

## How To Run

### Dependencies

- Python 3.10+ recommended
- Provider and tooling dependencies in `requirements.txt`

Install:

```bash
pip install -r requirements.txt
```

### Environment Variables

Set these in your shell or `.env`:

```bash
OPENAI_API_KEY=...
GEMINI_API_KEY=...
GROQ_API_KEY=...
ANTHROPIC_API_KEY=...
```

Notes:

- `OPENAI_API_KEY` is required for OpenAI eval runs
- `GEMINI_API_KEY` is required for Gemini eval runs
- `GROQ_API_KEY` is required for Groq eval runs
- `ANTHROPIC_API_KEY` is only needed for the LLM judge path

### Eval Commands

Full run:

```bash
python run_eval.py
```

Provider-specific:

```bash
python run_eval.py --provider openai
python run_eval.py --provider gemini
python run_eval.py --provider groq
```

Task-specific:

```bash
python run_eval.py --task deal_copy
python run_eval.py --task insurance_intent
python run_eval.py --task credit_narrative
```

Setup check:

```bash
python run_eval.py --dry-run
```

Update baseline intentionally:

```bash
python run_eval.py --update-baseline
```

Budget override:

```bash
python run_eval.py --max-cost 1.00
```

### Dashboard CLI

Historical slice:

```bash
python -m dashboard.query history --task-type deal_copy --language telugu
```

Drop detector:

```bash
python -m dashboard.query drop --language telugu --task-type deal_copy
```

### Streamlit Dashboard

```bash
streamlit run dashboard/app.py
```

## Eval Results

Latest full raw report in this repo:

- JSON: [reports/run_20260508_045023_c7ce95.json](/d:/Projects/the-guard/reports/run_20260508_045023_c7ce95.json:1)
- Markdown: [reports/run_20260508_045023_c7ce95.md](/d:/Projects/the-guard/reports/run_20260508_045023_c7ce95.md:1)

Key metrics from that run:

### OpenAI

- Model: `gpt-4o-mini`
- Decision: `GO`
- Pass rate: `67.9%`
- Mean accuracy: `0.7981`
- Avg latency: `2365.6 ms`
- Total cost: `$0.007930`

### Gemini

- Model: `gemini-2.5-flash-lite`
- Decision: `NO-GO`
- Pass rate: `68.6%`
- Mean accuracy: `0.8207`
- Avg latency: `1435.1 ms`
- Total cost: `$0.004797`

### Groq

- Model: `llama-3.1-8b-instant`
- Decision: `GO`
- Raw details are in the same report file

The raw eval report includes:

- pass/fail per test case
- numerical scores
- cost
- latency
- statistical comparison against baseline

That report is the artifact to submit for the "actual eval output" requirement.

## CI/CD Integration

Workflow file:

- [.github/workflows/eval.yml](/d:/Projects/the-guard/.github/workflows/eval.yml:1)

Behavior:

- Runs on PRs touching prompts, providers, scoring, detector, versioning, dashboard, history, tests, or workflow code
- Runs `run_eval.py`
- Uploads raw reports
- Posts markdown report to the PR
- Blocks merge on `NO-GO`
- Blocks `INCONCLUSIVE` for manual review

Repository settings note:

- GitHub branch protection must mark this workflow as a required status check for PR blocking to be enforced.

## What Broke First

Only the major bugs are listed here.

1. `score_format_compliance` was reading the wrong length key.
   `tests.json` used `max_chars`, while the scorer expected `max_length`. That made deal-copy length checks effectively wrong across the suite. I fixed it by supporting both shapes and the title/body split path.

2. Gemini calls were succeeding, then failing during response serialization.
   The provider tried to call `candidate.to_dict()`, which was not valid for the installed SDK object shape and converted successful generations into provider errors. I replaced it with defensive serialization that supports multiple SDK representations.

3. Prompt versioning looked implemented in the README, but it was not actually diffing prompt versions.
   The old code stored hash/timestamp metadata but did not retain prompt text by task, so there was no real baseline-vs-current prompt diff. I fixed this by storing prompt bundles, per-task hashes, and unified diffs for regressed task types.

4. Git metadata kept showing `unknown`.
   The repo was a valid git worktree, but `git` was not on `PATH` in the environment. I fixed this by resolving common Windows git paths and falling back to reading `.git/HEAD` directly.

5. Report generation crashed on JSON serialization.
   Statistical results contained NumPy scalar booleans/floats, which `json.dump` rejected. I added recursive normalization before writing report JSON.

## What I Would Change With 2 More Weeks

- Move history storage from JSON to SQLite so questions like "when did Telugu quality drop?" become fast, reliable queries.
- Add bounded concurrency to provider execution with per-provider rate-limit controls to reduce total eval time.
- Add a true Telugu script/quality validator instead of language-only grouping.
- Split baselines into local vs canonical CI baselines more cleanly.
- Add first-class model config files instead of embedding all provider config in Python classes.
- Add richer dashboard visualizations and commit links back to GitHub.
- Add unit tests around prompt versioning, report generation, and dashboard queries.

## Exit Codes

| Code | Meaning |
|------|---------|
| `0` | GO |
| `1` | NO-GO |
| `2` | INCONCLUSIVE |
