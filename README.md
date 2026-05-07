# 🛡️ The Guard — GrabOn AI Output Eval Pipeline

A production eval framework that detects quality regressions in GrabOn's AI-generated outputs across model swaps, prompt rewrites, and tool-chain changes — with statistical rigor, an agentic loop, prompt versioning, and a CI/CD gate that blocks bad deployments.

---

## Architecture

```
run_eval.py  (entrypoint + CLI)
│
├── agent.py                         ← Agentic loop: Plan/Act/Observe/Decide
│   ├── Phase: PLAN   — select providers + test cases based on git diff
│   ├── Phase: ACT    — call providers with retry + typed error handling
│   ├── Phase: OBSERVE — score via ToolRegistry (discoverable, not hardcoded)
│   └── Phase: DECIDE — statistical gate → GO / NO-GO / INCONCLUSIVE
│
├── providers/                        ← Pluggable LLM provider wrappers
│   ├── openai_provider.py            GPT-4o-mini — quality tasks ($0.15/1M in)
│   ├── gemini_provider.py            Gemini-Flash — shadow testing ($0.075/1M)
│   └── groq_provider.py             LLaMA-3-8B — classification ($0.05/1M)
│
├── tests/
│   └── golden_suite.py              Loads 140 cases from tests.json
│
├── tests.json                        Source of truth: 45 deal + 40 insurance + 55 credit
│
├── scorer/scorer.py                  7 scoring functions (all return 0–1 + 95% CI)
│   ├── format_compliance            max_chars / title+body split / coupon_any / expired penalty
│   ├── intent_match                 5 new insurance labels with legacy-label fallback
│   ├── factual_grounding            only numeric ground-truth keys, skips meta-flags
│   ├── llm_judge                    Real Anthropic Haiku API call; fallback to semantic
│   ├── semantic                     Cosine similarity via all-MiniLM-L6-v2
│   ├── exact                        Exact label match
│   └── json_match                   Field-level JSON comparison
│
├── stats/engine.py                   Statistical tests
│   ├── Paired bootstrap (primary)   accuracy scores — p-value + 95% CI
│   ├── McNemar's test               paired pass/fail counts
│   └── Welch's t-test               latency + cost (continuous metrics)
│
├── detector/detector.py              GO / NO-GO / INCONCLUSIVE gate
├── versioning/prompt_versioning.py   SHA-256 hash + git commit + diff
├── history/tracker.py                Append-only run history (JSON)
├── reports/generator.py             JSON + Markdown reports per run
├── errors.py                        Typed error taxonomy
├── .github/workflows/eval.yml       CI/CD gate — blocks PR on NO-GO
└── logs/eval.log                    Structured log (all phases + errors)
```

---

## The 3 GrabOn Eval Tasks

| Task | Cases | Channels / Labels | Scorer | Adversarial |
|------|-------|-------------------|--------|-------------|
| Deal Copy Quality | 45 | Email, Push, WhatsApp, Glance × English/Hindi/Telugu/Hinglish | `format_compliance` + `llm_judge` | 17 cases (expired offers, char overflow) |
| Insurance Intent | 40 | device_protection, travel_insurance, health_micro, accidental_damage, no_insurance | `intent_match` | 14 cases (ambiguous profiles) |
| Credit Narrative | 55 | — | `factual_grounding` | 5 edge cases (zero txns, negative GMV) |
| **Total** | **140** | | | |

---

## Statistical Approach

**Regression is flagged ONLY when both hold:**

1. **Effect size** ≥ 3 percentage points (MIN_EFFECT_SIZE = 0.03)
2. **Statistical significance** p < 0.05

This prevents "average went up 2% — is this real?" false alarms.

| Metric | Test | Why |
|--------|------|-----|
| Accuracy scores | **Paired Bootstrap** (2000 resamples) | Primary test. Paired on same test cases → correct for correlated samples. |
| Pass/fail counts | **McNemar's test** | Tests if pass→fail events exceed fail→pass — the right null for paired binary outcomes. |
| Latency / cost | **Welch's t-test** | Continuous, approximately normal — unpaired OK. |

Every result produces: `Δ`, `95% CI`, `p-value`, `effect_size` (Cohen's d).

**Is this 2% improvement real or noise?** → Read the p-value and CI. If CI contains 0 and p > 0.05, it's noise.

---

## Agent Loop (Plan/Act/Observe/Decide)

```
PLAN    → Select which providers + test cases to run
           Log tool registry for discoverability
ACT     → Call each provider with typed-error-aware retry
           Budget guard kills runaway if cost > MAX_COST_USD
           Retryable: APIRateLimitError, APITimeoutError, APIServerError (exp backoff)
           Non-retryable: ContentPolicyError, ProviderRefusalError (score=0, continue)
OBSERVE → Score results via ToolRegistry (not hardcoded calls)
           Each scorer is a named, discoverable tool
DECIDE  → Statistical gate → GO / NO-GO / INCONCLUSIVE
```

All phase transitions are logged with timestamps to `logs/eval.log`.

---

## Multi-LLM Cost Strategy

| Provider | Model | Task rationale | Cost (in/out per 1M tokens) |
|----------|-------|----------------|------------------------------|
| OpenAI | `gpt-4o-mini` | deal_copy, credit_narrative (quality-sensitive) | $0.15 / $0.60 |
| Gemini | `gemini-1.5-flash` | Shadow testing — validates regressions aren't provider-specific | $0.075 / $0.30 |
| Groq | `llama3-8b-8192` | insurance_intent classification — cheapest at scale | $0.05 / $0.08 |
| Haiku | `claude-haiku-4-5` | LLM-as-judge for deal copy (oracle, not generating) | $0.80 / $4.00 |

Shadow testing: Groq runs the same cases as OpenAI. Compare `deal_copy` accuracy between them to quantify the quality/cost tradeoff per task.

---

## GO / NO-GO Gate

```
GO            → All metrics within bounds, p-values non-significant. Safe to deploy.
NO-GO         → Regression detected. PR blocked. Report includes:
                  ▸ Which task type regressed (e.g. "deal_copy accuracy")
                  ▸ Delta + 95% CI ([−0.08, −0.02])
                  ▸ p-value (0.003)
                  ▸ Which specific test case IDs degraded
                  ▸ Prompt diff (what changed between versions)
INCONCLUSIVE  → First run (no baseline) or insufficient samples. Human review.
```

---

## Prompt Versioning

Every prompt is versioned by:
- **Content hash** (SHA-256[:12]) — detects any character change
- **Git commit** — ties version to exact code state
- **Git branch** — separates PR baselines from main
- **Diff** — `git diff HEAD~1 -- prompts/` shows what changed

On regression: the report includes the diff between the last two prompt versions for the regressed task type.

---

## What Broke First (Honest Post-Mortem)

1. **`score_format_compliance` read `max_length` but `tests.json` sends `max_chars`** — all 45 deal cases scored wrong (length_score always 1.0 because the key was missing). Fixed by reading both + push title/body split.

2. **`score_factual_grounding` iterated ALL dict keys including `adversarial`, `expected_claims`, `zero_transactions`** — boolean/list meta-flags inflated the denominator, tanking scores on valid narratives. Fixed by `_CREDIT_META_KEYS` skiplist.

3. **`score_intent_match` had old 5-label taxonomy** — `tests.json` uses completely different labels (`device_protection`, `health_micro` etc.). Fixed with explicit label set + legacy-label mapping.

4. **`score_llm_judge` was not calling any API** — it was reusing semantic similarity with a different weighting. Fixed with real Anthropic Haiku call + graceful fallback.

5. **Groq provider existed in `.env` but was never imported or wired** — `GROQ_API_KEY` was dead config. Fixed by adding `GroqProvider` to `providers/__init__.py` and `run_eval.py`.

6. **No agent loop** — `run_eval.py` was a flat script. The Plan/Act/Observe/Decide phases and ToolRegistry were entirely absent. Added `agent.py`.

7. **No budget guard** — a runaway eval could spend unbounded money. Fixed with `BudgetGuard` (default $5 hard limit).

8. **No typed errors** — all provider exceptions caught as `Exception`. Fixed with `errors.py` taxonomy (APIRateLimitError, APITimeoutError, etc.) and typed retry logic.

---

## What I'd Change Next

- **Real Telugu localization test** — currently detected via `metadata.language == "telugu"` but the scorer doesn't verify script. Would add a Unicode-range check.
- **Confidence calibration scoring** — 3 insurance cases have `expected_confidence_range`. Currently unused; would add a scorer that checks model confidence against the range.
- **Per-run cost budget by task type** — currently budget is global. Could set `$0.50 max for insurance_intent` separately since Groq is cheap enough.
- **SQLite history** instead of append-only JSON — querying "when did Telugu quality drop" is a SQL query, not file parsing.
- **Shadow deploy** — run new model in shadow before promoting to primary, diff outputs at scale before any CI gate.

---

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/the-guard
cd the-guard

pip install -r requirements.txt

cp .env.example .env
# Add: OPENAI_API_KEY, GEMINI_API_KEY, GROQ_API_KEY, ANTHROPIC_API_KEY

# First run (saves baseline)
python run_eval.py

# Subsequent runs compare against baseline
python run_eval.py

# Update baseline after intentional improvement
python run_eval.py --update-baseline
```

## CLI Reference

```bash
python run_eval.py                           # Full eval, all providers
python run_eval.py --provider groq           # Groq only (cheapest)
python run_eval.py --provider openai         # OpenAI only
python run_eval.py --task deal_copy          # One task type
python run_eval.py --dry-run                 # Validate setup, list tools
python run_eval.py --update-baseline         # Run + save as new baseline
python run_eval.py --simulate-regression     # Inject bad prompt, test gate
python run_eval.py --max-cost 1.00           # Hard budget limit $1
```

## CI/CD

- Every PR touching `prompts/`, `providers/`, `scorer/`, `stats/`, `tests.json` triggers eval
- Report posted as PR comment (Markdown)
- PR **blocked** on NO-GO (exit code 1)
- Baselines cached per branch in GitHub Actions cache

Add secrets: `OPENAI_API_KEY`, `GEMINI_API_KEY`, `GROQ_API_KEY`

## Exit Codes

| Code | Verdict | Meaning |
|------|---------|---------|
| `0` | GO | No regressions — safe to deploy |
| `1` | NO-GO | Regression detected — PR blocked |
| `2` | INCONCLUSIVE | First run / insufficient data |
