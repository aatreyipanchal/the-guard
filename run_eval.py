"""
run_eval.py — The Guard eval pipeline entrypoint.

Usage:
  python run_eval.py                        # run full eval vs baseline
  python run_eval.py --update-baseline      # run eval + save as new baseline
  python run_eval.py --provider openai      # run only one provider
  python run_eval.py --provider groq        # run Groq (LLaMA-3) — cheapest
  python run_eval.py --task deal_copy       # run only one task type
  python run_eval.py --dry-run              # validate setup without API calls
  python run_eval.py --simulate-regression  # inject bad prompt to test the gate

Exit codes:
  0 — GO (no regressions)
  1 — NO-GO (regressions detected — CI blocks merge)
  2 — INCONCLUSIVE (insufficient data, human review required)

Multi-LLM cost rationale (documented for reviewers):
  Provider   | Model             | Best for                     | $/1M tokens
  -----------|-------------------|------------------------------|-------------
  OpenAI     | gpt-4o-mini       | deal_copy, credit_narrative  | $0.15 in
  Gemini     | gemini-1.5-flash  | shadow testing, cross-val    | $0.075 in
  Groq       | llama3-8b-8192    | classification, intent tasks | $0.05 in
  Haiku      | claude-haiku-*    | LLM-as-judge (cheap oracle)  | $0.80 in

Shadow testing: Groq runs same cases as OpenAI → quantify per-task cost tradeoff.
"""

import argparse
import logging
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

load_dotenv()

Path("logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    handlers=[
        logging.FileHandler("logs/eval.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("the_guard")

from tests.golden_suite import ALL_TEST_CASES, TestCase
from providers import OpenAIProvider, GeminiProvider, GroqProvider, BaseProvider, ProviderResponse
from detector.detector import detect_regressions, build_snapshot
from reports.generator import generate_json_report, generate_markdown_report
from versioning.prompt_versioning import build_prompt_bundle, create_prompt_metadata
from history.tracker import append_run
from agent import EvalAgent, BudgetGuard, build_tool_registry

console = Console()


def build_providers(provider_filter: str | None) -> list[BaseProvider]:
    candidates = []

    if provider_filter in (None, "openai"):
        if OpenAIProvider is None:
            console.print("[yellow]⚠ Skipping OpenAI: openai package not installed.[/yellow]")
        else:
            try:
                candidates.append(OpenAIProvider())
                logger.info("Provider: openai/gpt-4o-mini — deal_copy, credit_narrative tasks")
            except EnvironmentError as e:
                console.print(f"[yellow]⚠ Skipping OpenAI: {e}[/yellow]")

    if provider_filter in (None, "gemini"):
        if GeminiProvider is None:
            console.print("[yellow]⚠ Skipping Gemini: google-generativeai package not installed.[/yellow]")
        else:
            try:
                candidates.append(GeminiProvider())
                logger.info("Provider: gemini/gemini-2.5-flash-lite — shadow testing")
            except EnvironmentError as e:
                console.print(f"[yellow]⚠ Skipping Gemini: {e}[/yellow]")

    if provider_filter in (None, "groq"):
        if GroqProvider is None:
            console.print("[yellow]⚠ Skipping Groq: groq package not installed.[/yellow]")
        else:
            try:
                candidates.append(GroqProvider())
                logger.info("Provider: groq/llama-3.1-8b-instant — cheap classification baseline")
            except EnvironmentError as e:
                console.print(f"[yellow]⚠ Skipping Groq: {e}[/yellow]")

    return candidates


def compute_task_cost_breakdown(responses: list[ProviderResponse], test_cases: list[TestCase]) -> dict:
    tc_map = {tc.id: tc for tc in test_cases}
    breakdown: dict[str, dict] = {}
    for resp in responses:
        tc = tc_map.get(resp.test_id)
        task = tc.task_type if tc else "unknown"
        if task not in breakdown:
            breakdown[task] = {"cost_usd": 0.0, "tokens": 0, "count": 0}
        breakdown[task]["cost_usd"] += resp.cost_usd
        breakdown[task]["tokens"]   += resp.total_tokens
        breakdown[task]["count"]    += 1
    return breakdown


def print_summary_table(provider_name: str, responses: list, scores_list: list, test_cases: list) -> None:
    table = Table(title=f"{provider_name.upper()} — Per-test results", show_lines=False)
    table.add_column("ID",      style="dim",  width=12)
    table.add_column("Type",    style="cyan", width=18)
    table.add_column("Method",  style="dim",  width=16)
    table.add_column("Score",   justify="right")
    table.add_column("Pass",    justify="center")
    table.add_column("Latency", justify="right", style="dim")
    table.add_column("Cost $",  justify="right", style="dim")

    tc_map   = {tc.id: tc for tc in test_cases}
    resp_map = {r.test_id: r for r in responses}

    for s in scores_list:
        tc   = tc_map.get(s.test_id)
        resp = resp_map.get(s.test_id)
        table.add_row(
            s.test_id,
            tc.task_type if tc else "—",
            s.scoring_method,
            f"{s.score:.3f}",
            "[green]✓[/green]" if s.passed else "[red]✗[/red]",
            f"{resp.latency_ms:.0f}ms" if resp else "—",
            f"{resp.cost_usd:.6f}"     if resp else "—",
        )
    console.print(table)


def print_cost_breakdown(provider_name: str, breakdown: dict) -> None:
    table = Table(title=f"{provider_name.upper()} — Per-task cost breakdown", show_lines=False)
    table.add_column("Task type",  style="cyan", width=22)
    table.add_column("Cases",      justify="right")
    table.add_column("Tokens",     justify="right")
    table.add_column("Cost $",     justify="right")
    table.add_column("$/case",     justify="right", style="dim")

    for task, data in sorted(breakdown.items()):
        per_case = data["cost_usd"] / data["count"] if data["count"] else 0
        table.add_row(
            task,
            str(data["count"]),
            f"{data['tokens']:,}",
            f"{data['cost_usd']:.6f}",
            f"{per_case:.6f}",
        )
    console.print(table)


def _simulate_bad_prompt():
    """Corrupt the insurance prompt slightly to test the NO-GO gate."""
    try:
        from versioning.prompt_versioning import create_prompt_metadata
        bad = "Classify insurance. Always return 'no_insurance' unless you are certain. Return only one label."
        create_prompt_metadata(bad, provider="simulation")
        console.print("[yellow]⚠ Simulated bad prompt version registered[/yellow]")
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="The Guard — GrabOn LLM Eval Pipeline")
    parser.add_argument("--update-baseline", action="store_true")
    parser.add_argument("--provider", choices=["openai", "gemini", "groq"])
    parser.add_argument("--task", choices=[
        "deal_copy", "insurance_intent", "credit_narrative",
        "summarization", "classification", "extraction",
    ])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--simulate-regression", action="store_true")
    parser.add_argument("--max-cost", type=float, default=5.0,
                        help="Hard budget limit in USD (default: $5.00)")
    args = parser.parse_args()

    run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    console.rule(f"[bold]🛡️ The Guard — GrabOn Eval — {run_id}[/bold]")
    logger.info(f"=== Run started: {run_id} ===")

    if args.simulate_regression:
        _simulate_bad_prompt()

    providers = build_providers(args.provider)
    if not providers:
        console.print("[red]No providers available. Set at least one API key in .env[/red]")
        sys.exit(1)

    test_cases = [c for c in ALL_TEST_CASES if not args.task or c.task_type == args.task]
    suite_stats = {
        "total": len(test_cases),
        **{tt: len([c for c in test_cases if c.task_type == tt])
           for tt in ["deal_copy", "insurance_intent", "credit_narrative",
                      "summarization", "classification", "extraction"]},
    }

    console.print(f"\n[bold]Suite:[/bold] {suite_stats['total']} tests | "
                  f"deal={suite_stats['deal_copy']} | insurance={suite_stats['insurance_intent']} | "
                  f"credit={suite_stats['credit_narrative']} | providers={len(providers)}\n")

    if args.dry_run:
        console.print("[green]✓ Dry run complete. Imports OK. Tools registered.[/green]")
        registry = build_tool_registry()
        for t in registry.list_tools():
            console.print(f"  [dim]{t['name']:20}[/dim] {t['description']}")
        sys.exit(0)

    budget = BudgetGuard(max_cost_usd=args.max_cost)
    agent  = EvalAgent(providers, test_cases, budget=budget)
    state  = agent.run(run_id)

    if state.aborted:
        console.print(f"[bold red]🚨 Agent ABORTED: {state.abort_reason}[/bold red]")
        logger.error(f"[{run_id}] Run aborted: {state.abort_reason}")
        sys.exit(1)

    detector_results = []

    for provider in providers:
        pname = provider.name
        responses_p  = state.responses.get(pname, [])
        scores_p     = state.scores.get(pname, [])

        print_summary_table(pname, responses_p, scores_p, test_cases)

        breakdown = compute_task_cost_breakdown(responses_p, test_cases)
        print_cost_breakdown(pname, breakdown)

        prompt_metadata = create_prompt_metadata(
            prompt=build_prompt_bundle(test_cases),
            provider=pname,
        )

        snapshot = build_snapshot(
            provider=pname, model=provider.model, run_id=run_id,
            scorer_results=scores_p, provider_responses=responses_p,
            test_cases=test_cases,
            prompt_hash=prompt_metadata["prompt_hash"],
            prompt_version=prompt_metadata.get("prompt_hash", ""),
            git_commit=prompt_metadata["git_commit"],
            git_branch=prompt_metadata["git_branch"],
            prompt_diff=prompt_metadata["prompt_diff"],
            task_prompt_hashes=prompt_metadata.get("task_prompt_hashes", {}),
            metadata={"prompt_metadata": prompt_metadata, "cost_by_task": breakdown},
        )

        det_result = detect_regressions(snapshot, update_baseline=args.update_baseline)
        append_run(snapshot, det_result)
        detector_results.append(det_result)

        verdict_color = {"GO": "green", "NO-GO": "red", "INCONCLUSIVE": "yellow"}.get(
            det_result.decision, "white"
        )
        console.print(Panel(det_result.summary, border_style=verdict_color))

    console.rule("[bold]Statistical Summary[/bold]")
    for dr in detector_results:
        if not dr.stat_report:
            continue
        for t in dr.stat_report.tests:
            color = {"improved": "green", "degraded": "red", "unchanged": "dim"}.get(t.direction, "white")
            console.print(
                f"  [{color}]{dr.provider:14}[/{color}] {t.metric:22} "
                f"Δ={t.delta:+.4f} ({t.delta_pct:+.1f}%)  "
                f"95%CI=[{t.ci_lower:.4f},{t.ci_upper:.4f}]  "
                f"p={t.p_value:.4f}  {'⚠ REGRESSED' if t.regressed else '✓ ok'}"
            )

    console.rule("[bold]Agent Phase Log[/bold]")
    for transition in state.phase_history:
        console.print(f"  [dim]{transition.timestamp}[/dim]  [cyan]{transition.phase.value:10}[/cyan]  {transition.notes}")
    if state.errors:
        console.print(f"\n  Typed errors during run ({len(state.errors)}):")
        for test_id, err_type, prov in state.errors[:10]:
            console.print(f"  [red]  {prov}/{test_id}: {err_type}[/red]")

    console.print(f"\n  [bold]Budget used:[/bold] ${state.total_cost_usd:.6f} / ${budget.max_cost_usd:.2f}  |  "
                  f"Tokens: {state.total_tokens:,} / {budget.max_tokens:,}")

    evaluation_meta = {
        "evaluation_config": {
            "providers": [p.name for p in providers],
            "max_cost_usd": args.max_cost,
            "statistical_significance_threshold": 0.05,
            "confidence_level": 0.95,
            "run_mode": (
                "baseline_update"
                if args.update_baseline
                else "standard"
            ),
        }
    }

    console.rule("[bold]Reports[/bold]")
    json_path = generate_json_report(
        run_id,
        detector_results,
        suite_stats,
        extra_meta=evaluation_meta,
    )
    md_path   = generate_markdown_report(run_id, detector_results, suite_stats)

    # Use relative paths for better terminal clickability
    rel_json = Path(json_path).relative_to(Path.cwd()) if Path(json_path).is_absolute() else json_path
    rel_md = Path(md_path).relative_to(Path.cwd()) if Path(md_path).is_absolute() else md_path

    console.print(f"  JSON: {rel_json}")
    console.print(f"  MD:   {rel_md}")
    console.print(f"  Log:  logs/eval.log")

    decisions = [dr.decision for dr in detector_results]
    overall = "NO-GO" if "NO-GO" in decisions else ("INCONCLUSIVE" if "INCONCLUSIVE" in decisions else "GO")
    console.rule()

    if overall == "GO":
        console.print("[bold green]✅ GO — All providers passed. Safe to deploy.[/bold green]")
        sys.exit(0)
    elif overall == "NO-GO":
        failed = [r.provider for r in detector_results if r.decision == "NO-GO"]
        console.print(f"[bold red]❌ NO-GO — Regressions in: {', '.join(failed)}. PR BLOCKED.[/bold red]")
        sys.exit(1)
    else:
        console.print("[bold yellow]⚠ INCONCLUSIVE — Insufficient data. Manual review required.[/bold yellow]")
        sys.exit(2)


if __name__ == "__main__":
    main()
