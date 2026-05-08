"""
Report generator for The Guard eval pipeline.

Produces two artifacts per run:
  1. reports/<run_id>.json  — machine-readable full report
  2. reports/<run_id>.md    — human-readable Markdown report
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from detector.detector import DetectorResult


REPORTS_DIR = Path(__file__).parent.parent / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


def _json_safe(value):
    try:
        import numpy as np
    except Exception:
        np = None

    if np is not None and isinstance(value, np.generic):
        return value.item()

    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]

    return value

def generate_json_report(
    run_id: str,
    detector_results: list[DetectorResult],
    suite_stats: dict,
    extra_meta: dict = {},
) -> Path:
    overall_decision = (
        "NO-GO"
        if any(r.decision == "NO-GO" for r in detector_results)
        else (
            "INCONCLUSIVE"
            if any(r.decision == "INCONCLUSIVE" for r in detector_results)
            else "GO"
        )
    )

    overall_pass = overall_decision == "GO"

    report = {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "suite": suite_stats,
        "overall_decision": overall_decision,
        "overall_pass": overall_pass,
        "providers": [],
        **extra_meta,
    }

    for dr in detector_results:
        stat_tests = []

        if dr.stat_report:
            for t in dr.stat_report.tests:
                stat_tests.append(
                    {
                        "metric": t.metric,
                        "baseline_mean": t.baseline_mean,
                        "current_mean": t.current_mean,
                        "delta": t.delta,
                        "delta_pct": t.delta_pct,
                        "test": t.test_name,
                        "p_value": t.p_value,
                        "significant": t.significant,
                        "regressed": t.regressed,
                        "direction": t.direction,
                        "ci_lower": getattr(t, "ci_lower", None),
                        "ci_upper": getattr(t, "ci_upper", None),
                        "affected_tests": getattr(t, "affected_tests", []),
                    }
                )

        report["providers"].append(
            {
                "provider": dr.provider,
                "model": dr.current_snapshot.model,
                "passed": dr.passed,
                "decision": dr.decision,
                "regressions": dr.regressions,
                "summary": dr.summary,
                "prompt_hash": dr.current_snapshot.prompt_hash,
                "task_prompt_hashes": dr.current_snapshot.task_prompt_hashes,
                "git_commit": dr.current_snapshot.git_commit,
                "git_branch": dr.current_snapshot.git_branch,
                "prompt_diff": dr.current_snapshot.prompt_diff,
                "prompt_diffs_by_task": dr.current_snapshot.prompt_diffs_by_task,
                "current": {
                    "pass_rate": dr.current_snapshot.pass_rate,
                    "mean_accuracy": dr.current_snapshot.mean_accuracy,
                    "mean_latency_ms": dr.current_snapshot.mean_latency_ms,
                    "p50_latency_ms": dr.current_snapshot.p50_latency_ms,
                    "p95_latency_ms": dr.current_snapshot.p95_latency_ms,
                    "latency_stddev_ms": dr.current_snapshot.latency_stddev_ms,
                    "total_cost_usd": dr.current_snapshot.total_cost_usd,
                    "n_tests": dr.current_snapshot.n_tests,
                    "task_scores": dr.current_snapshot.task_scores,
                    "task_passes": dr.current_snapshot.task_passes,
                    "task_cost_usd": dr.current_snapshot.task_cost_usd,
                    "task_latency_ms": dr.current_snapshot.task_latency_ms,
                    "testcase_results": dr.current_snapshot.testcase_results,
                    "metadata": dr.current_snapshot.metadata,
                },
                "baseline": {
                    "pass_rate": dr.baseline_snapshot.pass_rate if dr.baseline_snapshot else None,
                    "mean_accuracy": dr.baseline_snapshot.mean_accuracy if dr.baseline_snapshot else None,
                    "mean_latency_ms": dr.baseline_snapshot.mean_latency_ms if dr.baseline_snapshot else None,
                    "p95_latency_ms": dr.baseline_snapshot.p95_latency_ms if dr.baseline_snapshot else None,
                }
                if dr.baseline_snapshot
                else None,
                "stat_tests": stat_tests,
            }
        )

    path = REPORTS_DIR / f"{run_id}.json"

    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(report), f, indent=2)

    return path



def generate_markdown_report(
    run_id: str,
    detector_results: list[DetectorResult],
    suite_stats: dict,
) -> Path:
    overall_decision = (
        "NO-GO"
        if any(r.decision == "NO-GO" for r in detector_results)
        else (
            "INCONCLUSIVE"
            if any(r.decision == "INCONCLUSIVE" for r in detector_results)
            else "GO"
        )
    )

    overall_pass = overall_decision == "GO"

    status_badge = (
        "✅ PASSED"
        if overall_pass
        else "❌ FAILED — REGRESSIONS DETECTED"
    )

    decision = overall_decision

    lines = [
        f"# The Guard — Eval Report",
        f"",
        f"**Run ID:** `{run_id}`  ",
        f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  ",
        f"**Status:** {status_badge}",
        f"",
        f"## Deployment Decision",
        f"",
        f"### {decision}",
        f"",
        f"## Test Suite",
        f"",
        f"| Task type | Cases |",
        f"|-----------|-------|",
        f"| Deal Copy Generation | {suite_stats.get('deal_copy', '—')} |",
        f"| Insurance Intent Classification | {suite_stats.get('insurance_intent', '—')} |",
        f"| Credit Narrative Faithfulness | {suite_stats.get('credit_narrative', '—')} |",
        f"| Summarization Quality | {suite_stats.get('summarization', '—')} |",
        f"| Customer Query Classification | {suite_stats.get('classification', '—')} |",
        f"| Structured Data Extraction | {suite_stats.get('extraction', '—')} |",
        f"| **Total** | **{suite_stats.get('total', '—')}** |",
        f"",
        f"## Provider Results",
        f"",
    ]

    for dr in detector_results:
        s = dr.current_snapshot

        icon = "✅" if dr.passed else "❌"

        lines += [
            f"### {icon} {dr.provider.upper()} (`{s.model}`)",
            f"",
            f"> {dr.summary}",
            f"",
            f"| Metric | Current | 95% CI | Baseline | Delta | p-value | Status |",
            f"|--------|---------|---------|----------|-------|---------|--------|",
        ]

        if dr.stat_report:
            for t in dr.stat_report.tests:
                status = (
                    "🔴 REGRESSED"
                    if t.regressed
                    else (
                        "🟡 significant"
                        if t.significant
                        else "🟢 ok"
                    )
                )

                b_val = (
                    f"{t.baseline_mean:.4f}"
                    if dr.baseline_snapshot
                    else "—"
                )

                ci = (
                    f"[{getattr(t, 'ci_lower', 0.0):.4f}, "
                    f"{getattr(t, 'ci_upper', 1.0):.4f}]"
                )

                lines.append(
                    f"| {t.metric} | "
                    f"{t.current_mean:.4f} | "
                    f"{ci} | "
                    f"{b_val} | "
                    f"{t.delta:+.4f} ({t.delta_pct:+.1f}%) | "
                    f"{t.p_value:.4f} | "
                    f"{status} |"
                )

        else:
            lines.append(
                f"| pass_rate | {s.pass_rate:.2%} | — | — | — | — | 🟢 first run |"
            )

        lines += [
            f"",
            f"**Pass rate:** {s.pass_rate:.1%} &nbsp;|&nbsp; "
            f"**Avg latency:** {s.mean_latency_ms:.0f} ms &nbsp;|&nbsp; "
            f"**Total cost:** ${s.total_cost_usd:.6f}",
            f"",
        ]

        language_scores = s.metadata.get("language_scores", {})

        if language_scores:
            lines += [
                f"#### Language Breakdown",
                f"",
                f"| Language | Avg Score |",
                f"|----------|------------|",
            ]

            for lang, scores in language_scores.items():
                avg = sum(scores) / len(scores)
                lines.append(f"| {lang} | {avg:.4f} |")

            lines += [""]

        if not dr.passed:
            lines += [
                f"## NO-GO Regression Details",
                f"",
                f"### Prompt Version",
                f"",
                f"- Prompt hash: `{dr.current_snapshot.prompt_hash}`",
                f"- Git commit: `{dr.current_snapshot.git_commit}`",
                f"- Git branch: `{dr.current_snapshot.git_branch}`",
                f"",
            ]

            if dr.current_snapshot.task_prompt_hashes:
                lines += [
                    f"### Task Prompt Hashes",
                    f"",
                ]
                for task_type, prompt_hash in sorted(dr.current_snapshot.task_prompt_hashes.items()):
                    lines.append(f"- `{task_type}`: `{prompt_hash}`")
                lines += [""]

            prompt_diffs_by_task = dr.current_snapshot.prompt_diffs_by_task
            prompt_diff = dr.current_snapshot.prompt_diff

            if prompt_diffs_by_task:
                lines += [
                    f"### Prompt Diffs By Regressed Task",
                    f"",
                ]
                for task_type, diff_text in sorted(prompt_diffs_by_task.items()):
                    lines += [
                        f"#### {task_type}",
                        f"",
                        f"```diff",
                        diff_text[:1200],
                        f"```",
                        f"",
                    ]
            elif prompt_diff:
                lines += [
                    f"### Prompt Diff",
                    f"",
                    f"```diff",
                    prompt_diff[:1000],
                    f"```",
                    f"",
                ]

            lines += [
                f"### Regression Summary",
                f"",
            ]

            for regression in dr.regressions:
                severity = (
                    "HIGH"
                    if "10" in regression
                    else "MEDIUM"
                )
                lines.append(f"- [{severity}] {regression}")

            lines += [""]

            affected_cases = []

            if dr.stat_report:
                for t in dr.stat_report.tests:
                    if t.regressed and hasattr(t, "affected_tests"):
                        affected_cases.extend(t.affected_tests)

            if affected_cases:
                lines += [
                    f"### Affected Test Cases",
                    f"",
                ]

                for tc in sorted(set(affected_cases)):
                    lines.append(f"- `{tc}`")

                lines += [""]

            if "telugu" in language_scores:
                telugu_avg = (
                    sum(language_scores["telugu"])
                    / len(language_scores["telugu"])
                )

                if telugu_avg < 0.75:
                    lines += [
                        f"### Telugu Localization Warning",
                        f"",
                        f"Average Telugu quality dropped to `{telugu_avg:.4f}`.",
                        f"",
                    ]

        if dr.stat_report:
            lines += [
                f"<details><summary>Statistical test details</summary>",
                f"",
                f"| Metric | Test | Statistic | p-value | Significant | Regressed |",
                f"|--------|------|-----------|---------|-------------|-----------|",
            ]

            for t in dr.stat_report.tests:
                lines.append(
                    f"| {t.metric} | {t.test_name} | "
                    f"{t.statistic:.4f} | "
                    f"{t.p_value:.4f} | "
                    f"{'yes' if t.significant else 'no'} | "
                    f"{'**YES**' if t.regressed else 'no'} |"
                )

            lines += [
                f"",
                f"</details>",
                f"",
            ]

    total_regressions = sum(
        len(dr.regressions)
        for dr in detector_results
    )

    lines += [
        f"## Statistical Summary",
        f"",
        f"- Total providers evaluated: {len(detector_results)}",
        f"- Total regressions detected: {total_regressions}",
        f"- Statistical confidence threshold: 95%",
        f"- Regression significance threshold: p < 0.05",
        f"",
        f"---",
        f"",
        f"*Generated by The Guard eval pipeline.*",
    ]

    path = REPORTS_DIR / f"{run_id}.md"

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

        return path
