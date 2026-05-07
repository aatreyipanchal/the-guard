"""
Regression detector for The Guard eval pipeline.

Responsibilities:
  1. Load a saved baseline from disk (baselines/<provider>.json)
  2. Compare current run against baseline using the statistical engine
  3. Save the current run as the new baseline (on --update-baseline flag)
  4. Return a clear pass/fail decision with human-readable reasons
"""

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from stats.engine import compare_to_baseline, StatisticalReport


BASELINES_DIR = Path(__file__).parent.parent / "baselines"
BASELINES_DIR.mkdir(exist_ok=True)


@dataclass
class RunSnapshot:
    """A snapshot of one eval run for a single provider."""
    provider: str
    model: str
    run_id: str
    timestamp: str
    accuracy: list[float]          # per-test numerical scores
    latency_ms: list[float]        # per-test latencies
    cost_usd: list[float]          # per-test costs
    passes: list[bool]             # per-test pass/fail values
    pass_rate: float
    mean_accuracy: float
    mean_latency_ms: float
    total_cost_usd: float
    n_tests: int
    task_scores: dict[str, list[float]] = field(default_factory=dict)
    task_passes: dict[str, list[bool]] = field(default_factory=dict)
    task_cost_usd: dict[str, float] = field(default_factory=dict)
    task_latency_ms: dict[str, float] = field(default_factory=dict)
    task_counts: dict[str, int] = field(default_factory=dict)
    prompt_hash: str = ""
    prompt_version: str = ""
    git_commit: str = ""
    git_branch: str = ""
    prompt_diff: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class DetectorResult:
    provider: str
    run_id: str
    has_baseline: bool
    passed: bool
    decision: str
    regressions: list[str]
    stat_report: Optional[StatisticalReport]
    current_snapshot: RunSnapshot
    baseline_snapshot: Optional[RunSnapshot]
    summary: str


def _baseline_path(provider: str) -> Path:
    return BASELINES_DIR / f"{provider}.json"


def load_baseline(provider: str) -> Optional[RunSnapshot]:
    path = _baseline_path(provider)
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)

    # Backward compatibility for older baselines.
    data.setdefault("passes", [])
    data.setdefault("task_scores", {})
    data.setdefault("task_passes", {})
    data.setdefault("task_cost_usd", {})
    data.setdefault("task_latency_ms", {})
    data.setdefault("task_counts", {})
    data.setdefault("prompt_hash", "")
    data.setdefault("prompt_version", "")
    data.setdefault("git_commit", "")
    data.setdefault("git_branch", "")
    data.setdefault("prompt_diff", "")
    data.setdefault("metadata", {})

    return RunSnapshot(**data)


def save_baseline(snapshot: RunSnapshot) -> None:
    path = _baseline_path(snapshot.provider)
    with open(path, "w") as f:
        json.dump(asdict(snapshot), f, indent=2)


def detect_regressions(
    current: RunSnapshot,
    update_baseline: bool = False,
) -> DetectorResult:
    """
    Compare current run to saved baseline.
    If no baseline exists, save current as baseline and pass.
    If update_baseline=True, save current as new baseline after comparison.
    """
    baseline = load_baseline(current.provider)

    if baseline is None:
        save_baseline(current)
        return DetectorResult(
            provider=current.provider,
            run_id=current.run_id,
            has_baseline=False,
            passed=True,
            decision="INCONCLUSIVE",
            regressions=[],
            stat_report=None,
            current_snapshot=current,
            baseline_snapshot=None,
            summary=(
                f"No baseline found for {current.provider}. "
                f"Current run saved as baseline. INCONCLUSIVE until next run."
            ),
        )

    baseline_scores = {
        "accuracy": baseline.accuracy,
        "latency_ms": baseline.latency_ms,
        "cost_usd": baseline.cost_usd,
    }
    current_scores = {
        "accuracy": current.accuracy,
        "latency_ms": current.latency_ms,
        "cost_usd": current.cost_usd,
    }

    baseline_pass = {
        "overall_pass": baseline.passes,
    }
    current_pass = {
        "overall_pass": current.passes,
    }

    for task, values in baseline.task_scores.items():
        baseline_scores[f"{task}_accuracy"] = values
    for task, values in current.task_scores.items():
        current_scores[f"{task}_accuracy"] = values

    for task, values in baseline.task_passes.items():
        baseline_pass[f"{task}_pass"] = values
    for task, values in current.task_passes.items():
        current_pass[f"{task}_pass"] = values

    stat_report = compare_to_baseline(
        provider=current.provider,
        baseline_scores=baseline_scores,
        current_scores=current_scores,
        baseline_pass=baseline_pass,
        current_pass=current_pass,
    )

    passed = not stat_report.has_regressions
    decision = "GO" if passed else "NO-GO"
    regressions = []
    for test in stat_report.tests:
        if test.regressed:
            regressions.append(
                (
                    f"{test.metric} regressed by {abs(test.delta_pct):.2f}% "
                    f"(p={test.p_value:.3f}, 95% CI=[{test.ci_lower:.4f}, {test.ci_upper:.4f}])"
                )
            )

    if passed:
        summary = (
            f"{current.provider}: {decision}. "
            f"No regressions detected. Pass rate {current.pass_rate:.1%} "
            f"(baseline: {baseline.pass_rate:.1%}). ✓"
        )
    else:
        reg_str = ", ".join(regressions)
        summary = (
            f"{current.provider}: {decision} — Regression detected. "
            f"Pass rate {current.pass_rate:.1%} "
            f"(baseline: {baseline.pass_rate:.1%}). \n"
            f"Details: {reg_str}"
        )

    if update_baseline:
        save_baseline(current)

    return DetectorResult(
        provider=current.provider,
        run_id=current.run_id,
        has_baseline=True,
        passed=passed,
        decision=decision,
        regressions=regressions,
        stat_report=stat_report,
        current_snapshot=current,
        baseline_snapshot=baseline,
        summary=summary,
    )


def build_snapshot(
    provider: str,
    model: str,
    run_id: str,
    scorer_results: list,       # list[ScorerResult]
    provider_responses: list,   # list[ProviderResponse]
    test_cases: list,
    prompt_hash: str = "",
    prompt_version: str = "",
    git_commit: str = "",
    git_branch: str = "",
    prompt_diff: str = "",
    metadata: dict | None = None,
) -> RunSnapshot:
    """Build a RunSnapshot from raw eval results."""
    tc_map = {tc.id: tc for tc in test_cases}
    accuracy = [r.score for r in scorer_results]
    latency_ms = [resp.latency_ms for resp in provider_responses]
    cost_usd = [resp.cost_usd for resp in provider_responses]
    passes = [r.passed for r in scorer_results]

    task_scores: dict[str, list[float]] = {}
    task_passes: dict[str, list[bool]] = {}
    task_cost_usd: dict[str, float] = {}
    task_latency: dict[str, list[float]] = {}
    task_counts: dict[str, int] = {}
    language_scores: dict[str, list[float]] = {}

    for r, resp in zip(scorer_results, provider_responses):
        tc = tc_map.get(r.test_id)
        task_type = tc.task_type if tc else "unknown"
        language = tc.metadata.get("language", "english") if tc else "english"
        task_scores.setdefault(task_type, []).append(r.score)
        task_passes.setdefault(task_type, []).append(r.passed)
        task_cost_usd[task_type] = task_cost_usd.get(task_type, 0.0) + resp.cost_usd
        task_latency.setdefault(task_type, []).append(resp.latency_ms)
        task_counts[task_type] = task_counts.get(task_type, 0) + 1
        language_scores.setdefault(language, []).append(r.score)

    task_latency_ms = {
        task: float(np.mean(values)) if values else 0.0
        for task, values in task_latency.items()
    }

    return RunSnapshot(
        provider=provider,
        model=model,
        run_id=run_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        accuracy=accuracy,
        latency_ms=latency_ms,
        cost_usd=cost_usd,
        passes=passes,
        pass_rate=sum(passes) / len(passes) if passes else 0.0,
        mean_accuracy=sum(accuracy) / len(accuracy) if accuracy else 0.0,
        mean_latency_ms=sum(latency_ms) / len(latency_ms) if latency_ms else 0.0,
        total_cost_usd=sum(cost_usd),
        n_tests=len(scorer_results),
        task_scores=task_scores,
        task_passes=task_passes,
        task_cost_usd=task_cost_usd,
        task_latency_ms=task_latency_ms,
        task_counts=task_counts,
        prompt_hash=prompt_hash,
        prompt_version=prompt_version,
        git_commit=git_commit,
        git_branch=git_branch,
        prompt_diff=prompt_diff,
        metadata={
            **(metadata or {}),
            "language_scores": language_scores,
        },
    )
