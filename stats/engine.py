"""
Statistical engine for The Guard eval pipeline.

Compares current eval run against a saved baseline and flags regressions
using statistically rigorous methods:

  - Accuracy / score distributions → Paired bootstrap + p-value
  - Pass/fail classification     → McNemar's test
  - Latency / cost               → Welch's t-test

A regression is flagged when:
  1. The metric degrades beyond the configured threshold, AND
  2. The degradation is statistically significant (p < ALPHA)

This prevents noisy single-run results from triggering false alarms.
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from scipy import stats


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
ALPHA              = 0.05   # significance level
MIN_EFFECT_SIZE    = 0.03   # minimum meaningful delta (3 percentage points)
MIN_SAMPLE_SIZE    = 5      # need at least N samples to run tests
BOOTSTRAP_SAMPLES  = 2000


# ─────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────
@dataclass
class MetricSummary:
    name: str
    mean: float
    std: float
    median: float
    min: float
    max: float
    n: int


@dataclass
class StatTestResult:
    metric: str
    baseline_mean: float
    current_mean:  float
    delta:         float          # current - baseline (negative = regression)
    delta_pct:     float          # delta as % of baseline
    test_name:     str
    statistic:     float
    p_value:       float
    ci_lower:      float
    ci_upper:      float
    significant:   bool           # p < ALPHA
    regressed:     bool           # significant AND delta < -MIN_EFFECT_SIZE
    direction:     str            # "improved", "degraded", "unchanged"


@dataclass
class StatisticalReport:
    provider: str
    tests: list[StatTestResult] = field(default_factory=list)
    regressions: list[str]      = field(default_factory=list)

    @property
    def has_regressions(self) -> bool:
        return len(self.regressions) > 0


# ─────────────────────────────────────────────
# Core stats helpers
# ─────────────────────────────────────────────
def summarise(values: list[float], name: str) -> MetricSummary:
    arr = np.array(values)
    return MetricSummary(
        name=name,
        mean=float(np.mean(arr)),
        std=float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        median=float(np.median(arr)),
        min=float(np.min(arr)),
        max=float(np.max(arr)),
        n=len(arr),
    )


def _paired_bootstrap(
    baseline: list[float],
    current: list[float],
    n_resamples: int = BOOTSTRAP_SAMPLES,
    alpha: float = ALPHA,
) -> tuple[float, float, float, float]:
    b = np.array(baseline)
    c = np.array(current)
    if len(b) != len(c) or len(b) < MIN_SAMPLE_SIZE:
        return 0.0, 0.0, 0.0, 1.0

    diff = c - b
    rng = np.random.default_rng(12345)
    boot_diffs = np.empty(n_resamples, dtype=float)
    n = len(diff)
    for i in range(n_resamples):
        sample = diff[rng.integers(0, n, n)]
        boot_diffs[i] = float(np.mean(sample))

    delta = float(np.mean(diff))
    lower = float(np.quantile(boot_diffs, alpha / 2))
    upper = float(np.quantile(boot_diffs, 1 - alpha / 2))
    p_value = float(np.mean(np.abs(boot_diffs) >= abs(delta)))
    return delta, lower, upper, p_value


def bootstrap_confidence_interval(values, n_resamples=2000):
    arr = np.array(values)

    if len(arr) == 0:
        return (0.0, 0.0)

    rng = np.random.default_rng(42)
    means = []

    for _ in range(n_resamples):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means.append(np.mean(sample))

    return (
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    )


def mcnemar_test(
    baseline_pass: list[bool],
    current_pass: list[bool],
    metric: str,
) -> StatTestResult:
    b = np.array(baseline_pass, dtype=bool)
    c = np.array(current_pass, dtype=bool)

    if len(b) != len(c) or len(b) < MIN_SAMPLE_SIZE:
        return _insufficient_data_result(metric, b.astype(float), c.astype(float), "McNemar's test")

    b_pass_c_fail = int(np.sum(b & ~c))
    b_fail_c_pass = int(np.sum(~b & c))
    n = b_pass_c_fail + b_fail_c_pass

    if n == 0:
        p_value = 1.0
        statistic = 0.0
    else:
        statistic = ((abs(b_pass_c_fail - b_fail_c_pass) - 1) ** 2) / n
        p_value = float(stats.chi2.sf(statistic, df=1))

    baseline_mean = float(np.mean(b.astype(float)))
    current_mean = float(np.mean(c.astype(float)))
    delta = current_mean - baseline_mean
    delta_pct = (delta / baseline_mean * 100) if baseline_mean != 0 else 0.0
    significant = p_value < ALPHA
    regressed = significant and delta < -MIN_EFFECT_SIZE
    direction = _direction(delta)

    return StatTestResult(
        metric=metric,
        baseline_mean=round(baseline_mean, 4),
        current_mean=round(current_mean, 4),
        delta=round(delta, 4),
        delta_pct=round(delta_pct, 2),
        test_name="McNemar's test",
        statistic=round(statistic, 4),
        p_value=round(p_value, 4),
        ci_lower=0.0,
        ci_upper=0.0,
        significant=significant,
        regressed=regressed,
        direction=direction,
    )


def paired_bootstrap_test(
    baseline: list[float],
    current: list[float],
    metric: str,
    higher_is_better: bool = True,
) -> StatTestResult:
    b = np.array(baseline)
    c = np.array(current)

    if len(b) < MIN_SAMPLE_SIZE or len(c) < MIN_SAMPLE_SIZE or len(b) != len(c):
        return _insufficient_data_result(metric, b, c, "Paired bootstrap")

    delta, lower, upper, p_value = _paired_bootstrap(b.tolist(), c.tolist())
    baseline_mean = float(np.mean(b))
    current_mean = float(np.mean(c))
    delta_pct = (delta / baseline_mean * 100) if baseline_mean != 0 else 0.0
    significant = p_value < ALPHA
    regressed = significant and delta < -MIN_EFFECT_SIZE
    direction = _direction(delta, higher_is_better)

    return StatTestResult(
        metric=metric,
        baseline_mean=round(baseline_mean, 4),
        current_mean=round(current_mean, 4),
        delta=round(delta, 4),
        delta_pct=round(delta_pct, 2),
        test_name="Paired bootstrap",
        statistic=0.0,
        p_value=round(p_value, 4),
        ci_lower=round(lower, 4),
        ci_upper=round(upper, 4),
        significant=significant,
        regressed=regressed,
        direction=direction,
    )


def welch_t_test(
    baseline: list[float],
    current: list[float],
    metric: str,
    higher_is_better: bool = True,
) -> StatTestResult:
    """
    Welch's t-test — for continuous metrics like latency and cost.
    Two-tailed: we care if latency goes up OR cost changes significantly.
    For latency/cost, regression = significantly higher.
    """
    b = np.array(baseline)
    c = np.array(current)

    if len(b) < MIN_SAMPLE_SIZE or len(c) < MIN_SAMPLE_SIZE:
        return _insufficient_data_result(metric, b, c, "Welch's t-test")

    stat, p = stats.ttest_ind(c, b, equal_var=False, alternative="two-sided")

    baseline_mean = float(np.mean(b))
    current_mean  = float(np.mean(c))
    delta         = current_mean - baseline_mean
    delta_pct     = (delta / baseline_mean * 100) if baseline_mean != 0 else 0.0
    significant   = p < ALPHA

    if higher_is_better:
        regressed = significant and delta < -MIN_EFFECT_SIZE
    else:
        regressed = significant and delta_pct > (MIN_EFFECT_SIZE * 100)

    direction = _direction(delta, higher_is_better)

    return StatTestResult(
        metric=metric,
        baseline_mean=round(baseline_mean, 4),
        current_mean=round(current_mean, 4),
        delta=round(delta, 4),
        delta_pct=round(delta_pct, 2),
        test_name="Welch's t-test",
        statistic=round(float(stat), 4),
        p_value=round(float(p), 4),
        ci_lower=0.0,
        ci_upper=0.0,
        significant=significant,
        regressed=regressed,
        direction=direction,
    )


def _direction(delta: float, higher_is_better: bool = True) -> str:
    if abs(delta) < 1e-6:
        return "unchanged"
    improved = delta > 0 if higher_is_better else delta < 0
    return "improved" if improved else "degraded"


def _insufficient_data_result(
    metric: str,
    baseline: np.ndarray,
    current: np.ndarray,
    test_name: str,
) -> StatTestResult:
    bm = float(np.mean(baseline)) if len(baseline) > 0 else 0.0
    cm = float(np.mean(current))  if len(current)  > 0 else 0.0
    return StatTestResult(
        metric=metric,
        baseline_mean=round(bm, 4),
        current_mean=round(cm, 4),
        delta=round(cm - bm, 4),
        delta_pct=0.0,
        test_name=test_name,
        statistic=0.0,
        p_value=1.0,
        ci_lower=0.0,
        ci_upper=0.0,
        significant=False,
        regressed=False,
        direction="unchanged",
    )


# ─────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────
def compare_to_baseline(
    provider: str,
    baseline_scores: dict,
    current_scores: dict,
    baseline_pass: dict | None = None,
    current_pass: dict | None = None,
) -> StatisticalReport:
    """
    Run all statistical tests and return a StatisticalReport.
    """
    report = StatisticalReport(provider=provider)

    for metric, baseline_values in baseline_scores.items():
        if metric not in current_scores:
            continue

        current_values = current_scores[metric]
        lower_is_better = metric.endswith("latency_ms") or metric.endswith("cost_usd")
        if metric.startswith("accuracy") or metric.endswith("score") or metric.endswith("pass_rate"):
            result = paired_bootstrap_test(
                baseline_values,
                current_values,
                metric=metric,
                higher_is_better=not lower_is_better,
            )
        else:
            result = welch_t_test(
                baseline_values,
                current_values,
                metric=metric,
                higher_is_better=not lower_is_better,
            )

        report.tests.append(result)
        if result.regressed:
            report.regressions.append(metric)

    if baseline_pass and current_pass:
        for metric, baseline_pass_values in baseline_pass.items():
            if metric not in current_pass:
                continue
            result = mcnemar_test(
                baseline_pass_values,
                current_pass[metric],
                metric=metric,
            )
            report.tests.append(result)
            if result.regressed:
                report.regressions.append(metric)

    return report
