"""Scoring functions for The Guard eval pipeline."""

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


SEMANTIC_THRESHOLD = 0.75
FORMAT_PASS_THRESH = 0.75

_CREDIT_META_KEYS = frozenset({
    "expected_claims",
    "adversarial",
    "expected_failure_modes",
    "zero_transactions",
    "negative_gmv",
})

_INSURANCE_LABELS = frozenset({
    "device_protection",
    "travel_insurance",
    "health_micro",
    "accidental_damage",
    "no_insurance",
})

_embed_model: Optional[SentenceTransformer] = None


@dataclass
class ScorerResult:
    test_id: str
    provider: str
    scoring_method: str
    score: float
    passed: bool
    expected: Any
    actual: str
    confidence: float = 1.0
    ci_95: tuple = (0.0, 1.0)
    language: str = "english"
    task_type: str = "general"
    details: dict = field(default_factory=dict)


def _score_ci(score: float, width: float = 0.03) -> tuple:
    return (
        max(0.0, score - width),
        min(1.0, score + width),
    )


def _get_embed_model() -> SentenceTransformer:
    global _embed_model

    if _embed_model is None:
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    return _embed_model


def _safe_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def score_exact(
    test_id: str,
    provider: str,
    actual: str,
    expected: str,
) -> ScorerResult:

    norm_a = actual.strip().lower().rstrip(".")
    norm_e = expected.strip().lower().rstrip(".")

    passed = norm_a == norm_e
    s = 1.0 if passed else 0.0

    return ScorerResult(
        test_id=test_id,
        provider=provider,
        scoring_method="exact",
        score=s,
        passed=passed,
        expected=expected,
        actual=actual,
        confidence=1.0 if passed else 0.5,
        ci_95=(s, s),
        details={
            "normalized_actual": norm_a,
            "normalized_expected": norm_e,
        },
    )


def score_semantic(
    test_id: str,
    provider: str,
    actual: str,
    expected: str,
) -> ScorerResult:

    model = _get_embed_model()

    emb = model.encode([actual, expected])

    sim = float(
        cosine_similarity([emb[0]], [emb[1]])[0][0]
    )

    sim = max(0.0, min(1.0, sim))

    passed = sim >= SEMANTIC_THRESHOLD

    return ScorerResult(
        test_id=test_id,
        provider=provider,
        scoring_method="semantic",
        score=round(sim, 4),
        passed=passed,
        expected=expected,
        actual=actual,
        confidence=0.95,
        ci_95=_score_ci(sim),
        details={
            "cosine_similarity": round(sim, 4),
            "threshold": SEMANTIC_THRESHOLD,
        },
    )


def _extract_json(text: str) -> Optional[dict]:
    text = text.strip()

    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)

    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)

        if match:
            try:
                return json.loads(match.group())

            except json.JSONDecodeError:
                pass

    return None


def score_json_match(
    test_id: str,
    provider: str,
    actual: str,
    expected: dict,
) -> ScorerResult:

    parsed = _extract_json(actual)

    if parsed is None:
        return ScorerResult(
            test_id=test_id,
            provider=provider,
            scoring_method="json_match",
            score=0.0,
            passed=False,
            expected=expected,
            actual=actual,
            confidence=0.5,
            ci_95=(0.0, 0.0),
            details={
                "error": "Could not parse JSON",
                "raw": actual,
            },
        )

    fields = {}
    matched = 0

    for key, exp_val in expected.items():

        act_val = parsed.get(key, "")

        match = (
            str(act_val).strip().lower()
            ==
            str(exp_val).strip().lower()
        )

        fields[key] = {
            "expected": exp_val,
            "actual": act_val,
            "match": match,
        }

        if match:
            matched += 1

    total = len(expected)

    s = round(
        matched / total if total > 0 else 0.0,
        4,
    )

    return ScorerResult(
        test_id=test_id,
        provider=provider,
        scoring_method="json_match",
        score=s,
        passed=(s == 1.0),
        expected=expected,
        actual=actual,
        confidence=0.95,
        ci_95=_score_ci(s),
        details={
            "fields": fields,
            "matched_fields": matched,
            "total_fields": total,
        },
    )


def _phrase_hit(actual: str, phrase: str) -> bool:
    return phrase.strip().lower() in actual.lower()


def score_format_compliance(
    test_id: str,
    provider: str,
    actual: str,
    expected: dict,
) -> ScorerResult:

    actual_len = len(actual)
    details = {}

    # Character limits
    if "max_chars" in expected:

        max_chars = int(expected["max_chars"])

        length_score = (
            1.0
            if actual_len <= max_chars
            else max(
                0.0,
                1.0 - (actual_len - max_chars) / max_chars
            )
        )

        details["char_limit"] = {
            "max": max_chars,
            "actual": actual_len,
            "ok": actual_len <= max_chars,
        }

    elif (
        "title_max_chars" in expected
        and
        "body_max_chars" in expected
    ):

        parts = actual.split("|", 1)

        title = parts[0].strip() if len(parts) > 0 else actual
        body = parts[1].strip() if len(parts) > 1 else ""

        t_ok = len(title) <= expected["title_max_chars"]
        b_ok = len(body) <= expected["body_max_chars"]

        length_score = (
            (0.5 if t_ok else 0.0)
            +
            (0.5 if b_ok else 0.0)
        )

        details["char_limit"] = {
            "title_max": expected["title_max_chars"],
            "title_len": len(title),
            "title_ok": t_ok,
            "body_max": expected["body_max_chars"],
            "body_len": len(body),
            "body_ok": b_ok,
        }

    elif "max_length" in expected:

        max_chars = int(expected["max_length"])

        length_score = (
            1.0
            if actual_len <= max_chars
            else max(
                0.0,
                1.0 - (actual_len - max_chars) / max_chars
            )
        )

    else:
        length_score = 1.0

    coupon_any = _safe_list(
        expected.get("coupon_any")
    )

    required_phrases = _safe_list(
        expected.get("required_phrases")
    )

    coupon_ok = True
    coupon_used = None

    if coupon_any:

        coupon_ok = any(
            _phrase_hit(actual, c)
            for c in coupon_any
        )

        coupon_used = next(
            (
                c for c in coupon_any
                if _phrase_hit(actual, c)
            ),
            None,
        )

    details["coupon"] = {
        "options": coupon_any,
        "found": coupon_used,
        "ok": coupon_ok,
    }

    phrase_hits = [
        p for p in required_phrases
        if _phrase_hit(actual, p)
    ]

    phrase_score = (
        len(phrase_hits) / len(required_phrases)
        if required_phrases
        else 1.0
    )

    expired_penalty = 0.0

    if (
        expected.get("expired_offer")
        and
        "expired" in actual.lower()
    ):
        expired_penalty = 0.25

    coupon_weight = 0.20 if coupon_any else 0.0
    length_weight = 0.45 + (0.20 if not coupon_any else 0.0)
    phrase_weight = 0.35

    s = round(
        (
            length_score * length_weight
            +
            (1.0 if coupon_ok else 0.0) * coupon_weight
            +
            phrase_score * phrase_weight
            -
            expired_penalty
        ),
        4,
    )

    s = max(0.0, min(1.0, s))

    passed = s >= FORMAT_PASS_THRESH

    return ScorerResult(
        test_id=test_id,
        provider=provider,
        scoring_method="format_compliance",
        score=s,
        passed=passed,
        expected=expected,
        actual=actual,
        confidence=0.95,
        ci_95=_score_ci(s),
        details=details,
    )


def _normalize_label(text: str) -> str:

    norm = re.sub(
        r"[^a-z0-9_]+",
        "_",
        text.strip().lower(),
    ).strip("_")

    legacy = {
        "auto_insurance": "accidental_damage",
        "health_insurance": "health_micro",
        "life_insurance": "no_insurance",
        "micro_insurance": "health_micro",
    }

    return legacy.get(norm, norm)


def score_intent_match(
    test_id: str,
    provider: str,
    actual: str,
    expected: str,
) -> ScorerResult:

    raw = (
        actual.splitlines()[0]
        .split(",")[0]
        .split(".")[0]
    )

    norm_actual = _normalize_label(raw)
    norm_expected = _normalize_label(str(expected))

    passed = norm_actual == norm_expected

    s = 1.0 if passed else 0.0

    return ScorerResult(
        test_id=test_id,
        provider=provider,
        scoring_method="intent_match",
        score=s,
        passed=passed,
        expected=expected,
        actual=actual,
        confidence=1.0 if passed else 0.5,
        ci_95=(s, s),
        details={
            "normalized_actual": norm_actual,
            "normalized_expected": norm_expected,
            "valid_labels": sorted(_INSURANCE_LABELS),
        },
    )


def score_factual_grounding(
    test_id: str,
    provider: str,
    actual: str,
    expected: dict,
) -> ScorerResult:

    actual_lower = actual.lower()

    field_results = {}

    matched = 0
    total = 0

    expected = expected or {}

    for key, value in expected.items():

        if key in _CREDIT_META_KEYS:
            continue

        if isinstance(value, bool):
            continue

        if isinstance(value, list):
            continue

        if value is None:
            continue

        total += 1

        target = str(value).strip().lower()

        if isinstance(value, (int, float)):

            candidates = [
                str(value),
                str(int(value))
                if float(value) == int(value)
                else None,
                f"{value}%",
                f"{int(value)}%",
            ]

            match = any(
                c and c.lower() in actual_lower
                for c in candidates
                if c
            )

        else:
            match = target in actual_lower

        field_results[key] = {
            "expected": value,
            "match": match,
        }

        if match:
            matched += 1

    if (
        expected.get("zero_transactions")
        and
        re.search(r"\b[1-9]\d{2,}\b", actual)
    ):

        matched = max(0, matched - 1)

        field_results["zero_transactions_hallucination"] = {
            "expected": "no large numbers",
            "match": False,
        }

    s = round(
        matched / total if total > 0 else 0.0,
        4,
    )

    passed = s >= 0.80

    return ScorerResult(
        test_id=test_id,
        provider=provider,
        scoring_method="factual_grounding",
        score=s,
        passed=passed,
        expected=expected,
        actual=actual,
        confidence=0.95,
        ci_95=_score_ci(s),
        details={
            "field_results": field_results,
            "matched": matched,
            "total": total,
        },
    )


_LLM_JUDGE_PROMPT = """
You are a senior GrabOn marketing reviewer.

Score this deal copy on:
1. persuasion
2. clarity
3. factuality
4. tone

Return ONLY valid JSON.
"""


def _call_judge_api(actual: str, reference: str):

    api_key = os.getenv("ANTHROPIC_API_KEY")

    if not api_key:
        return None

    try:
        from anthropic import Anthropic

        client = Anthropic(api_key=api_key)

        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=128,
            temperature=0,
            system=_LLM_JUDGE_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content":
                        f"Reference: {reference}\n\n"
                        f"Generated: {actual}",
                }
            ],
        )

        raw = resp.content[0].text.strip()

        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        return json.loads(raw)

    except Exception:
        return None


def score_llm_judge(
    test_id: str,
    provider: str,
    actual: str,
    expected: Any,
) -> ScorerResult:

    if (
        isinstance(expected, dict)
        and
        all(
            k in expected
            for k in (
                "persuasion",
                "clarity",
                "factuality",
                "tone",
            )
        )
    ):
        rubric = expected

    else:

        reference = (
            str(expected)
            if not isinstance(expected, dict)
            else json.dumps(expected)
        )

        rubric = _call_judge_api(actual, reference)

        if rubric is None:

            sem = score_semantic(
                test_id,
                provider,
                actual,
                str(expected),
            )

            return ScorerResult(
                test_id=test_id,
                provider=provider,
                scoring_method="llm_judge",
                score=sem.score,
                passed=sem.score >= 0.70,
                expected=expected,
                actual=actual,
                confidence=0.75,
                ci_95=_score_ci(sem.score, 0.08),
                details={
                    "fallback": "semantic",
                    "semantic_score": sem.score,
                },
            )

    persuasion = float(rubric.get("persuasion", 0.0))
    clarity = float(rubric.get("clarity", 0.0))
    factuality = float(rubric.get("factuality", 0.0))
    tone = float(rubric.get("tone", 0.0))

    final = round(
        persuasion * 0.30
        + clarity * 0.25
        + factuality * 0.30
        + tone * 0.15,
        4,
    )

    return ScorerResult(
        test_id=test_id,
        provider=provider,
        scoring_method="llm_judge",
        score=final,
        passed=final >= 0.70,
        expected=expected,
        actual=actual,
        confidence=0.85,
        ci_95=_score_ci(final, 0.06),
        details={
            "rubric_scores": rubric,
        },
    )


def score(
    test_id: str,
    provider: str,
    actual: str,
    expected: Any,
    scoring_method: str,
) -> ScorerResult:

    from errors import ScoringError

    try:

        if scoring_method == "exact":
            return score_exact(
                test_id,
                provider,
                actual,
                str(expected),
            )

        elif scoring_method == "semantic":
            return score_semantic(
                test_id,
                provider,
                actual,
                str(expected),
            )

        elif scoring_method == "json_match":
            return score_json_match(
                test_id,
                provider,
                actual,
                expected,
            )

        elif scoring_method == "format_compliance":
            return score_format_compliance(
                test_id,
                provider,
                actual,
                expected,
            )

        elif scoring_method == "intent_match":
            return score_intent_match(
                test_id,
                provider,
                actual,
                str(expected),
            )

        elif scoring_method == "factual_grounding":
            return score_factual_grounding(
                test_id,
                provider,
                actual,
                expected,
            )

        elif scoring_method == "llm_judge":
            return score_llm_judge(
                test_id,
                provider,
                actual,
                expected,
            )

        else:
            raise ValueError(
                f"Unknown scoring method: {scoring_method}"
            )

    except Exception as exc:

        if not isinstance(exc, ValueError):
            raise ScoringError(
                str(exc),
                test_id=test_id,
            ) from exc

        raise
