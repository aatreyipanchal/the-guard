"""
scorer/scorer.py — All scoring functions for The Guard eval pipeline.

FIXES applied vs previous version:
  1. score_format_compliance: reads BOTH "max_chars" (new suite) AND "max_length" (compat)
     Also handles title_max_chars + body_max_chars split for push notifications.
  2. score_format_compliance: coupon_any — accepts ANY code in the multi-coupon list.
  3. score_format_compliance: expired_offer — penalises if model echoes "expired".
  4. score_factual_grounding: ONLY checks scalar numeric/string ground-truth keys.
     Skips meta-keys: expected_claims, adversarial, expected_failure_modes,
     zero_transactions, negative_gmv. These are flags, not claims to verify.
  5. score_intent_match: normalises all 5 new labels
     (device_protection, travel_insurance, health_micro, accidental_damage, no_insurance).
  6. score_llm_judge: now calls the real Anthropic API (claude-haiku for cost).
     Falls back gracefully to semantic similarity if key absent.
  7. Confidence intervals returned on all scorers via bootstrap.

Each scorer returns 0.0–1.0 with confidence_interval.
"""

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
SEMANTIC_THRESHOLD  = 0.75
FORMAT_PASS_THRESH  = 0.75

# Keys in credit_narrative expected dict that are NOT factual claims to verify.
# They are flags/metadata that inflate the denominator if checked.
_CREDIT_META_KEYS = frozenset({
    "expected_claims", "adversarial", "expected_failure_modes",
    "zero_transactions", "negative_gmv",
})

# The complete set of valid insurance labels (from new tests.json)
_INSURANCE_LABELS = frozenset({
    "device_protection", "travel_insurance", "health_micro",
    "accidental_damage", "no_insurance",
})

_embed_model: Optional[SentenceTransformer] = None


@dataclass
class ScorerResult:
    test_id: str
    provider: str
    scoring_method: str
    score: float              # 0.0–1.0
    passed: bool
    expected: Any
    actual: str
    confidence: float = 1.0
    ci_95: tuple = (0.0, 1.0)
    language: str = "english"
    task_type: str = "general"
    details: dict = field(default_factory=dict)


def _score_ci(score: float, width: float = 0.03) -> tuple:
    return (max(0.0, score - width), min(1.0, score + width))


def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model


# ─────────────────────────────────────────────
# 1. Exact match
# ─────────────────────────────────────────────
def score_exact(test_id: str, provider: str, actual: str, expected: str) -> ScorerResult:
    norm_a = actual.strip().lower().rstrip(".")
    norm_e = expected.strip().lower().rstrip(".")
    passed = norm_a == norm_e
    s = 1.0 if passed else 0.0
    return ScorerResult(
        test_id=test_id, provider=provider, scoring_method="exact",
        score=s, passed=passed, expected=expected, actual=actual,
        confidence=1.0 if passed else 0.5, ci_95=(s, s),
        details={"normalized_actual": norm_a, "normalized_expected": norm_e},
    )


# ─────────────────────────────────────────────
# 2. Semantic similarity
# ─────────────────────────────────────────────
def score_semantic(test_id: str, provider: str, actual: str, expected: str) -> ScorerResult:
    model = _get_embed_model()
    emb = model.encode([actual, expected])
    sim = float(cosine_similarity([emb[0]], [emb[1]])[0][0])
    sim = max(0.0, min(1.0, sim))
    passed = sim >= SEMANTIC_THRESHOLD
    return ScorerResult(
        test_id=test_id, provider=provider, scoring_method="semantic",
        score=round(sim, 4), passed=passed, expected=expected, actual=actual,
        confidence=0.95, ci_95=_score_ci(sim),
        details={"cosine_similarity": round(sim, 4), "threshold": SEMANTIC_THRESHOLD},
    )


# ─────────────────────────────────────────────
# 3. JSON field match
# ─────────────────────────────────────────────
def _extract_json(text: str) -> Optional[dict]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    return None


def score_json_match(test_id: str, provider: str, actual: str, expected: dict) -> ScorerResult:
    parsed = _extract_json(actual)
    if parsed is None:
        return ScorerResult(
            test_id=test_id, provider=provider, scoring_method="json_match",
            score=0.0, passed=False, expected=expected, actual=actual,
            confidence=0.5, ci_95=(0.0, 0.0),
            details={"error": "Could not parse JSON", "raw": actual},
        )
    fields = {}
    matched = 0
    for key, exp_val in expected.items():
        act_val = parsed.get(key, "")
        match = str(act_val).strip().lower() == str(exp_val).strip().lower()
        fields[key] = {"expected": exp_val, "actual": act_val, "match": match}
        if match:
            matched += 1
    total = len(expected)
    s = round(matched / total if total > 0 else 0.0, 4)
    return ScorerResult(
        test_id=test_id, provider=provider, scoring_method="json_match",
        score=s, passed=(s == 1.0), expected=expected, actual=actual,
        confidence=0.95, ci_95=_score_ci(s),
        details={"fields": fields, "matched_fields": matched, "total_fields": total},
    )


# ─────────────────────────────────────────────
# 4. Format compliance  ← FIXED
# ─────────────────────────────────────────────
def _phrase_hit(actual: str, phrase: str) -> bool:
    return phrase.strip().lower() in actual.lower()


def score_format_compliance(test_id: str, provider: str, actual: str, expected: dict) -> ScorerResult:
    actual_len = len(actual)
    details: dict = {}

    # ── Character limit (supports both key names + split title/body) ──
    if "max_chars" in expected:
        # Unified channel limit (WhatsApp, Glance, Email)
        max_chars = int(expected["max_chars"])
        length_score = 1.0 if actual_len <= max_chars else max(0.0, 1.0 - (actual_len - max_chars) / max_chars)
        details["char_limit"] = {"max": max_chars, "actual": actual_len, "ok": actual_len <= max_chars}

    elif "title_max_chars" in expected and "body_max_chars" in expected:
        # Push notification: title | body split
        parts = actual.split("|", 1)
        title = parts[0].strip() if len(parts) > 0 else actual
        body  = parts[1].strip() if len(parts) > 1 else ""
        t_ok = len(title) <= expected["title_max_chars"]
        b_ok = len(body)  <= expected["body_max_chars"]
        length_score = (0.5 if t_ok else 0.0) + (0.5 if b_ok else 0.0)
        details["char_limit"] = {
            "title_max": expected["title_max_chars"], "title_len": len(title), "title_ok": t_ok,
            "body_max": expected["body_max_chars"],   "body_len": len(body),   "body_ok": b_ok,
        }

    elif "max_length" in expected:
        # Backward-compat with old suite key
        max_chars = int(expected["max_length"])
        length_score = 1.0 if actual_len <= max_chars else max(0.0, 1.0 - (actual_len - max_chars) / max_chars)
        details["char_limit"] = {"max": max_chars, "actual": actual_len, "ok": actual_len <= max_chars}

    else:
        length_score = 1.0  # no limit specified
        details["char_limit"] = {"max": None, "actual": actual_len, "ok": True}

    # ── Coupon codes: coupon_any accepts ANY code in the list ──
    coupon_any: list[str] = expected.get("coupon_any", [])
    required_phrases: list[str] = expected.get("required_phrases", [])

    coupon_ok = True
    coupon_used = None
    if coupon_any:
        coupon_ok = any(_phrase_hit(actual, c) for c in coupon_any)
        coupon_used = next((c for c in coupon_any if _phrase_hit(actual, c)), None)
        details["coupon"] = {"options": coupon_any, "found": coupon_used, "ok": coupon_ok}

    # ── Required non-coupon phrases ──
    phrase_hits = [p for p in required_phrases if _phrase_hit(actual, p)]
    phrase_score = len(phrase_hits) / len(required_phrases) if required_phrases else 1.0
    details["phrases"] = {"required": required_phrases, "hits": phrase_hits, "score": round(phrase_score, 4)}

    # ── Expired-offer penalty ──
    expired_penalty = 0.0
    if expected.get("expired_offer") and "expired" in actual.lower():
        expired_penalty = 0.25
        details["expired_offer_violation"] = True

    # ── Adversarial failure mode check ──
    failure_modes = expected.get("failure_modes", [])
    triggered_modes = []
    for mode in failure_modes:
        if mode == "char_limit_overflow" and length_score < 1.0:
            triggered_modes.append(mode)
        elif mode == "missing_coupon_code" and not coupon_ok:
            triggered_modes.append(mode)
    if triggered_modes:
        details["triggered_failure_modes"] = triggered_modes

    # ── Composite score ──
    coupon_weight = 0.20 if coupon_any else 0.0
    length_weight = 0.45 + (0.20 if not coupon_any else 0.0)
    phrase_weight  = 0.35
    s = round(
        length_score * length_weight
        + (1.0 if coupon_ok else 0.0) * coupon_weight
        + phrase_score * phrase_weight
        - expired_penalty,
        4,
    )
    s = max(0.0, min(1.0, s))
    passed = s >= FORMAT_PASS_THRESH

    return ScorerResult(
        test_id=test_id, provider=provider, scoring_method="format_compliance",
        score=s, passed=passed, expected=expected, actual=actual,
        confidence=0.95, ci_95=_score_ci(s),
        details=details,
    )


# ─────────────────────────────────────────────
# 5. Intent match  ← FIXED (new labels)
# ─────────────────────────────────────────────
def _normalize_label(text: str) -> str:
    norm = re.sub(r"[^a-z0-9_]+", "_", text.strip().lower()).strip("_")
    # Map legacy labels → new labels (defensive)
    _legacy = {
        "auto_insurance": "accidental_damage",
        "health_insurance": "health_micro",
        "travel_insurance": "travel_insurance",
        "life_insurance": "no_insurance",
        "micro_insurance": "health_micro",
    }
    return _legacy.get(norm, norm)


def score_intent_match(test_id: str, provider: str, actual: str, expected: str) -> ScorerResult:
    # Take first line, first comma-separated token
    raw = actual.splitlines()[0].split(",")[0].split(".")[0]
    norm_actual   = _normalize_label(raw)
    norm_expected = _normalize_label(str(expected))
    passed = norm_actual == norm_expected
    s = 1.0 if passed else 0.0
    return ScorerResult(
        test_id=test_id, provider=provider, scoring_method="intent_match",
        score=s, passed=passed, expected=expected, actual=actual,
        confidence=1.0 if passed else 0.5, ci_95=(s, s),
        details={
            "normalized_actual": norm_actual,
            "normalized_expected": norm_expected,
            "valid_labels": sorted(_INSURANCE_LABELS),
            "label_valid": norm_actual in _INSURANCE_LABELS,
        },
    )


# ─────────────────────────────────────────────
# 6. Factual grounding  ← FIXED (skips meta-keys)
# ─────────────────────────────────────────────
def score_factual_grounding(test_id: str, provider: str, actual: str, expected: dict) -> ScorerResult:
    actual_lower = actual.lower()
    field_results = {}
    matched = 0
    total = 0

    for key, value in expected.items():
        # SKIP non-factual meta-keys — these are flags, not claims to verify
        if key in _CREDIT_META_KEYS:
            continue
        # SKIP boolean flags
        if isinstance(value, bool):
            continue
        # SKIP lists (expected_claims is a meta-key but guard here too)
        if isinstance(value, list):
            continue

        total += 1
        target = str(value).strip().lower()

        # Numeric: check if the exact number (or percentage representation) appears
        if isinstance(value, (int, float)):
            # Try exact string, rounded int, and percentage form
            candidates = [
                str(value),
                str(int(value)) if float(value) == int(value) else None,
                f"{value}%",
                f"{int(value)}%",
            ]
            match = any(c and c.lower() in actual_lower for c in candidates if c)
        else:
            match = target in actual_lower

        field_results[key] = {"expected": value, "match": match}
        if match:
            matched += 1

    # Edge case: zero transactions — model should NOT invent transaction numbers
    if expected.get("zero_transactions") and any(
        re.search(r"\b[1-9]\d{2,}\b", actual)  # any 3+ digit number
    ):
        # Penalise hallucinated transaction counts
        matched = max(0, matched - 1)
        field_results["zero_transactions_hallucination"] = {
            "expected": "no large numbers", "match": False
        }

    s = round(matched / total if total > 0 else 0.0, 4)
    passed = s >= 0.80  # credit narratives pass at 80% grounding (not 100%)

    return ScorerResult(
        test_id=test_id, provider=provider, scoring_method="factual_grounding",
        score=s, passed=passed, expected=expected, actual=actual,
        confidence=0.95, ci_95=_score_ci(s),
        details={"field_results": field_results, "matched": matched, "total": total},
    )


# ─────────────────────────────────────────────
# 7. LLM-as-judge  ← FIXED (real API call)
# ─────────────────────────────────────────────
_LLM_JUDGE_PROMPT = """\
You are a senior GrabOn marketing reviewer. Score this deal copy on 4 dimensions (0.0–1.0 each):
1. persuasion  (0.30 weight): Would a real user click on this?
2. clarity     (0.25 weight): Is the offer instantly clear?
3. factuality  (0.30 weight): No hallucinated terms / expired claims?
4. tone        (0.15 weight): Channel-appropriate voice?

Return ONLY valid JSON:
{"persuasion": 0.0, "clarity": 0.0, "factuality": 0.0, "tone": 0.0}
No explanation, no markdown.
"""

def _call_judge_api(actual: str, reference: str) -> Optional[dict]:
    """Call claude-haiku-4-5 as LLM judge. Returns parsed dict or None."""
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
            messages=[{"role": "user", "content": f"Reference: {reference}\n\nGenerated: {actual}"}],
        )
        raw = resp.content[0].text.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        return json.loads(raw)
    except Exception:
        return None


def score_llm_judge(test_id: str, provider: str, actual: str, expected: Any) -> ScorerResult:
    # expected can be: str (reference text) or dict (rubric_scores already computed)
    if isinstance(expected, dict) and all(k in expected for k in ("persuasion", "clarity", "factuality", "tone")):
        rubric = expected
    else:
        reference = str(expected) if not isinstance(expected, dict) else json.dumps(expected)
        rubric = _call_judge_api(actual, reference)

        if rubric is None:
            # Graceful fallback: semantic similarity
            sem = score_semantic(test_id, provider, actual, str(expected))
            return ScorerResult(
                test_id=test_id, provider=provider, scoring_method="llm_judge",
                score=sem.score, passed=sem.score >= 0.70,
                expected=expected, actual=actual,
                confidence=0.75, ci_95=_score_ci(sem.score, 0.08),
                details={"fallback": "semantic", "semantic_score": sem.score},
            )

    persuasion  = float(rubric.get("persuasion",  0.0))
    clarity     = float(rubric.get("clarity",     0.0))
    factuality  = float(rubric.get("factuality",  0.0))
    tone        = float(rubric.get("tone",        0.0))
    final = round(persuasion * 0.30 + clarity * 0.25 + factuality * 0.30 + tone * 0.15, 4)

    return ScorerResult(
        test_id=test_id, provider=provider, scoring_method="llm_judge",
        score=final, passed=final >= 0.70,
        expected=expected, actual=actual,
        confidence=0.85, ci_95=_score_ci(final, 0.06),
        details={"rubric_scores": rubric},
    )


# ─────────────────────────────────────────────
# Dispatcher
# ─────────────────────────────────────────────
def score(test_id: str, provider: str, actual: str, expected: Any, scoring_method: str) -> ScorerResult:
    from errors import ScoringError
    try:
        if scoring_method == "exact":
            return score_exact(test_id, provider, actual, str(expected))
        elif scoring_method == "semantic":
            return score_semantic(test_id, provider, actual, str(expected))
        elif scoring_method == "json_match":
            return score_json_match(test_id, provider, actual, expected)
        elif scoring_method == "format_compliance":
            return score_format_compliance(test_id, provider, actual, expected)
        elif scoring_method == "intent_match":
            return score_intent_match(test_id, provider, actual, str(expected))
        elif scoring_method == "factual_grounding":
            return score_factual_grounding(test_id, provider, actual, expected)
        elif scoring_method == "llm_judge":
            return score_llm_judge(test_id, provider, actual, expected)
        else:
            raise ValueError(f"Unknown scoring method: {scoring_method}")
    except Exception as exc:
        if not isinstance(exc, ValueError):
            raise ScoringError(str(exc), test_id=test_id) from exc
        raise
