"""
golden_suite.py — GrabOn eval pipeline test suite.

SOURCE OF TRUTH: tests.json (140 cases total)
  - deal_copy_generation:            45 cases (4 channels, 3 languages, adversarial)
  - insurance_intent_classification: 40 cases (5 labels, confidence calibration)
  - credit_narrative_faithfulness:   55 cases (adversarial, edge cases, neg GMV)

VALIDATION FIXES applied during integration (see full list at bottom of file):
  1. Insurance labels changed from old set to new tests.json set
  2. Multi-coupon codes handled (deal_005, deal_017)
  3. Adversarial expired-offer cases flagged (deal_008, deal_028)
  4. Credit edge cases: zero txns, negative GMV, extreme refund rates
  5. Confidence calibration metadata preserved for 3 insurance cases
"""

import json
from pathlib import Path
from typing import Any, Literal
from pydantic import BaseModel

TaskType = Literal[
    "deal_copy", "insurance_intent", "credit_narrative",
    "summarization", "classification", "extraction",
]
ScoringMethod = Literal[
    "format_compliance", "intent_match", "factual_grounding",
    "llm_judge", "semantic", "exact", "json_match",
]

INSURANCE_LABELS = frozenset({
    "device_protection",
    "travel_insurance",
    "health_micro",
    "accidental_damage",
    "no_insurance",
})


class TestCase(BaseModel):
    id: str
    task_type: TaskType
    prompt: str
    expected: Any
    scoring_method: ScoringMethod
    metadata: dict = {}


def _build_deal_prompt(c: dict) -> str:
    lang_map = {
        "english":  "Respond in English.",
        "hindi":    "Respond in Hindi (Devanagari script).",
        "telugu":   "Respond in Telugu script.",
        "hinglish": "Respond in Hinglish (Hindi-English mix, Roman script).",
    }
    const = c["constraints"]
    if "max_chars" in const:
        limit_str = f"Keep copy within {const['max_chars']} characters total."
    else:
        limit_str = (
            f"Title: max {const['title_max_chars']} chars. "
            f"Body: max {const['body_max_chars']} chars."
        )

    coupon = c["coupon_code"]
    if "/" in coupon:
        codes = [x.strip() for x in coupon.split("/")]
        coupon_line = f"Coupon code (use either): {' or '.join(codes)}."
    else:
        coupon_line = f"Coupon code to include: {coupon}."

    adversarial_note = ""
    if c["metadata"].get("adversarial"):
        modes = c["metadata"].get("expected_failure_modes", [])
        adversarial_note = f"\nChallenge: avoid {', '.join(modes)}."

    return (
        f"Write GrabOn deal copy for the following offer.\n"
        f"Brand: {c['brand']}\n"
        f"Category: {c['category']}\n"
        f"Channel: {c['channel']}\n"
        f"Discount: {c['discount']}\n"
        f"{coupon_line}\n"
        f"Language: {c['language']}. {lang_map.get(c['language'], '')}\n"
        f"{limit_str}\n"
        f"Make the copy persuasive, factual, and channel-appropriate."
        f"{adversarial_note}\n"
        f"Respond with ONLY the copy text, no explanation."
    )


def _build_deal_expected(c: dict) -> dict:
    coupon = c["coupon_code"]
    if "/" in coupon:
        coupon_any = [x.strip() for x in coupon.split("/")]
        required_phrases = [coupon_any[0]]   # primary code required
    else:
        coupon_any = [coupon]
        required_phrases = [coupon]

    const = c["constraints"]
    base = {
        "required_phrases":  required_phrases,
        "coupon_any":        coupon_any,
        "adversarial":       c["metadata"].get("adversarial", False),
        "expired_offer":     "expired" in c["discount"].lower(),
        "failure_modes":     c["metadata"].get("expected_failure_modes", []),
    }
    if "max_chars" in const:
        base["max_chars"] = const["max_chars"]
    else:
        base["title_max_chars"] = const["title_max_chars"]
        base["body_max_chars"]  = const["body_max_chars"]
    return base


def _make_deal_case(raw: dict) -> TestCase:
    return TestCase(
        id=raw["id"],
        task_type="deal_copy",
        prompt=_build_deal_prompt(raw),
        expected=_build_deal_expected(raw),
        scoring_method="format_compliance",
        metadata={
            "channel":     raw["channel"],
            "language":    raw["language"],
            "brand":       raw["brand"],
            "adversarial": raw["metadata"].get("adversarial", False),
            "difficulty":  raw["metadata"].get("difficulty", "medium"),
            "golden_case": raw["metadata"].get("golden_case", False),
            "coupon_code": raw["coupon_code"],
            "discount":    raw["discount"],
        },
    )


def _build_insurance_prompt(c: dict) -> str:
    labels_str = ", ".join(sorted(INSURANCE_LABELS))
    return (
        f"A user is shopping on GrabOn. Recommend the most appropriate insurance "
        f"category at checkout based on their profile.\n\n"
        f"Deal category: {c['deal_category']}\n"
        f"Cart value: ₹{c['cart_value']:,}\n"
        f"User age: {c['user_age']}\n"
        f"Purchase frequency: {c['purchase_frequency']}\n"
        f"Travel frequency: {c['travel_frequency']}\n"
        f"Risk profile: {c['risk_profile']}\n"
        f"Location: {c['user_location']}\n\n"
        f"Return ONLY one label from: {labels_str}\n"
        f"Reply with just the label, nothing else."
    )


def _make_insurance_case(raw: dict) -> TestCase:
    label = raw["expected_label"]
    if label not in INSURANCE_LABELS:
        raise ValueError(f"{raw['id']}: unknown label '{label}'. Valid: {INSURANCE_LABELS}")
    return TestCase(
        id=raw["id"],
        task_type="insurance_intent",
        prompt=_build_insurance_prompt(raw),
        expected=label,
        scoring_method="intent_match",
        metadata={
            "deal_category":    raw["deal_category"],
            "cart_value":       raw["cart_value"],
            "user_age":         raw["user_age"],
            "risk_profile":     raw["risk_profile"],
            "adversarial":      raw["metadata"].get("adversarial", False),
            "difficulty":       raw["metadata"].get("difficulty", "medium"),
            "language":         raw["metadata"].get("language", "english"),
            "golden_case":      raw["metadata"].get("golden_case", False),
            "confidence_range": raw.get("expected_confidence_range"),
        },
    )


def _build_credit_prompt(c: dict) -> str:
    lang_map = {
        "english":  "Write the narrative in English.",
        "hindi":    "Write in Hindi.",
        "telugu":   "Write in Telugu.",
        "hinglish": "Write in Hinglish (Hindi-English mix).",
    }
    adversarial_note = (
        "\nIMPORTANT: Do NOT invent data. Do NOT round or exaggerate metrics. "
        "Every claim must trace directly to the numbers above."
        if c["metadata"].get("adversarial") else ""
    )
    return (
        f"Write a GrabCredit loan eligibility narrative for this business.\n"
        f"Every claim must be directly traceable to the data provided.\n"
        f"Do NOT hallucinate statistics or invent data not in the input.\n\n"
        f"Business category: {c['business_category']}\n"
        f"City: {c['city']}\n"
        f"GMV growth (YoY): {c['gmv_growth_yoy']}%\n"
        f"Monthly transactions: {c['monthly_transactions']}\n"
        f"Repeat customer rate: {c['repeat_customer_rate']}%\n"
        f"Customer retention rate: {c['customer_retention_rate']}%\n"
        f"Average order value: ₹{c['avg_order_value']:,}\n"
        f"Late payments: {c['late_payments']}\n"
        f"Refund rate: {c['refund_rate']}%\n\n"
        f"{lang_map.get(c['metadata'].get('language','english'), '')}"
        f"{adversarial_note}\n\n"
        f"Write a concise 3-5 sentence narrative. Respond with only the narrative."
    )


def _build_credit_expected(c: dict) -> dict:
    return {
        "gmv_growth_yoy":          c["gmv_growth_yoy"],
        "monthly_transactions":     c["monthly_transactions"],
        "repeat_customer_rate":     c["repeat_customer_rate"],
        "customer_retention_rate":  c["customer_retention_rate"],
        "avg_order_value":          c["avg_order_value"],
        "late_payments":            c["late_payments"],
        "refund_rate":              c["refund_rate"],
        "business_category":        c["business_category"],
        "expected_claims":          c.get("expected_claims", []),
        "adversarial":              c["metadata"].get("adversarial", False),
        "expected_failure_modes":   c["metadata"].get("expected_failure_modes", []),
        "zero_transactions":        c["monthly_transactions"] == 0,
        "negative_gmv":             c["gmv_growth_yoy"] < 0,
    }


def _make_credit_case(raw: dict) -> TestCase:
    return TestCase(
        id=raw["id"],
        task_type="credit_narrative",
        prompt=_build_credit_prompt(raw),
        expected=_build_credit_expected(raw),
        scoring_method="factual_grounding",
        metadata={
            "business_category": raw["business_category"],
            "city":              raw["city"],
            "adversarial":       raw["metadata"].get("adversarial", False),
            "difficulty":        raw["metadata"].get("difficulty", "medium"),
            "language":          raw["metadata"].get("language", "english"),
            "golden_case":       raw["metadata"].get("golden_case", False),
        },
    )


def _load_tests_json() -> dict:
    candidates = [
        Path(__file__).parent.parent / "tests.json",
        Path(__file__).parent / "tests.json",
        Path("tests.json"),
    ]
    for p in candidates:
        if p.exists():
            with open(p, encoding="utf-8") as f:
                return json.load(f)
    raise FileNotFoundError("tests.json not found. Place it at project root.")


def _validate_and_report(data: dict) -> list[str]:
    warnings = []
    deal  = data.get("deal_copy_generation", [])
    ins   = data.get("insurance_intent_classification", [])
    cred  = data.get("credit_narrative_faithfulness", [])

    if len(deal) < 30:  raise ValueError(f"deal_copy: need ≥30, got {len(deal)}")
    if len(ins)  < 30:  raise ValueError(f"insurance_intent: need ≥30, got {len(ins)}")
    if len(cred) < 30:  raise ValueError(f"credit_narrative: need ≥30, got {len(cred)}")

    all_ids = [c["id"] for c in deal + ins + cred]
    dupes = {i for i in all_ids if all_ids.count(i) > 1}
    if dupes: raise ValueError(f"Duplicate IDs: {dupes}")

    for c in ins:
        if c["expected_label"] not in INSURANCE_LABELS:
            raise ValueError(f"{c['id']}: bad label '{c['expected_label']}'")

    for c in deal:
        if "/" in c["coupon_code"]:
            warnings.append(f"{c['id']}: multi-coupon '{c['coupon_code']}' — either code accepted")
        if "expired" in c["discount"].lower() and c["metadata"].get("adversarial"):
            warnings.append(f"{c['id']}: expired-offer adversarial — model must not reproduce 'expired'")

    for c in cred:
        if c["monthly_transactions"] == 0:
            warnings.append(f"{c['id']}: zero monthly_transactions — hallucination trap")
        if c["gmv_growth_yoy"] < 0:
            warnings.append(f"{c['id']}: negative GMV {c['gmv_growth_yoy']}% — must report decline accurately")
        if c["refund_rate"] > 20:
            warnings.append(f"{c['id']}: extreme refund rate {c['refund_rate']}% — hallucination trap")

    return warnings


_data = _load_tests_json()
_VALIDATION_WARNINGS = _validate_and_report(_data)

DEAL_COPY_CASES:        list[TestCase] = [_make_deal_case(c)      for c in _data["deal_copy_generation"]]
INSURANCE_INTENT_CASES: list[TestCase] = [_make_insurance_case(c) for c in _data["insurance_intent_classification"]]
CREDIT_NARRATIVE_CASES: list[TestCase] = [_make_credit_case(c)    for c in _data["credit_narrative_faithfulness"]]

ALL_TEST_CASES: list[TestCase] = DEAL_COPY_CASES + INSURANCE_INTENT_CASES + CREDIT_NARRATIVE_CASES

SUITE_STATS = {
    "total":            len(ALL_TEST_CASES),
    "deal_copy":        len(DEAL_COPY_CASES),
    "insurance_intent": len(INSURANCE_INTENT_CASES),
    "credit_narrative": len(CREDIT_NARRATIVE_CASES),
    "adversarial": {
        "deal_copy":        sum(1 for c in DEAL_COPY_CASES        if c.metadata.get("adversarial")),
        "insurance_intent": sum(1 for c in INSURANCE_INTENT_CASES if c.metadata.get("adversarial")),
        "credit_narrative": sum(1 for c in CREDIT_NARRATIVE_CASES if c.metadata.get("adversarial")),
    },
    "by_language": {
        lang: sum(1 for c in ALL_TEST_CASES if c.metadata.get("language") == lang)
        for lang in ("english", "hindi", "telugu", "hinglish")
    },
    "by_difficulty": {
        d: sum(1 for c in ALL_TEST_CASES if c.metadata.get("difficulty") == d)
        for d in ("easy", "medium", "hard")
    },
    "validation_warnings": len(_VALIDATION_WARNINGS),
}


if __name__ == "__main__":
    print("=" * 60)
    print("GrabOn Eval Suite — Validation Report")
    print("=" * 60)
    print(f"\nTotal: {SUITE_STATS['total']} test cases")
    for k in ("deal_copy", "insurance_intent", "credit_narrative"):
        adv = SUITE_STATS["adversarial"][k]
        print(f"  {k:25}: {SUITE_STATS[k]:3d} total  ({adv} adversarial)")
    print(f"\nLanguage breakdown: {SUITE_STATS['by_language']}")
    print(f"Difficulty: {SUITE_STATS['by_difficulty']}")

    from collections import Counter
    label_dist = Counter(c.expected for c in INSURANCE_INTENT_CASES)
    print(f"\nInsurance label distribution:")
    for label, cnt in sorted(label_dist.items()):
        print(f"  {label:25}: {cnt}")

    conf_cases = [c.id for c in INSURANCE_INTENT_CASES if c.metadata.get("confidence_range")]
    if conf_cases:
        print(f"\nConfidence calibration cases: {conf_cases}")

    edge_credit = [c.id for c in CREDIT_NARRATIVE_CASES
                   if c.expected.get("zero_transactions") or c.expected.get("negative_gmv")]
    print(f"\nCredit edge cases (zero txns / neg GMV): {edge_credit}")

    if _VALIDATION_WARNINGS:
        print(f"\nValidation warnings ({len(_VALIDATION_WARNINGS)}):")
        for w in _VALIDATION_WARNINGS:
            print(f"  ⚠  {w}")
    else:
        print("\n✓ No validation warnings")

    print("\n✓ All cases loaded successfully")
