# The Guard — Eval Report

**Run ID:** `run_20260508_101227_456e90`  
**Generated:** 2026-05-08 10:17 UTC  

## Deployment Decision: **NO-GO**

## Test Suite

| Task type | Cases |
|-----------|-------|
| Deal Copy Generation | 45 |
| Insurance Intent Classification | 40 |
| Credit Narrative Faithfulness | 55 |
| Summarization Quality | 10 |
| Customer Query Classification | 10 |
| Structured Data Extraction | 10 |
| **Total** | **170** |

## Provider Results

### ✅ OPENAI (`gpt-4o-mini`)

> openai: GO. No regressions detected. Pass rate 72.4% (baseline: 72.4%). ✓

| Metric | Current | 95% CI | Baseline | Delta | p-value | Status |
|--------|---------|---------|----------|-------|---------|--------|
| accuracy | 0.8236 | [-0.0556, 0.0581] | 0.8225 | +0.0011 (+0.1%) | 0.9705 | 🟢 ok |
| latency_ms | 1922.1816 | — | 1986.3524 | -64.1708 (-3.2%) | 0.6531 | 🟢 ok |
| cost_usd | 0.0000 | — | 0.0000 | +0.0000 (+0.1%) | 0.9894 | 🟢 ok |
| deal_copy_accuracy | 0.9395 | — | 0.9410 | -0.0015 (-0.2%) | 0.9419 | 🟢 ok |
| insurance_intent_accuracy | 0.6500 | — | 0.6500 | +0.0000 (+0.0%) | 1.0000 | 🟢 ok |
| credit_narrative_accuracy | 0.8023 | — | 0.7977 | +0.0045 (+0.6%) | 0.7496 | 🟢 ok |
| summarization_accuracy | 0.8951 | — | 0.8939 | +0.0012 (+0.1%) | 0.9431 | 🟢 ok |
| classification_accuracy | 1.0000 | — | 1.0000 | +0.0000 (+0.0%) | nan | 🟢 ok |
| extraction_accuracy | 0.8667 | — | 0.8667 | +0.0000 (+0.0%) | 1.0000 | 🟢 ok |
| overall_pass | 0.7235 | — | 0.7235 | +0.0000 (+0.0%) | 0.8802 | 🟢 ok |
| deal_copy_pass | 1.0000 | — | 1.0000 | +0.0000 (+0.0%) | 1.0000 | 🟢 ok |
| insurance_intent_pass | 0.6500 | — | 0.6500 | +0.0000 (+0.0%) | 0.8231 | 🟢 ok |
| credit_narrative_pass | 0.4545 | — | 0.4545 | +0.0000 (+0.0%) | 0.8137 | 🟢 ok |
| summarization_pass | 1.0000 | — | 1.0000 | +0.0000 (+0.0%) | 1.0000 | 🟢 ok |
| classification_pass | 1.0000 | — | 1.0000 | +0.0000 (+0.0%) | 1.0000 | 🟢 ok |
| extraction_pass | 0.7000 | — | 0.7000 | +0.0000 (+0.0%) | 0.6171 | 🟢 ok |

**Pass rate:** 72.4% &nbsp;|&nbsp; **Avg latency:** 1922 ms &nbsp;|&nbsp; **Total cost:** $0.008469

#### Language Breakdown

| Language | Avg Score |
|----------|------------|
| english | 0.8643 |
| hinglish | 0.8875 |
| telugu | 0.7830 |
| hindi | 0.7698 |

<details><summary>Statistical test details</summary>

| Metric | Test | Statistic | p-value | Significant | Regressed |
|--------|------|-----------|---------|-------------|-----------|
| accuracy | Paired bootstrap | 0.0000 | 0.9705 | no | no |
| latency_ms | Welch's t-test | -0.4498 | 0.6531 | no | no |
| cost_usd | Welch's t-test | 0.0133 | 0.9894 | no | no |
| deal_copy_accuracy | Welch's t-test | -0.0731 | 0.9419 | no | no |
| insurance_intent_accuracy | Welch's t-test | 0.0000 | 1.0000 | no | no |
| credit_narrative_accuracy | Welch's t-test | 0.3200 | 0.7496 | no | no |
| summarization_accuracy | Welch's t-test | 0.0723 | 0.9431 | no | no |
| classification_accuracy | Welch's t-test | nan | nan | no | no |
| extraction_accuracy | Welch's t-test | 0.0000 | 1.0000 | no | no |
| overall_pass | McNemar's test | 0.0227 | 0.8802 | no | no |
| deal_copy_pass | McNemar's test | 0.0000 | 1.0000 | no | no |
| insurance_intent_pass | McNemar's test | 0.0500 | 0.8231 | no | no |
| credit_narrative_pass | McNemar's test | 0.0556 | 0.8137 | no | no |
| summarization_pass | McNemar's test | 0.0000 | 1.0000 | no | no |
| classification_pass | McNemar's test | 0.0000 | 1.0000 | no | no |
| extraction_pass | McNemar's test | 0.2500 | 0.6171 | no | no |

</details>

### ❌ GEMINI (`gemini-2.5-flash-lite`)

> gemini: NO-GO — Regression detected. Pass rate 72.4% (baseline: 72.4%). 
Details: latency_ms regressed by 108.83% (p=0.000, 95% CI=[0.0000, 0.0000])

| Metric | Current | 95% CI | Baseline | Delta | p-value | Status |
|--------|---------|---------|----------|-------|---------|--------|
| accuracy | 0.8351 | [-0.0418, 0.0421] | 0.8351 | +0.0000 (+0.0%) | 1.0000 | 🟢 ok |
| latency_ms | 2458.5250 | — | 1177.2771 | +1281.2479 (+108.8%) | 0.0000 | 🔴 REGRESSED |
| cost_usd | 0.0000 | — | 0.0000 | +0.0000 (+0.0%) | 1.0000 | 🟢 ok |
| deal_copy_accuracy | 0.9450 | — | 0.9450 | -0.0000 (-0.0%) | 1.0000 | 🟢 ok |
| insurance_intent_accuracy | 0.7250 | — | 0.7250 | +0.0000 (+0.0%) | 1.0000 | 🟢 ok |
| credit_narrative_accuracy | 0.7886 | — | 0.7886 | +0.0000 (+0.0%) | 1.0000 | 🟢 ok |
| summarization_accuracy | 0.8402 | — | 0.8402 | +0.0000 (+0.0%) | 1.0000 | 🟢 ok |
| classification_accuracy | 1.0000 | — | 1.0000 | +0.0000 (+0.0%) | nan | 🟢 ok |
| extraction_accuracy | 0.8667 | — | 0.8667 | +0.0000 (+0.0%) | 1.0000 | 🟢 ok |
| overall_pass | 0.7235 | — | 0.7235 | +0.0000 (+0.0%) | 0.8802 | 🟢 ok |
| deal_copy_pass | 1.0000 | — | 1.0000 | +0.0000 (+0.0%) | 1.0000 | 🟢 ok |
| insurance_intent_pass | 0.7250 | — | 0.7250 | +0.0000 (+0.0%) | 0.7518 | 🟢 ok |
| credit_narrative_pass | 0.4000 | — | 0.4000 | +0.0000 (+0.0%) | 0.8445 | 🟢 ok |
| summarization_pass | 1.0000 | — | 1.0000 | +0.0000 (+0.0%) | 1.0000 | 🟢 ok |
| classification_pass | 1.0000 | — | 1.0000 | +0.0000 (+0.0%) | 1.0000 | 🟢 ok |
| extraction_pass | 0.7000 | — | 0.7000 | +0.0000 (+0.0%) | 0.6171 | 🟢 ok |

**Pass rate:** 72.4% &nbsp;|&nbsp; **Avg latency:** 2459 ms &nbsp;|&nbsp; **Total cost:** $0.005217

#### Language Breakdown

| Language | Avg Score |
|----------|------------|
| hinglish | 0.8875 |
| telugu | 0.8677 |
| hindi | 0.7981 |
| english | 0.8469 |

## NO-GO Regression Details

### Prompt Version

- Prompt hash: `976d0ed1175608d5757013e6b639431d16221b3299dfe9e2d1c0e65bddbd08d3`
- Git commit: `d58b98b2a8c58e2c50bceeb2b47f2212befbfba8`
- Git branch: `main`

### Task Prompt Hashes

- `classification`: `cae6d35f0469fb9d2bda6fcf2d38f3767710208e96d561156233235ce0e2455f`
- `credit_narrative`: `afde4a2041a08b8b7ec3b4836b19f864541c0bfff496553ebf49baae07bcd690`
- `deal_copy`: `322623f2505a808ff6fabc3f425fc79aeeac1ad017002c7451325d6c40e9d3b5`
- `extraction`: `0241c2c24584067444a7224ef3207e68b17199df7684c37d9644cf6db56c711e`
- `insurance_intent`: `ffe023628595660d07ad2096b2d49f942086d8bbedaa1c85bb3eb8360ace5962`
- `summarization`: `0eefb6d60583d593d583e32cb6e93f48d84dc30d2669e63e83f8f20373e26292`

### Prompt Diff

```diff
diff --git a/tests/golden_suite.py b/tests/golden_suite.py
index 02e6153..3ee28cb 100644
--- a/tests/golden_suite.py
+++ b/tests/golden_suite.py
@@ -1,22 +1,3 @@
-"""
-golden_suite.py — GrabOn eval pipeline test suite.
-
-SOURCE OF TRUTH: tests.json (170 cases total)
-  - deal_copy_generation:            45 cases (4 channels, 3 languages, adversarial)
-  - insurance_intent_classification: 40 cases (5 labels, confidence calibration)
-  - credit_narrative_faithfulness:   55 cases (adversarial, edge cases, neg GMV)
-  - summarization:                   10 cases (multi-provider cross-val)
-  - classification:                   10 cases (customer intent sorting)
-  - extraction:                       10 cases (structured JSON data)
-
-VALIDATION FIXES applied during integration (see full list at bottom of file):
-  1. Insurance labels changed from old set to new tests.json set
-  2. Multi-coupon codes handled (deal_005, deal_017)
-  3. Adversarial expired-offer cases flagged (deal_008, deal
```

### Regression Summary

- [HIGH] latency_ms regressed by 108.83% (p=0.000, 95% CI=[0.0000, 0.0000])

<details><summary>Statistical test details</summary>

| Metric | Test | Statistic | p-value | Significant | Regressed |
|--------|------|-----------|---------|-------------|-----------|
| accuracy | Paired bootstrap | 0.0000 | 1.0000 | no | no |
| latency_ms | Welch's t-test | 13.2930 | 0.0000 | yes | **YES** |
| cost_usd | Welch's t-test | 0.0000 | 1.0000 | no | no |
| deal_copy_accuracy | Welch's t-test | -0.0000 | 1.0000 | no | no |
| insurance_intent_accuracy | Welch's t-test | 0.0000 | 1.0000 | no | no |
| credit_narrative_accuracy | Welch's t-test | 0.0000 | 1.0000 | no | no |
| summarization_accuracy | Welch's t-test | 0.0000 | 1.0000 | no | no |
| classification_accuracy | Welch's t-test | nan | nan | no | no |
| extraction_accuracy | Welch's t-test | 0.0000 | 1.0000 | no | no |
| overall_pass | McNemar's test | 0.0227 | 0.8802 | no | no |
| deal_copy_pass | McNemar's test | 0.0000 | 1.0000 | no | no |
| insurance_intent_pass | McNemar's test | 0.1000 | 0.7518 | no | no |
| credit_narrative_pass | McNemar's test | 0.0385 | 0.8445 | no | no |
| summarization_pass | McNemar's test | 0.0000 | 1.0000 | no | no |
| classification_pass | McNemar's test | 0.0000 | 1.0000 | no | no |
| extraction_pass | McNemar's test | 0.2500 | 0.6171 | no | no |

</details>

### ✅ GROQ (`llama-3.1-8b-instant`)

> groq: GO. No regressions detected. Pass rate 27.6% (baseline: 26.5%). ✓

| Metric | Current | 95% CI | Baseline | Delta | p-value | Status |
|--------|---------|---------|----------|-------|---------|--------|
| accuracy | 0.3463 | [-0.0567, 0.0909] | 0.3308 | +0.0155 (+4.7%) | 0.7070 | 🟢 ok |
| latency_ms | 138.1126 | — | 112.6010 | +25.5116 (+22.7%) | 0.1017 | 🟢 ok |
| cost_usd | 0.0000 | — | 0.0000 | +0.0000 (+2.5%) | 0.8415 | 🟢 ok |
| deal_copy_accuracy | 0.6161 | — | 0.6106 | +0.0055 (+0.9%) | 0.9481 | 🟢 ok |
| insurance_intent_accuracy | 0.2750 | — | 0.2500 | +0.0250 (+10.0%) | 0.8025 | 🟢 ok |
| credit_narrative_accuracy | 0.1727 | — | 0.1682 | +0.0045 (+2.7%) | 0.9319 | 🟢 ok |
| summarization_accuracy | 0.3318 | — | 0.3508 | -0.0189 (-5.4%) | 0.9249 | 🟢 ok |
| classification_accuracy | 0.4000 | — | 0.4000 | +0.0000 (+0.0%) | 1.0000 | 🟢 ok |
| extraction_accuracy | 0.3333 | — | 0.2000 | +0.1333 (+66.7%) | 0.4701 | 🟢 ok |
| overall_pass | 0.2765 | — | 0.2647 | +0.0118 (+4.4%) | 0.8852 | 🟢 ok |
| deal_copy_pass | 0.5556 | — | 0.5333 | +0.0222 (+4.2%) | 1.0000 | 🟢 ok |
| insurance_intent_pass | 0.2750 | — | 0.2500 | +0.0250 (+10.0%) | 1.0000 | 🟢 ok |
| credit_narrative_pass | 0.0364 | — | 0.0364 | +0.0000 (+0.0%) | 0.6171 | 🟢 ok |
| summarization_pass | 0.3000 | — | 0.4000 | -0.1000 (-25.0%) | 1.0000 | 🟢 ok |
| classification_pass | 0.4000 | — | 0.4000 | +0.0000 (+0.0%) | 0.6171 | 🟢 ok |
| extraction_pass | 0.2000 | — | 0.1000 | +0.1000 (+100.0%) | 1.0000 | 🟢 ok |

**Pass rate:** 27.6% &nbsp;|&nbsp; **Avg latency:** 138 ms &nbsp;|&nbsp; **Total cost:** $0.000854

#### Language Breakdown

| Language | Avg Score |
|----------|------------|
| english | 0.3539 |
| hinglish | 0.8875 |
| hindi | 0.3696 |
| telugu | 0.2211 |

<details><summary>Statistical test details</summary>

| Metric | Test | Statistic | p-value | Significant | Regressed |
|--------|------|-----------|---------|-------------|-----------|
| accuracy | Paired bootstrap | 0.0000 | 0.7070 | no | no |
| latency_ms | Welch's t-test | 1.6413 | 0.1017 | no | no |
| cost_usd | Welch's t-test | 0.2001 | 0.8415 | no | no |
| deal_copy_accuracy | Welch's t-test | 0.0653 | 0.9481 | no | no |
| insurance_intent_accuracy | Welch's t-test | 0.2510 | 0.8025 | no | no |
| credit_narrative_accuracy | Welch's t-test | 0.0856 | 0.9319 | no | no |
| summarization_accuracy | Welch's t-test | -0.0956 | 0.9249 | no | no |
| classification_accuracy | Welch's t-test | 0.0000 | 1.0000 | no | no |
| extraction_accuracy | Welch's t-test | 0.7386 | 0.4701 | no | no |
| overall_pass | McNemar's test | 0.0208 | 0.8852 | no | no |
| deal_copy_pass | McNemar's test | 0.0000 | 1.0000 | no | no |
| insurance_intent_pass | McNemar's test | 0.0000 | 1.0000 | no | no |
| credit_narrative_pass | McNemar's test | 0.2500 | 0.6171 | no | no |
| summarization_pass | McNemar's test | 0.0000 | 1.0000 | no | no |
| classification_pass | McNemar's test | 0.2500 | 0.6171 | no | no |
| extraction_pass | McNemar's test | 0.0000 | 1.0000 | no | no |

</details>

## Statistical Summary

- Total providers evaluated: 3
- Total regressions detected: 1
- Statistical confidence threshold: 95%
- Regression significance threshold: p < 0.05

---

*Generated by The Guard eval pipeline.*