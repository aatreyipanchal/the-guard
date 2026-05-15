import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

import threading
from concurrent.futures import ThreadPoolExecutor

from errors import (
    BudgetExceededError, TokenBudgetExceededError, ScoringError
)
from scorer.scorer import (
    score_exact, score_semantic, score_json_match,
    score_format_compliance, score_intent_match,
    score_factual_grounding, score_llm_judge,
    ScorerResult
)

logger = logging.getLogger("the_guard.agent")


# ─────────────────────────────────────────────
# Agent phases 
# ─────────────────────────────────────────────
class Phase(str, Enum):
    PLAN    = "PLAN"
    ACT     = "ACT"
    OBSERVE = "OBSERVE"
    DECIDE  = "DECIDE"
    DONE    = "DONE"
    ABORTED = "ABORTED"


@dataclass
class PhaseTransition:
    phase: Phase
    timestamp: str
    notes: str = ""


@dataclass
class AgentState:
    run_id: str
    phase: Phase = Phase.PLAN
    phase_history: list = field(default_factory=list)
    plan: dict = field(default_factory=dict)      
    responses: dict = field(default_factory=dict)   
    scores: dict = field(default_factory=dict)     
    total_cost_usd: float = 0.0
    total_tokens: int = 0
    errors: list = field(default_factory=list)      
    aborted: bool = False
    abort_reason: str = ""

    def transition(self, phase: Phase, notes: str = "") -> None:
        self.phase = phase
        self.phase_history.append(PhaseTransition(
            phase=phase,
            timestamp=datetime.now(timezone.utc).isoformat(),
            notes=notes,
        ))
        logger.info(f"[{self.run_id}] Phase -> {phase.value}  {notes}")


# ─────────────────────────────────────────────
# Budget guard
# ─────────────────────────────────────────────
@dataclass
class BudgetGuard:
    max_cost_usd: float = 5.00        # hard stop at $5
    max_tokens:   int   = 2_000_000   # hard stop at 2M tokens
    warn_cost_usd: float = 3.00       # warn at $3

    def check(self, total_cost: float, total_tokens: int, test_id: str = "") -> None:
        if total_cost >= self.max_cost_usd:
            raise BudgetExceededError(
                f"Budget exceeded: ${total_cost:.4f} >= ${self.max_cost_usd}",
                spent_usd=total_cost,
                limit_usd=self.max_cost_usd,
                test_id=test_id,
            )
        if total_tokens >= self.max_tokens:
            raise TokenBudgetExceededError(
                f"Token budget exceeded: {total_tokens:,} >= {self.max_tokens:,}",
                test_id=test_id,
            )
        if total_cost >= self.warn_cost_usd:
            logger.warning(f"Budget warn: ${total_cost:.4f} — approaching limit ${self.max_cost_usd}")


# ─────────────────────────────────────────────
# Tool registry — scorers as discoverable tools
# ─────────────────────────────────────────────
class ToolRegistry:
    def __init__(self):
        self._tools: dict = {}

    def register(self, name: str, fn, description: str = "") -> None:
        self._tools[name] = {"fn": fn, "description": description}
        logger.debug(f"ToolRegistry: registered '{name}'")

    def call(self, name: str, **kwargs):
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not registered. Available: {list(self._tools)}")
        return self._tools[name]["fn"](**kwargs)

    def list_tools(self) -> list[dict]:
        return [{"name": k, "description": v["description"]} for k, v in self._tools.items()]

    def __contains__(self, name: str) -> bool:
        return name in self._tools


def build_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register("exact",              score_exact,              "Exact string match (classification)")
    registry.register("semantic",           score_semantic,           "Cosine similarity via sentence embeddings (summarisation)")
    registry.register("json_match",         score_json_match,         "Field-level JSON comparison (extraction)")
    registry.register("format_compliance",  score_format_compliance,  "Channel char-limit + coupon + phrase compliance (deal copy)")
    registry.register("intent_match",       score_intent_match,       "Insurance intent label classification")
    registry.register("factual_grounding",  score_factual_grounding,  "Credit narrative hallucination check")
    registry.register("llm_judge",          score_llm_judge,          "LLM-as-judge for deal copy quality")
    return registry


# ─────────────────────────────────────────────
# Main agent
# ─────────────────────────────────────────────
class EvalAgent:

    MAX_RETRIES_PER_TEST = 3
    RETRY_BACKOFF_BASE   = 2.0   # seconds

    def __init__(self, providers: list, test_cases: list, budget: Optional[BudgetGuard] = None):
        self.providers   = providers
        self.test_cases  = test_cases
        self.budget      = budget or BudgetGuard()
        self.registry    = build_tool_registry()

    def run(self, run_id: str, **kwargs) -> AgentState:
        state = AgentState(run_id=run_id)
        for k, v in kwargs.items():
            setattr(state, k, v)

        try:
            self._phase_plan(state)
            self._phase_act(state)
            self._phase_observe(state)
            self._phase_decide(state)
        except BudgetExceededError as e:
            state.aborted = True
            state.abort_reason = str(e)
            state.transition(Phase.ABORTED, notes=f"Budget killed: {e}")
            logger.error(f"[{run_id}] ABORTED — {e}")
        except Exception as e:
            state.aborted = True
            state.abort_reason = str(e)
            state.transition(Phase.ABORTED, notes=f"Fatal: {e}")
            logger.exception(f"[{run_id}] ABORTED — unexpected error")

        return state

    # ── PLAN ─────────────────────────────────────
    def _phase_plan(self, state: AgentState) -> None:
        state.transition(Phase.PLAN, notes=f"{len(self.providers)} providers, {len(self.test_cases)} test cases")
        state.plan = {p.name: [tc.id for tc in self.test_cases] for p in self.providers}

        logger.info(f"[{state.run_id}] Plan: {state.plan}")

        # Log tool registry for discoverability
        tools = self.registry.list_tools()
        logger.info(f"[{state.run_id}] Tool registry ({len(tools)} tools): {[t['name'] for t in tools]}")

    # ── ACT ──────────────────────────────────────
    def _phase_act(self, state: AgentState) -> None:
        state.transition(Phase.ACT)
        tc_map = {tc.id: tc for tc in self.test_cases}
        
        state_lock = threading.Lock()
        max_workers = 5 

        for provider in self.providers:
            pname = provider.name
            state.responses[pname] = []
            test_ids = state.plan.get(pname, [])

            def _worker(test_id):
                resp = self._run_one_test(provider, tc_map[test_id], state)
                
                # SABOTAGE MODE for Demo: if simulate_regression is on, drop accuracy for insurance
                if getattr(state, "simulate_regression", False) and tc_map[test_id].task_type == "insurance_intent":
                    import random
                    if random.random() > 0.5:
                        resp.actual = "corrupted_output_for_demo"

                with state_lock:
                    state.responses[pname].append(resp)
                    state.total_cost_usd += resp.cost_usd
                    state.total_tokens   += resp.total_tokens
                    self.budget.check(state.total_cost_usd, state.total_tokens, test_id=test_id)
                return resp

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                list(executor.map(_worker, test_ids))

    def _run_one_test(self, provider, tc, state: AgentState):
        last_resp = None
        for attempt in range(self.MAX_RETRIES_PER_TEST):
            resp = provider.call(tc.prompt, tc.id)
            last_resp = resp
            if resp.succeeded:
                return resp

            # Classify the error
            err_type = resp.error or "UnknownError"
            is_retryable = any(t in err_type for t in ("APIRateLimitError", "APITimeoutError", "APIServerError"))

            state.errors.append((tc.id, err_type, provider.name))
            logger.warning(f"[{state.run_id}] {provider.name}/{tc.id} attempt={attempt+1} error={err_type}")

            if not is_retryable or attempt == self.MAX_RETRIES_PER_TEST - 1:
                break

            backoff = self.RETRY_BACKOFF_BASE ** (attempt + 1)
            logger.info(f"[{state.run_id}] Retrying {tc.id} in {backoff:.1f}s")
            time.sleep(backoff)

        return last_resp

    # ── OBSERVE ───────────────────────────────────
    def _phase_observe(self, state: AgentState) -> None:
        state.transition(Phase.OBSERVE)
        tc_map = {tc.id: tc for tc in self.test_cases}

        for provider in self.providers:
            pname = provider.name
            state.scores[pname] = []

            for resp in state.responses[pname]:
                tc = tc_map[resp.test_id]

                if not resp.succeeded:
                    result = ScorerResult(
                        test_id=tc.id, provider=pname, scoring_method=tc.scoring_method,
                        score=0.0, passed=False, expected=tc.expected, actual="",
                        details={"error": resp.error},
                    )
                else:
                    # Use tool registry — discoverable, not hardcoded
                    try:
                        result = self.registry.call(
                            tc.scoring_method,
                            test_id=tc.id, provider=pname,
                            actual=resp.output, expected=tc.expected,
                        )
                    except (KeyError, ScoringError) as e:
                        result = ScorerResult(
                            test_id=tc.id, provider=pname, scoring_method=tc.scoring_method,
                            score=0.0, passed=False, expected=tc.expected, actual=resp.output,
                            details={"scoring_error": str(e)},
                        )

                state.scores[pname].append(result)

        logger.info(f"[{state.run_id}] Observe complete — {sum(len(v) for v in state.scores.values())} scores")

    # ── DECIDE ────────────────────────────────────
    def _phase_decide(self, state: AgentState) -> None:
        state.transition(Phase.DECIDE)
        # Actual detector logic runs in run_eval.py on state.scores + state.responses
        # The agent surfaces the verdict via the state object
        state.transition(Phase.DONE, notes="Verdict deferred to detector")
