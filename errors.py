"""
errors.py — Typed error taxonomy for The Guard eval pipeline.

Every exception the pipeline can throw is a subclass of EvalError.
This lets the agent loop distinguish error types and decide:
  - retry with backoff     → APIRateLimitError, APITimeoutError
  - skip + score=0         → ProviderRefusalError, ContentPolicyError
  - abort task             → BudgetExceededError
  - abort run              → FatalConfigError
"""

from typing import Optional


class EvalError(Exception):
    """Base class for all pipeline errors."""
    def __init__(self, msg: str, test_id: Optional[str] = None, provider: Optional[str] = None):
        super().__init__(msg)
        self.test_id = test_id
        self.provider = provider
    def __repr__(self):
        return f"{self.__class__.__name__}({self.args[0]!r})"


# ── Retryable API errors ─────────────────────────────────────
class APIError(EvalError):
    """Any transient API-level error."""
    pass

class APIRateLimitError(APIError):
    """429 / quota exceeded — back off and retry."""
    def __init__(self, msg: str, retry_after_seconds: float = 60.0, **kw):
        super().__init__(msg, **kw)
        self.retry_after_seconds = retry_after_seconds

class APITimeoutError(APIError):
    """Network timeout — retry with backoff."""
    pass

class APIServerError(APIError):
    """5xx from provider — retry with backoff."""
    def __init__(self, msg: str, status_code: int = 500, **kw):
        super().__init__(msg, **kw)
        self.status_code = status_code


# ── Non-retryable provider errors ────────────────────────────
class ProviderRefusalError(EvalError):
    """Model refused to answer (content policy, safety filter). Score=0, no retry."""
    pass

class ContentPolicyError(ProviderRefusalError):
    """Explicit content-policy block from provider."""
    pass

class JudgeRefusalError(EvalError):
    """LLM-as-judge refused to score the output. Fallback to semantic scorer."""
    pass

class JSONParseError(EvalError):
    """Model returned unparseable JSON. Score by field-availability, not crash."""
    def __init__(self, msg: str, raw_output: str = "", **kw):
        super().__init__(msg, **kw)
        self.raw_output = raw_output


# ── Budget / resource errors ─────────────────────────────────
class BudgetExceededError(EvalError):
    """Cumulative cost / token spend exceeded MAX_COST_USD. Abort run."""
    def __init__(self, msg: str, spent_usd: float = 0.0, limit_usd: float = 0.0, **kw):
        super().__init__(msg, **kw)
        self.spent_usd = spent_usd
        self.limit_usd = limit_usd

class TokenBudgetExceededError(BudgetExceededError):
    """Per-task token limit hit."""
    pass


# ── Configuration / setup errors ─────────────────────────────
class FatalConfigError(EvalError):
    """Missing API key, bad config — cannot proceed at all."""
    pass

class BaselineCorruptError(EvalError):
    """Baseline JSON is corrupt or incompatible with current schema."""
    pass


# ── Scoring errors ───────────────────────────────────────────
class ScoringError(EvalError):
    """Scorer raised an unexpected exception. Score=0, logged, pipeline continues."""
    pass


# ── Error classifier ─────────────────────────────────────────
def classify_provider_exception(exc: Exception, provider: str = "", test_id: str = "") -> EvalError:
    """Convert a raw provider SDK exception into a typed EvalError."""
    msg = str(exc)
    cls_name = type(exc).__name__

    # OpenAI
    if "RateLimitError" in cls_name or "rate_limit" in msg.lower() or "429" in msg:
        return APIRateLimitError(msg, provider=provider, test_id=test_id)
    if "Timeout" in cls_name or "timeout" in msg.lower():
        return APITimeoutError(msg, provider=provider, test_id=test_id)
    if "APIStatusError" in cls_name and ("500" in msg or "502" in msg or "503" in msg):
        return APIServerError(msg, provider=provider, test_id=test_id)
    if "content_policy" in msg.lower() or "content_filter" in msg.lower():
        return ContentPolicyError(msg, provider=provider, test_id=test_id)

    # Generic fallback
    return APIError(msg, provider=provider, test_id=test_id)
