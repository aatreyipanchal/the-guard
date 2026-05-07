"""
OpenAI provider — gpt-4o-mini for cost efficiency during evals.
Pricing (2025): $0.15 / 1M input tokens, $0.60 / 1M output tokens.

Deliberate model choice: gpt-4o-mini over gpt-4o saves ~97% cost.
Used for: deal_copy + credit_narrative (quality-sensitive tasks).
"""
import os
import time
from openai import OpenAI
from .base import BaseProvider, ProviderResponse

INPUT_COST_PER_TOKEN  = 0.15  / 1_000_000
OUTPUT_COST_PER_TOKEN = 0.60  / 1_000_000
MAX_RETRIES = 3


class OpenAIProvider(BaseProvider):
    name  = "openai"
    model = "gpt-4o-mini"

    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY not set in environment.")
        self.client = OpenAI(api_key=api_key)

    def call(self, prompt: str, test_id: str) -> ProviderResponse:
        from errors import classify_provider_exception, APIRateLimitError, APITimeoutError, APIServerError
        last_err = None
        for attempt in range(MAX_RETRIES):
            try:
                raw_response, latency_ms = self._timed_call(
                    self.client.chat.completions.create,
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=512,
                )
                usage   = raw_response.usage
                content = raw_response.choices[0].message.content or ""
                cost = (usage.prompt_tokens * INPUT_COST_PER_TOKEN +
                        usage.completion_tokens * OUTPUT_COST_PER_TOKEN)
                return ProviderResponse(
                    provider=self.name, model=self.model, test_id=test_id,
                    output=content.strip(), latency_ms=latency_ms,
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                    total_tokens=usage.total_tokens, cost_usd=cost,
                    raw=raw_response.model_dump(),
                )
            except Exception as e:
                typed = classify_provider_exception(e, provider=self.name, test_id=test_id)
                last_err = typed
                if isinstance(typed, (APIRateLimitError, APITimeoutError, APIServerError)):
                    backoff = (2 ** attempt) * 2
                    time.sleep(backoff)
                    continue
                break  # non-retryable

        return ProviderResponse(
            provider=self.name, model=self.model, test_id=test_id,
            output="", latency_ms=0.0,
            prompt_tokens=0, completion_tokens=0, total_tokens=0,
            cost_usd=0.0, error=f"{type(last_err).__name__}: {last_err}",
        )
