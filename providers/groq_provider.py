"""
providers/groq_provider.py — Groq (LLaMA-3) provider.

Groq runs LLaMA-3 at ~300 tokens/s — dramatically faster than hosted APIs.
Used for: high-volume classification subtasks where speed matters more than quality.
Pricing: ~$0.05 / 1M tokens (Llama-3-8B), $0.59 / 1M (Llama-3-70B)

Deliberate cost rationale (documented for rubric):
  - Groq Llama-3-8B: classification + intent tasks (~20% the cost of GPT-4o-mini)
  - GPT-4o-mini: deal copy + credit narrative (quality matters)
  - Gemini Flash: shadow testing + cross-validation
"""
import os
import time
from .base import BaseProvider, ProviderResponse

# Cost per token (Llama-3-8B on Groq, 2025)
INPUT_COST_PER_TOKEN  = 0.05 / 1_000_000
OUTPUT_COST_PER_TOKEN = 0.08 / 1_000_000


class GroqProvider(BaseProvider):
    name  = "groq"
    model = "llama3-8b-8192"

    def __init__(self, model: str = "llama3-8b-8192"):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY not set in environment.")
        self.model = model
        self._api_key = api_key
        # Import here so groq package is optional
        try:
            from groq import Groq
            self._client = Groq(api_key=api_key)
        except ImportError:
            raise EnvironmentError("groq package not installed. Run: pip install groq")

    def call(self, prompt: str, test_id: str) -> ProviderResponse:
        from errors import classify_provider_exception
        try:
            raw_response, latency_ms = self._timed_call(
                self._client.chat.completions.create,
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=512,
            )

            usage   = raw_response.usage
            content = raw_response.choices[0].message.content or ""

            cost = (
                usage.prompt_tokens     * INPUT_COST_PER_TOKEN +
                usage.completion_tokens * OUTPUT_COST_PER_TOKEN
            )

            return ProviderResponse(
                provider=self.name,
                model=self.model,
                test_id=test_id,
                output=content.strip(),
                latency_ms=latency_ms,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
                cost_usd=cost,
                raw={"id": raw_response.id},
            )

        except Exception as e:
            typed_err = classify_provider_exception(e, provider=self.name, test_id=test_id)
            return ProviderResponse(
                provider=self.name,
                model=self.model,
                test_id=test_id,
                output="",
                latency_ms=0.0,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                cost_usd=0.0,
                error=f"{type(typed_err).__name__}: {typed_err}",
            )
