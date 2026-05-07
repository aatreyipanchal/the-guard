"""
Google Gemini provider — gemini-1.5-flash for cost efficiency.
Pricing (2025): $0.075 / 1M input, $0.30 / 1M output.

Deliberate model choice: Flash over Pro. Used for: shadow testing + cross-validation.
Cheapest hosted frontier model — validates that regression isn't provider-specific.
"""
import os
import time
import google.generativeai as genai
from .base import BaseProvider, ProviderResponse

INPUT_COST_PER_TOKEN  = 0.075 / 1_000_000
OUTPUT_COST_PER_TOKEN = 0.30  / 1_000_000
MAX_RETRIES = 3


class GeminiProvider(BaseProvider):
    name  = "gemini"
    model = "gemini-1.5-flash"

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY not set in environment.")
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(
            model_name=self.model,
            generation_config=genai.GenerationConfig(temperature=0, max_output_tokens=512),
        )

    def call(self, prompt: str, test_id: str) -> ProviderResponse:
        from errors import classify_provider_exception, APIRateLimitError, APITimeoutError, APIServerError
        last_err = None
        for attempt in range(MAX_RETRIES):
            try:
                raw_response, latency_ms = self._timed_call(self._model.generate_content, prompt)
                content = raw_response.text or ""
                usage = raw_response.usage_metadata
                pt = usage.prompt_token_count if usage else 0
                ct = usage.candidates_token_count if usage else 0
                cost = pt * INPUT_COST_PER_TOKEN + ct * OUTPUT_COST_PER_TOKEN
                return ProviderResponse(
                    provider=self.name, model=self.model, test_id=test_id,
                    output=content.strip(), latency_ms=latency_ms,
                    prompt_tokens=pt, completion_tokens=ct, total_tokens=pt + ct,
                    cost_usd=cost,
                    raw={"candidates": [c.to_dict() for c in raw_response.candidates]},
                )
            except Exception as e:
                typed = classify_provider_exception(e, provider=self.name, test_id=test_id)
                last_err = typed
                if isinstance(typed, (APIRateLimitError, APITimeoutError, APIServerError)):
                    time.sleep((2 ** attempt) * 2)
                    continue
                break

        return ProviderResponse(
            provider=self.name, model=self.model, test_id=test_id,
            output="", latency_ms=0.0,
            prompt_tokens=0, completion_tokens=0, total_tokens=0,
            cost_usd=0.0, error=f"{type(last_err).__name__}: {last_err}",
        )
