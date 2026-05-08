import os
import time
import random

from .base import BaseProvider, ProviderResponse

INPUT_COST_PER_TOKEN = 0.05 / 1_000_000
OUTPUT_COST_PER_TOKEN = 0.08 / 1_000_000


class GroqProvider(BaseProvider):

    name = "groq"
    model = "llama-3.1-8b-instant"

    def __init__(self, model: str = "llama-3.1-8b-instant"):

        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY not set."
            )

        self.model = model
        self._api_key = api_key

        try:
            from groq import Groq

            self._client = Groq(
                api_key=api_key,
                max_retries=0,
                timeout=30,
            )

        except ImportError:
            raise EnvironmentError(
                "groq package not installed."
            )

    def call(
        self,
        prompt: str,
        test_id: str,
    ) -> ProviderResponse:

        from errors import classify_provider_exception

        MAX_RETRIES = 3

        for attempt in range(MAX_RETRIES):

            try:

                raw_response, latency_ms = self._timed_call(
                    self._client.chat.completions.create,

                    model=self.model,

                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],

                    temperature=0,
                    max_tokens=80,
                )

                usage = raw_response.usage

                content = (
                    raw_response.choices[0]
                    .message.content
                    or ""
                )

                cost = (
                    usage.prompt_tokens
                    * INPUT_COST_PER_TOKEN
                    +
                    usage.completion_tokens
                    * OUTPUT_COST_PER_TOKEN
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

                err_str = str(e).lower()

                if "429" in err_str or "rate limit" in err_str:

                    sleep_time = (
                        1.5 ** attempt
                        +
                        random.uniform(0.2, 0.8)
                    )

                    time.sleep(sleep_time)
                    continue

                typed_err = classify_provider_exception(
                    e,
                    provider=self.name,
                    test_id=test_id,
                )

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
            error="RateLimitError: Max retries exceeded",
        )
