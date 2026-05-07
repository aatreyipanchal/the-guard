"""
Base interface for LLM providers.
All providers must implement this to be pluggable in the eval pipeline.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import time


@dataclass
class ProviderResponse:
    provider: str
    model: str
    test_id: str
    output: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    error: Optional[str] = None
    raw: dict = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.error is None


class BaseProvider(ABC):
    name: str
    model: str

    @abstractmethod
    def call(self, prompt: str, test_id: str) -> ProviderResponse:
        """Call the LLM and return a structured response."""
        ...

    def _timed_call(self, fn, *args, **kwargs) -> tuple:
        """Utility: time any function call in ms."""
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        return result, elapsed_ms
