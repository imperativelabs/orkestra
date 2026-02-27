"""Abstract base class for provider backends."""

from abc import ABC, abstractmethod
from typing import Iterator


class ProviderBackend(ABC):
    """Base class that all provider backends must implement."""

    name: str

    @abstractmethod
    def call(
        self,
        model: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> dict:
        """Call the provider's API and return a response.

        Args:
            model: Model name to use.
            prompt: Input prompt text.
            max_tokens: Maximum output tokens.
            temperature: Sampling temperature.

        Returns:
            Dict with keys: "text", "input_tokens", "output_tokens".
        """

    @abstractmethod
    def stream(
        self,
        model: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> Iterator[str]:
        """Stream text chunks from the provider's API.

        Args:
            model: Model name to use.
            prompt: Input prompt text.
            max_tokens: Maximum output tokens.
            temperature: Sampling temperature.

        Yields:
            Text chunks as they arrive.
        """
