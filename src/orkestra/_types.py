from dataclasses import dataclass


@dataclass
class Response:
    """Response from Provider.chat() or MultiProvider.chat()."""

    text: str
    model: str
    provider: str
    cost: float
    input_tokens: int
    output_tokens: int
    input_cost: float
    output_cost: float
    savings: float
    savings_percent: float
    base_model: str
    base_cost: float

    def __str__(self) -> str:
        return (
            f"Response(provider={self.provider}, model={self.model}, "
            f"cost=${self.cost:.6f}, savings={self.savings_percent:.1f}%)\n"
            f"{self.text[:200]}{'...' if len(self.text) > 200 else ''}"
        )
