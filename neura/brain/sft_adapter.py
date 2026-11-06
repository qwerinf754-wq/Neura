from __future__ import annotations

from typing import Any, Protocol

from .orchestrator import NeuraOrchestrator


class SFTGenerator(Protocol):
    """
    Minimal contract expected from the existing SFT runtime.
    """

    def generate(self, text: str, **kwargs: Any) -> str:
        ...


class NeuraSFTAdapter:
    """
    Thin adapter between NeuraOrchestrator and the existing SFT runtime.

    This adapter does not:
      - modify NeuraTorchModel
      - modify tokenizer
      - modify SFT training code
      - modify checkpoint format
      - load or save checkpoints
    """

    def __init__(self, runtime: SFTGenerator):
        if runtime is None:
            raise ValueError("runtime is required")

        generate = getattr(runtime, "generate", None)

        if not callable(generate):
            raise TypeError(
                "Existing SFT runtime must expose callable generate(text, ...)"
            )

        self.runtime = runtime

    def generate(
        self,
        *,
        text: str,
        context: dict[str, Any] | None = None,
        route: str = "chat",
        **kwargs: Any,
    ) -> str:
        context = context or {}

        result = self.runtime.generate(
            text=text,
            context=context,
            route=route,
            **kwargs,
        )

        if result is None:
            raise RuntimeError("SFT runtime returned None")

        return str(result)

    def attach(self, orchestrator: NeuraOrchestrator) -> NeuraOrchestrator:
        if not isinstance(orchestrator, NeuraOrchestrator):
            raise TypeError(
                "orchestrator must be a NeuraOrchestrator instance"
            )

        orchestrator.generator = self.generate
        return orchestrator
