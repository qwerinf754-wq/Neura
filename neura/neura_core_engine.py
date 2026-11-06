from typing import Any

from .brain import NeuraOrchestrator


class NeuraCoreEngine:
    """
    Public entry point for the unified Neura pipeline.

    generator must be a callable compatible with:
        generator(text=..., context=..., route=...) -> str
    """

    def __init__(self, generator=None, capabilities: Any = None):
        self.orchestrator = NeuraOrchestrator(
            generator=generator,
            capabilities=capabilities,
        )

    def handle(
        self,
        text: str,
        conversation_id: str = "default",
        **kwargs: Any,
    ) -> dict[str, Any]:
        return self.orchestrator.handle(
            text=text,
            conversation_id=conversation_id,
            **kwargs,
        ).as_dict()


def create_engine(generator=None, capabilities=None) -> NeuraCoreEngine:
    return NeuraCoreEngine(
        generator=generator,
        capabilities=capabilities,
    )
