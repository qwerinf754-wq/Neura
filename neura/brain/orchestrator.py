from collections.abc import Callable
from typing import Any

from .context import ConversationContext
from .response import NeuraResponse
from .router import NeuraRouter


class NeuraOrchestrator:
    """
    Unified request layer for Neura.

    The language model remains the generation core.
    The orchestrator is responsible for routing, context and capability
    composition. Generation is injected as a callable so this layer does
    not depend on one particular checkpoint/runtime implementation.
    """

    def __init__(
        self,
        generator: Callable[..., str] | None = None,
        router: NeuraRouter | None = None,
        capabilities: Any | None = None,
    ):
        self.generator = generator
        self.router = router or NeuraRouter()
        self.capabilities = capabilities
        self.contexts: dict[str, ConversationContext] = {}

    def get_context(self, conversation_id: str) -> ConversationContext:
        if conversation_id not in self.contexts:
            self.contexts[conversation_id] = ConversationContext(
                conversation_id=conversation_id
            )
        return self.contexts[conversation_id]

    def handle(
        self,
        text: str,
        conversation_id: str = "default",
        **kwargs: Any,
    ) -> NeuraResponse:
        text = str(text).strip()

        if not text:
            return NeuraResponse(
                text="أرسل لي سؤالك أو طلبك.",
                route="chat",
            )

        context = self.get_context(conversation_id)
        decision = self.router.route(text)

        context.add_user(text)

        prompt_context = {
            "route": decision.name,
            "reason": decision.reason,
            "history": context.recent(),
            "memory": context.memory,
            "retrieved": context.retrieved,
            **kwargs,
        }

        if self.generator is None:
            answer = self._fallback(text, decision.name)
        else:
            answer = self.generator(
                text=text,
                context=prompt_context,
                route=decision.name,
            )
            answer = str(answer)

        context.add_assistant(answer)

        return NeuraResponse(
            text=answer,
            route=decision.name,
            metadata={
                "conversation_id": conversation_id,
                "router_reason": decision.reason,
            },
        )

    @staticmethod
    def _fallback(text: str, route: str) -> str:
        return (
            f"تم توجيه الطلب إلى مسار Neura: {route}. "
            "محرك التوليد غير موصول بعد بطبقة Orchestrator."
        )
