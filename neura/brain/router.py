from dataclasses import dataclass


@dataclass(frozen=True)
class RouteDecision:
    name: str
    reason: str


class NeuraRouter:
    """
    Lightweight deterministic first-stage router.

    It deliberately does not pretend to be an intelligent classifier.
    A future trained router can replace this class without changing the
    orchestrator interface.
    """

    CODING = (
        "python", "javascript", "typescript", "java", "c++", "c#", "rust",
        "bug", "error", "exception", "traceback", "code", "كود",
        "برمجة", "خطأ برمجي", "بايثون",
    )

    REASONING = (
        "why", "prove", "calculate", "solve", "reason", "explain why",
        "لماذا", "احسب", "حل", "استنتج", "برهن", "فسر",
    )

    DOCUMENT = (
        "file", "document", "pdf", "ملف", "وثيقة", "مستند",
        "لخّص الملف", "اقرأ الملف",
    )

    def route(self, text: str) -> RouteDecision:
        value = text.strip().lower()

        if any(term in value for term in self.CODING):
            return RouteDecision("coding", "coding signal")

        if any(term in value for term in self.REASONING):
            return RouteDecision("reasoning", "reasoning signal")

        if any(term in value for term in self.DOCUMENT):
            return RouteDecision("documents", "document signal")

        return RouteDecision("chat", "default conversational route")
