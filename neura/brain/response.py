from dataclasses import dataclass, field
from typing import Any


@dataclass
class NeuraResponse:
    text: str
    route: str = "chat"
    sources: list[Any] = field(default_factory=list)
    tool_calls: list[Any] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "route": self.route,
            "sources": self.sources,
            "tool_calls": self.tool_calls,
            "metadata": self.metadata,
        }
