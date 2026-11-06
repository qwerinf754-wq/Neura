from dataclasses import dataclass, field
from typing import Any


@dataclass
class ConversationContext:
    conversation_id: str = "default"
    messages: list[dict[str, str]] = field(default_factory=list)
    memory: dict[str, Any] = field(default_factory=dict)
    retrieved: list[Any] = field(default_factory=list)

    def add_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def add_assistant(self, text: str) -> None:
        self.messages.append({"role": "assistant", "content": text})

    def recent(self, limit: int = 12) -> list[dict[str, str]]:
        return self.messages[-limit:]
