"""
Neura Memory Message
"""

from dataclasses import dataclass
from datetime import datetime


@dataclass(slots=True)
class Message:

    role: str

    content: str

    timestamp: datetime

    @classmethod
    def create(cls, role, content):

        return cls(
            role=role,
            content=content,
            timestamp=datetime.utcnow(),
        )
