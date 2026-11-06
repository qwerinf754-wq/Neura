"""
Neura Chat Service Compatibility Layer

The canonical ChatService implementation lives in
server.services.chat.

This module remains available for backward-compatible imports.
"""

from server.services.chat import (
    ChatService,
    ChatServiceError,
)

__all__ = [
    "ChatService",
    "ChatServiceError",
]
