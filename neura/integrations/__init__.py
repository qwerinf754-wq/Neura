"""Optional third-party integrations for Neura."""

from .capabilities import (
    CapabilityInfo,
    get_capability,
    list_capabilities,
)
from .registry import IntegrationRegistry

__all__ = [
    "CapabilityInfo",
    "IntegrationRegistry",
    "get_capability",
    "list_capabilities",
]
