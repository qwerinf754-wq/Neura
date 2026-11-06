"""Registry for optional Neura integrations."""

from __future__ import annotations

from dataclasses import dataclass

from .capabilities import CapabilityInfo, list_capabilities


@dataclass(frozen=True)
class IntegrationRegistry:
    """Read-only registry of optional third-party capabilities."""

    capabilities: tuple[CapabilityInfo, ...]

    @classmethod
    def discover(cls) -> "IntegrationRegistry":
        """Discover currently available integrations."""
        return cls(capabilities=list_capabilities())

    def available(self) -> tuple[CapabilityInfo, ...]:
        """Return installed/available capabilities."""
        return tuple(
            capability
            for capability in self.capabilities
            if capability.available
        )

    def unavailable(self) -> tuple[CapabilityInfo, ...]:
        """Return capabilities that are not installed."""
        return tuple(
            capability
            for capability in self.capabilities
            if not capability.available
        )

    def required_missing(self) -> tuple[CapabilityInfo, ...]:
        """Return missing capabilities marked as required."""
        return tuple(
            capability
            for capability in self.capabilities
            if capability.required and not capability.available
        )

    def has(self, name: str) -> bool:
        """Return whether a named capability is available."""
        return any(
            capability.name == name and capability.available
            for capability in self.capabilities
        )
