"""Tests for Neura's optional integration registry."""

from neura.integrations import (
    IntegrationRegistry,
    get_capability,
    list_capabilities,
)


def test_registry_discovers_capabilities() -> None:
    registry = IntegrationRegistry.discover()

    assert registry.capabilities
    assert registry.has("pytorch")


def test_capabilities_have_unique_names() -> None:
    capabilities = list_capabilities()
    names = [item.name for item in capabilities]

    assert len(names) == len(set(names))


def test_pytorch_is_required() -> None:
    capability = get_capability("pytorch")

    assert capability is not None
    assert capability.required is True
    assert capability.available is True


def test_missing_optional_packages_do_not_break_registry() -> None:
    registry = IntegrationRegistry.discover()

    assert registry.required_missing() == ()
    assert registry.available()
