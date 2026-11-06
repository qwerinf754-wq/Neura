"""
Neura Generic Registry
"""


class RegistryError(Exception):
    """Raised when a registry operation fails."""


class Registry:
    """
    Generic named-object registry.
    """

    def __init__(self):
        self._items = {}

    def register(self, item, name=None):
        if name is None:
            name = getattr(item, "name", None)

        if not name:
            raise RegistryError(
                "Registered item must have a name."
            )

        if name in self._items:
            raise RegistryError(
                f"Item already registered: {name}"
            )

        self._items[name] = item
        return item

    def unregister(self, name):
        if name not in self._items:
            raise RegistryError(
                f"Item not registered: {name}"
            )

        return self._items.pop(name)

    def get(self, name):
        return self._items.get(name)

    def require(self, name):
        item = self.get(name)

        if item is None:
            raise RegistryError(
                f"Item not registered: {name}"
            )

        return item

    def has(self, name):
        return name in self._items

    def list(self):
        return list(self._items.values())

    def names(self):
        return list(self._items.keys())

    def clear(self):
        self._items.clear()

    def __len__(self):
        return len(self._items)
