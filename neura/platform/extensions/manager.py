"""
Neura Platform Extension Manager
"""

from .base import NeuraExtension


class ExtensionManager:
    """
    Loads and manages Neura extensions.
    """

    def __init__(self, runtime):
        self.runtime = runtime
        self.loaded = {}

    def load(self, extension):
        if not isinstance(
            extension,
            NeuraExtension
        ):
            raise TypeError(
                "Extension must inherit from "
                "NeuraExtension."
            )

        if extension.name in self.loaded:
            raise ValueError(
                f"Extension already loaded: "
                f"{extension.name}"
            )

        extension.register(self.runtime)

        self.loaded[extension.name] = extension

        return extension

    def unload(self, name):
        extension = self.loaded.get(name)

        if extension is None:
            raise KeyError(
                f"Extension not loaded: {name}"
            )

        extension.unregister(
            self.runtime
        )

        del self.loaded[name]

        return extension

    def get(self, name):
        return self.loaded.get(name)

    def list(self):
        return list(
            self.loaded.values()
        )

    def names(self):
        return list(
            self.loaded.keys()
        )
