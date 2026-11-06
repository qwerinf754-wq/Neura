"""
Neura Plugin Manager
"""

from .base import NeuraPlugin


class PluginManager:
    """
    Loads and manages Neura plugins.
    """

    def __init__(self, runtime):
        self.runtime = runtime
        self.loaded = {}

    def load(self, plugin):
        if not isinstance(plugin, NeuraPlugin):
            raise TypeError(
                "Plugin must inherit from NeuraPlugin."
            )

        if plugin.name in self.loaded:
            raise ValueError(
                f"Plugin already loaded: {plugin.name}"
            )

        plugin.initialize(self.runtime)
        self.loaded[plugin.name] = plugin

        self.runtime.registry.plugins.register(
            plugin
        )

        return plugin

    def unload(self, name):
        plugin = self.loaded.get(name)

        if plugin is None:
            raise KeyError(
                f"Plugin not loaded: {name}"
            )

        plugin.shutdown(self.runtime)

        self.runtime.registry.plugins.unregister(
            name
        )

        del self.loaded[name]

        return plugin

    def get(self, name):
        return self.loaded.get(name)

    def list(self):
        return list(self.loaded.values())

    def names(self):
        return list(self.loaded.keys())
