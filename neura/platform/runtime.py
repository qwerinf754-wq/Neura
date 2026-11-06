"""
Neura Platform Runtime
"""

from .registry.extension_registry import (
    ExtensionRegistry
)

from .plugins.manager import (
    PluginManager
)

from .extensions.manager import (
    ExtensionManager
)


class NeuraRuntime:
    """
    Central runtime for the extensible
    Neura platform.
    """

    def __init__(self):
        self.registry = ExtensionRegistry()

        self.plugins = PluginManager(
            self
        )

        self.extensions = ExtensionManager(
            self
        )

    # --------------------------------------------------
    # Plugin API
    # --------------------------------------------------

    def load_plugin(self, plugin):
        return self.plugins.load(
            plugin
        )

    def unload_plugin(self, name):
        return self.plugins.unload(
            name
        )

    # --------------------------------------------------
    # Extension API
    # --------------------------------------------------

    def load_extension(self, extension):
        return self.extensions.load(
            extension
        )

    def unload_extension(self, name):
        return self.extensions.unload(
            name
        )

    # --------------------------------------------------
    # Runtime status
    # --------------------------------------------------

    def status(self):
        return {
            "plugins": self.plugins.names(),
            "extensions": self.extensions.names(),
            "registry": self.registry.summary(),
        }
