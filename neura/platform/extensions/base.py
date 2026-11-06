"""
Neura Platform Extension SDK
Base extension API.
"""

from abc import ABC, abstractmethod


class NeuraExtension(ABC):
    """
    Base class for all Neura platform extensions.

    Extensions are reusable components that can extend
    the Neura runtime without modifying Neura Core.
    """

    name = "unnamed-extension"
    version = "0.0.0"
    description = ""

    @abstractmethod
    def register(self, runtime):
        """
        Register the extension with a Neura runtime.
        """
        raise NotImplementedError

    def unregister(self, runtime):
        """
        Optional cleanup hook.
        """
        return None
