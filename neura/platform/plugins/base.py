"""
Neura Platform Plugin API
"""

from abc import ABC, abstractmethod


class NeuraPlugin(ABC):
    """
    Base class for all Neura plugins.
    """

    name = "unnamed"
    version = "0.0.0"
    description = ""

    @abstractmethod
    def initialize(self, runtime):
        """
        Initialize the plugin against a Neura runtime.
        """
        raise NotImplementedError

    def shutdown(self, runtime):
        """
        Optional cleanup hook.
        """
        return None
