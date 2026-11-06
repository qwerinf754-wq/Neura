"""
Neura Extension Registry
"""

from .registry import Registry


class ExtensionRegistry:
    """
    Central registry for Neura platform extensions.
    """

    def __init__(self):
        self.plugins = Registry()
        self.tools = Registry()
        self.agents = Registry()
        self.models = Registry()
        self.memories = Registry()
        self.rag = Registry()

    def summary(self):
        return {
            "plugins": len(self.plugins),
            "tools": len(self.tools),
            "agents": len(self.agents),
            "models": len(self.models),
            "memories": len(self.memories),
            "rag": len(self.rag),
        }
