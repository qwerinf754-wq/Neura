"""
Neura Module Registry
"""

from typing import Dict

from .module import Module


class ModuleRegistry:

    def __init__(self):

        self._modules: Dict[str, Module] = {}


    def register(self, module: Module):

        self._modules[module.name] = module


    def get(self, name):

        return self._modules.get(name)


    def all(self):

        return list(self._modules.values())


    def count(self):

        return len(self._modules)


REGISTRY = ModuleRegistry()
