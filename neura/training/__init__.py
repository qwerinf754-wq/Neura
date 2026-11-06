"""
Neura Training Package
"""

from .dataset import CausalLMDataset

# Backward-compatible alias.
TextDataset = CausalLMDataset

__all__ = [
    "CausalLMDataset",
    "TextDataset",
]
