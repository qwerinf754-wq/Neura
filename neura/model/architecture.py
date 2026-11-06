"""
Neura Model Architecture
"""

from dataclasses import dataclass


@dataclass(slots=True)
class Architecture:

    vocab_size: int

    hidden_size: int

    layers: int

    heads: int

    context_length: int


    def summary(self):

        return {

            "vocab_size": self.vocab_size,

            "hidden_size": self.hidden_size,

            "layers": self.layers,

            "heads": self.heads,

            "context_length": self.context_length,

        }
