"""
Neura Model Configuration
"""

from dataclasses import dataclass


@dataclass(slots=True)
class ModelConfig:
    """
    Base configuration for every Neura model.
    """

    name: str

    layers: int
    hidden_size: int
    heads: int

    vocab_size: int
    context_length: int

    dropout: float = 0.0

    def validate(self):

        if self.layers <= 0:
            raise ValueError("layers must be positive")

        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")

        if self.heads <= 0:
            raise ValueError("heads must be positive")

        if self.hidden_size % self.heads != 0:
            raise ValueError(
                "hidden_size must be divisible by heads"
            )

        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")

        if self.context_length <= 0:
            raise ValueError(
                "context_length must be positive"
            )

        return True


def NeuraTiny():

    return ModelConfig(
        name="Neura-Tiny",
        layers=4,
        hidden_size=256,
        heads=8,
        vocab_size=32768,
        context_length=1024,
        dropout=0.1
    )


DEFAULT_MODEL_CONFIG = NeuraTiny()


def NeuraSmall():

    return ModelConfig(
        name="Neura-Small",
        layers=12,
        hidden_size=768,
        heads=12,
        vocab_size=32768,
        context_length=4096,
        dropout=0.1
    )


def NeuraBase():

    return ModelConfig(
        name="Neura-Base",
        layers=24,
        hidden_size=1024,
        heads=16,
        vocab_size=65536,
        context_length=8192,
        dropout=0.1
    )


def NeuraLarge():

    return ModelConfig(
        name="Neura-Large",
        layers=48,
        hidden_size=4096,
        heads=32,
        vocab_size=131072,
        context_length=32768,
        dropout=0.1
    )


def NeuraXL():

    return ModelConfig(
        name="Neura-XL",
        layers=64,
        hidden_size=8192,
        heads=64,
        vocab_size=262144,
        context_length=65536,
        dropout=0.1
    )


def get_model_config(name):

    models = {
        "tiny": NeuraTiny,
        "small": NeuraSmall,
        "base": NeuraBase,
        "large": NeuraLarge,
        "xl": NeuraXL,
    }

    key = name.lower()

    if key not in models:
        raise ValueError(
            f"Unknown Neura model: {name}"
        )

    return models[key]()
