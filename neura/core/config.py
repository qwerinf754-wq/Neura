"""
neura.core.config

Configuration system for the Neura AI Platform.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelConfig:
    """Neural network architecture configuration."""

    name: str = "Neura"
    vocab_size: int = 32768
    context_length: int = 4096
    embedding_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    dropout: float = 0.0


@dataclass
class TrainingConfig:
    """Training process configuration."""

    batch_size: int = 8
    epochs: int = 1
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 100


@dataclass
class RuntimeConfig:
    """Runtime environment configuration."""

    device: str = "cpu"
    precision: str = "fp32"
    workers: int = 1
    seed: int = 42


@dataclass
class NeuraConfig:
    """Main configuration container for Neura."""

    model: ModelConfig = field(
        default_factory=ModelConfig
    )

    training: TrainingConfig = field(
        default_factory=TrainingConfig
    )

    runtime: RuntimeConfig = field(
        default_factory=RuntimeConfig
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


def create_default_config() -> NeuraConfig:
    """Create the default Neura configuration."""

    return NeuraConfig()


def validate_config(config: NeuraConfig) -> bool:
    """Validate Neura configuration values."""

    if config.model.vocab_size <= 0:
        return False

    if config.model.context_length <= 0:
        return False

    if config.model.embedding_dim <= 0:
        return False

    if config.model.num_layers <= 0:
        return False

    if config.model.num_heads <= 0:
        return False

    if config.training.batch_size <= 0:
        return False

    if config.training.learning_rate <= 0:
        return False

    if config.runtime.workers <= 0:
        return False

    return True


if __name__ == "__main__":
    config = create_default_config()

    print("Neura Config")
    print("Model:", config.model.name)
    print("Context:", config.model.context_length)
    print("Layers:", config.model.num_layers)
    print("Embedding:", config.model.embedding_dim)
    print("Valid:", validate_config(config))
