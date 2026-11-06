"""
Neura Main Model
"""

from .transformer import Transformer
from .lm_head import LMHead
from .embeddings import Embedding
from .config import (
    ModelConfig,
    DEFAULT_MODEL_CONFIG,
)
from neura.inference.kv_cache import KVCache


class NeuraModel:

    def __init__(
        self,
        layers=None,
        hidden_size=None,
        heads=None,
        vocab_size=None,
        config=None
    ):

        # If no explicit configuration is supplied, use the
        # project's validated default model configuration.
        if config is None and all(
            value is None
            for value in (
                layers,
                hidden_size,
                heads,
                vocab_size,
            )
        ):
            config = DEFAULT_MODEL_CONFIG

        if config is not None:

            if not isinstance(
                config,
                ModelConfig
            ):
                raise TypeError(
                    "config must be ModelConfig"
                )

            layers = config.layers
            hidden_size = config.hidden_size
            heads = config.heads
            vocab_size = config.vocab_size

        self.config = config
        self.context_length = (
            config.context_length
            if config is not None
            else None
        )


        self.embedding = Embedding(
            vocab_size,
            hidden_size
        )

        if config is not None:
            self.transformer = Transformer(
                config=config
            )
        else:
            self.transformer = Transformer(
                layers,
                hidden_size,
                heads
            )

        self.kv_cache = KVCache()

        self.transformer.set_cache(
            self.kv_cache
        )

        self.lm_head = LMHead(
            hidden_size,
            vocab_size
        )

        self._last_hidden = None

    def forward(
        self,
        sequence,
        use_cache=False,
        cache_position=None,
    ):

        embedded = self.embedding.forward(
            sequence
        )

        hidden = self.transformer.forward(
            embedded,
            use_cache=use_cache,
            cache_position=cache_position,
        )

        self._last_hidden_sequence = hidden
        self._last_hidden = hidden[-1]

        return self.lm_head.forward_sequence(
            self._last_hidden_sequence,
            sequence
        )

    def clear_cache(self):

        if hasattr(self, "kv_cache"):
            self.kv_cache.clear()

    def cache_size(self):

        if not hasattr(self, "kv_cache"):
            return 0

        return self.kv_cache.total_size()

    def backward(self, gradient):

        # Sequence gradient path
        if (
            isinstance(gradient, list)
            and gradient
            and isinstance(gradient[0], list)
        ):

            sequence_gradient = []

            for grad, hidden in zip(
                gradient,
                self._last_hidden_sequence
            ):

                hidden_grad = self.lm_head.backward(
                    grad,
                    hidden
                )

                if hidden_grad is None:
                    hidden_grad = [
                        0.0
                        for _ in hidden
                    ]

                sequence_gradient.append(
                    hidden_grad
                )


            transformer_gradient = (
                self.transformer.backward(
                    sequence_gradient
                )
            )

            self.embedding.backward(
                transformer_gradient
            )

            return transformer_gradient


        # Single gradient compatibility
        hidden_gradient = self.lm_head.backward(
            gradient,
            self._last_hidden
        )

        if hidden_gradient is None:
            hidden_gradient = [
                0.0
                for _ in self._last_hidden
            ]

        transformer_gradient = [
            list(hidden_gradient)
            for _ in self._last_hidden_sequence
        ]

        transformer_gradient = self.transformer.backward(
            transformer_gradient
        )

        self.embedding.backward(
            transformer_gradient
        )

        return transformer_gradient
