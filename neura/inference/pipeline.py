"""
Neura Model Pipeline
"""

from neura.model import (
    Transformer,
    LMHead,
    Embedding,
    DEFAULT_MODEL_CONFIG
)


class ModelPipeline:

    def __init__(self):

        self.embedding = Embedding(
            DEFAULT_MODEL_CONFIG.vocab_size,
            DEFAULT_MODEL_CONFIG.hidden_size
        )

        self.transformer = Transformer(
            layers=12,
            hidden_size=DEFAULT_MODEL_CONFIG.hidden_size,
            heads=DEFAULT_MODEL_CONFIG.heads
        )

        self.last_hidden = None

        self.last_token_ids = []

        self.head = LMHead(
            DEFAULT_MODEL_CONFIG.hidden_size,
            DEFAULT_MODEL_CONFIG.vocab_size
        )


    def forward(self, token_ids):

        embeddings = self.embedding.forward(
            token_ids
        )

        hidden_states = self.transformer.forward(
            embeddings
        )

        if not hidden_states:

            return []

        hidden = hidden_states[-1]

        self.last_hidden = hidden
        self.last_token_ids = list(token_ids)

        self.head.register_tokens(
            token_ids
        )

        return self.head.forward(
            hidden,
            token_ids
        )


    def backward(self, gradients):

        if self.last_hidden is None:
            return

        self.head.backward(
            gradients,
            self.last_hidden
        )


    def summary(self):

        return {
            "embedding": "ready",
            "transformer": self.transformer.summary(),
            "lm_head": "ready"
        }
