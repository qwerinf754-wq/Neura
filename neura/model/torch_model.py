"""
Neura PyTorch Transformer Backend
"""

from __future__ import annotations

import torch
from torch import nn


class TorchCausalSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, heads: int, dropout: float = 0.0):
        super().__init__()

        if hidden_size % heads != 0:
            raise ValueError(
                "hidden_size must be divisible by heads"
            )

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, x):
        seq_len = x.size(1)

        causal_mask = torch.triu(
            torch.ones(
                seq_len,
                seq_len,
                device=x.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )

        output, _ = self.attention(
            x,
            x,
            x,
            attn_mask=causal_mask,
            need_weights=False,
        )

        return output


class TorchMLP(nn.Module):
    def __init__(self, hidden_size: int, expansion: int = 4):
        super().__init__()

        intermediate = hidden_size * expansion

        self.up = nn.Linear(
            hidden_size,
            intermediate,
        )

        self.down = nn.Linear(
            intermediate,
            hidden_size,
        )

        self.activation = nn.GELU()

    def forward(self, x):
        return self.down(
            self.activation(
                self.up(x)
            )
        )


class TorchTransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(hidden_size)

        self.attention = TorchCausalSelfAttention(
            hidden_size=hidden_size,
            heads=heads,
            dropout=dropout,
        )

        self.norm2 = nn.LayerNorm(hidden_size)

        self.mlp = TorchMLP(
            hidden_size=hidden_size,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(
            self.attention(
                self.norm1(x)
            )
        )

        x = x + self.dropout(
            self.mlp(
                self.norm2(x)
            )
        )

        return x


class NeuraTorchModel(nn.Module):
    """
    Small real causal language model for Neura training.

    Architecture:

        token ids
            ↓
        embedding
            ↓
        Transformer blocks
            ↓
        final norm
            ↓
        LM head
            ↓
        vocabulary logits
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 256,
        layers: int = 4,
        heads: int = 8,
        context_length: int = 1024,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.layers_count = layers
        self.heads = heads
        self.context_length = context_length

        self.token_embedding = nn.Embedding(
            vocab_size,
            hidden_size,
        )

        self.position_embedding = nn.Embedding(
            context_length,
            hidden_size,
        )

        self.blocks = nn.ModuleList(
            [
                TorchTransformerBlock(
                    hidden_size=hidden_size,
                    heads=heads,
                    dropout=dropout,
                )
                for _ in range(layers)
            ]
        )

        self.final_norm = nn.LayerNorm(
            hidden_size
        )

        self.lm_head = nn.Linear(
            hidden_size,
            vocab_size,
            bias=False,
        )

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.normal_(
            self.token_embedding.weight,
            mean=0.0,
            std=0.02,
        )

        nn.init.normal_(
            self.position_embedding.weight,
            mean=0.0,
            std=0.02,
        )

        nn.init.normal_(
            self.lm_head.weight,
            mean=0.0,
            std=0.02,
        )

    def forward(self, input_ids):
        if input_ids.dim() != 2:
            raise ValueError(
                "input_ids must have shape [batch, sequence]"
            )

        batch_size, sequence_length = input_ids.shape

        if sequence_length > self.context_length:
            raise ValueError(
                f"sequence length {sequence_length} exceeds "
                f"context length {self.context_length}"
            )

        positions = torch.arange(
            sequence_length,
            device=input_ids.device,
        )

        token_embeddings = self.token_embedding(
            input_ids
        )

        position_embeddings = self.position_embedding(
            positions
        )

        x = (
            token_embeddings
            + position_embeddings.unsqueeze(0)
        )

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)

        logits = self.lm_head(x)

        return logits


__all__ = [
    "NeuraTorchModel",
]
