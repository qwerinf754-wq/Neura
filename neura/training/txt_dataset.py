"""
Neura TXT Dataset
=================

Loads TXT files through the real corpus pipeline and converts
them into causal language-model training windows.

This module is CPU/Android friendly and uses only the existing
Neura tokenizer and PyTorch.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from neura.tokenizer.tokenizer import TOKENIZER
from neura.training.real_corpus import RealCorpusPipeline


@dataclass(slots=True)
class TXTTrainingData:
    train_tokens: list[int]
    validation_tokens: list[int]
    train_documents: int
    validation_documents: int
    source_files: int


class TXTCorpusDataset:
    """
    TXT corpus -> tokenizer -> causal LM token streams.

    The complete corpus is tokenized once. Window construction is
    performed separately so sequence length and stride remain
    configurable.
    """

    def __init__(
        self,
        data_dir: str | Path,
        minimum_length: int = 20,
        validation_ratio: float = 0.05,
        seed: int = 42,
    ):
        self.data_dir = Path(data_dir)

        self.pipeline = RealCorpusPipeline(
            minimum_length=minimum_length,
            validation_ratio=validation_ratio,
            seed=seed,
        )

        self.data: TXTTrainingData | None = None

    def load(self) -> TXTTrainingData:
        paths = self.pipeline.discover(self.data_dir)

        if not paths:
            raise RuntimeError(
                f"No TXT files found in {self.data_dir}"
            )

        split = self.pipeline.run(self.data_dir)

        train_tokens: list[int] = []
        validation_tokens: list[int] = []

        for text in split.train:
            ids = TOKENIZER.encode(text)

            if ids:
                train_tokens.extend(int(x) for x in ids)

        for text in split.validation:
            ids = TOKENIZER.encode(text)

            if ids:
                validation_tokens.extend(int(x) for x in ids)

        if not train_tokens:
            raise RuntimeError(
                "Training corpus produced zero tokens."
            )

        self.data = TXTTrainingData(
            train_tokens=train_tokens,
            validation_tokens=validation_tokens,
            train_documents=len(split.train),
            validation_documents=len(split.validation),
            source_files=len(paths),
        )

        return self.data


def build_windows(
    token_ids: list[int],
    sequence_length: int,
    stride: int | None = None,
):
    """
    Build causal-LM input/target tensors.

    input  = tokens[i : i+L]
    target = tokens[i+1 : i+L+1]
    """

    if sequence_length <= 0:
        raise ValueError(
            "sequence_length must be positive"
        )

    if stride is None:
        stride = sequence_length

    if stride <= 0:
        raise ValueError(
            "stride must be positive"
        )

    inputs = []
    targets = []

    maximum_start = (
        len(token_ids)
        - sequence_length
    )

    for start in range(
        0,
        maximum_start,
        stride,
    ):
        chunk = token_ids[
            start:
            start + sequence_length + 1
        ]

        if len(chunk) != sequence_length + 1:
            continue

        inputs.append(
            chunk[:-1]
        )

        targets.append(
            chunk[1:]
        )

    return inputs, targets


def tensors_from_tokens(
    token_ids: list[int],
    sequence_length: int,
    stride: int | None = None,
    device: str | torch.device = "cpu",
):
    inputs, targets = build_windows(
        token_ids,
        sequence_length,
        stride,
    )

    if not inputs:
        raise RuntimeError(
            "Unable to construct any causal-LM windows."
        )

    x = torch.tensor(
        inputs,
        dtype=torch.long,
        device=device,
    )

    y = torch.tensor(
        targets,
        dtype=torch.long,
        device=device,
    )

    return x, y


__all__ = [
    "TXTTrainingData",
    "TXTCorpusDataset",
    "build_windows",
    "tensors_from_tokens",
]
