"""
Neura Real Corpus Pipeline
==========================

Prepares real text corpora for training.

Pipeline:
    files -> documents -> normalize -> filter -> deduplicate
          -> deterministic train/validation split
          -> tokenizer -> training datasets
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import random


@dataclass(slots=True)
class CorpusDocument:
    text: str
    source: str


@dataclass(slots=True)
class CorpusSplit:
    train: list[str]
    validation: list[str]


@dataclass(slots=True)
class CorpusStats:
    discovered: int
    accepted: int
    rejected_empty: int
    rejected_short: int
    duplicates: int
    train_count: int
    validation_count: int


class RealCorpusError(RuntimeError):
    """Base corpus pipeline error."""


class RealCorpusPipeline:

    def __init__(
        self,
        minimum_length: int = 20,
        validation_ratio: float = 0.1,
        seed: int = 42,
    ):
        if minimum_length < 1:
            raise ValueError(
                "minimum_length must be positive"
            )

        if not 0.0 <= validation_ratio < 1.0:
            raise ValueError(
                "validation_ratio must be in [0, 1)"
            )

        self.minimum_length = minimum_length
        self.validation_ratio = validation_ratio
        self.seed = seed

        self.stats = None

    # ---------------------------------------------------------
    # Loading
    # ---------------------------------------------------------

    def discover(self, root):
        root = Path(root)

        if not root.exists():
            raise FileNotFoundError(root)

        if root.is_file():
            return [root]

        return sorted(
            path
            for path in root.rglob("*")
            if path.is_file()
            and path.suffix.lower() in {
                ".txt",
                ".text",
            }
        )

    def load_files(self, paths):
        """
        Load corpus files.

        A corpus file may contain multiple documents separated
        by blank lines. Each non-empty block becomes an independent
        CorpusDocument so the train/validation splitter can operate
        at document level.
        """
        documents = []

        for path in paths:
            try:
                text = path.read_text(
                    encoding="utf-8"
                )
            except UnicodeDecodeError:
                text = path.read_text(
                    encoding="utf-8",
                    errors="ignore"
                )

            parts = text.split("\n\n")

            for part in parts:
                part = part.strip()

                if not part:
                    continue

                documents.append(
                    CorpusDocument(
                        text=part,
                        source=str(path),
                    )
                )

        return documents

    # ---------------------------------------------------------
    # Normalization
    # ---------------------------------------------------------

    @staticmethod
    def normalize(text):
        """
        Normalize a document into a single clean text sequence.

        Newlines and repeated whitespace are collapsed into spaces.
        """
        text = text.replace("\r\n", "\n")
        text = text.replace("\r", "\n")

        return " ".join(text.split())

    # ---------------------------------------------------------
    # Deduplication
    # ---------------------------------------------------------

    @staticmethod
    def _fingerprint(text):
        return hashlib.sha256(
            text.encode("utf-8")
        ).hexdigest()

    # ---------------------------------------------------------
    # Preparation
    # ---------------------------------------------------------

    def prepare(self, documents):
        discovered = len(documents)

        accepted = []
        rejected_empty = 0
        rejected_short = 0
        duplicates = 0

        fingerprints = set()

        for document in documents:
            text = self.normalize(document.text)

            if not text:
                rejected_empty += 1
                continue

            if len(text) < self.minimum_length:
                rejected_short += 1
                continue

            fingerprint = self._fingerprint(text)

            if fingerprint in fingerprints:
                duplicates += 1
                continue

            fingerprints.add(fingerprint)
            accepted.append(text)

        if not accepted:
            raise RealCorpusError(
                "No usable documents remained after filtering."
            )

        return accepted, (
            discovered,
            len(accepted),
            rejected_empty,
            rejected_short,
            duplicates,
        )

    # ---------------------------------------------------------
    # Split
    # ---------------------------------------------------------

    def split(self, texts):
        values = list(texts)

        rng = random.Random(self.seed)
        rng.shuffle(values)

        if len(values) == 1:
            train = values
            validation = []
        else:
            validation_count = int(
                len(values) * self.validation_ratio
            )

            if self.validation_ratio > 0:
                validation_count = max(
                    1,
                    validation_count,
                )

            validation_count = min(
                validation_count,
                len(values) - 1,
            )

            if validation_count == 0:
                train = values
                validation = []
            else:
                validation = values[
                    -validation_count:
                ]
                train = values[
                    :-validation_count
                ]

        return CorpusSplit(
            train=train,
            validation=validation,
        )

    # ---------------------------------------------------------
    # Full pipeline
    # ---------------------------------------------------------

    def run(self, root):
        paths = self.discover(root)

        documents = self.load_files(paths)

        texts, counters = self.prepare(
            documents
        )

        split = self.split(texts)

        (
            discovered,
            accepted,
            rejected_empty,
            rejected_short,
            duplicates,
        ) = counters

        self.stats = CorpusStats(
            discovered=discovered,
            accepted=accepted,
            rejected_empty=rejected_empty,
            rejected_short=rejected_short,
            duplicates=duplicates,
            train_count=len(split.train),
            validation_count=len(split.validation),
        )

        return split

    # ---------------------------------------------------------
    # Tokenized dataset
    # ---------------------------------------------------------

    def tokenize(
        self,
        texts,
        tokenizer,
        sequence_length=16,
    ):
        from neura.training.dataset import (
            TextDataset,
        )

        if not texts:
            return TextDataset(
                [],
                tokenizer,
                sequence_length=sequence_length,
            )

        return TextDataset(
            texts,
            tokenizer,
            sequence_length=sequence_length,
        )


__all__ = [
    "CorpusDocument",
    "CorpusSplit",
    "CorpusStats",
    "RealCorpusError",
    "RealCorpusPipeline",
]
