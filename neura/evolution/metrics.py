"""Generation metrics for Neura Evolution Engine."""

from __future__ import annotations

from collections import Counter
from typing import Any


def generation_metrics(text: str) -> dict[str, Any]:
    text = str(text or "").strip()

    if not text:
        return {
            "characters": 0,
            "words": 0,
            "unique_words": 0,
            "unique_ratio": 0.0,
            "repeated_word_count": 0,
            "max_word_frequency": 0,
        }

    words = text.split()
    counts = Counter(words)

    repeated_word_count = sum(
        count - 1
        for count in counts.values()
        if count > 1
    )

    return {
        "characters": len(text),
        "words": len(words),
        "unique_words": len(counts),
        "unique_ratio": round(
            len(counts) / len(words),
            6,
        ),
        "repeated_word_count": repeated_word_count,
        "max_word_frequency": max(counts.values()),
    }
