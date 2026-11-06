#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "sft"

FILES = [
    "neura_general.jsonl",
    "neura_arabic.jsonl",
    "neura_iraqi.jsonl",
    "neura_programming.jsonl",
    "neura_reasoning.jsonl",
    "neura_multiturn.jsonl",
]


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def fingerprint(record: dict) -> str:
    messages = record["messages"]

    text = "\n".join(
        f"{m['role']}:{normalize(m['content'])}"
        for m in messages
    )

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def main() -> None:
    print("===== NEURA SFT DATA AUDIT =====")

    all_records = []
    fingerprints = {}
    duplicate_count = 0

    total_chars = 0
    total_words = 0

    role_counts = Counter()
    category_counts = Counter()

    for filename in FILES:
        path = DATA_DIR / filename

        if not path.exists():
            print(f"WARNING: missing {filename}")
            continue

        category = filename.removeprefix(
            "neura_"
        ).removesuffix(
            ".jsonl"
        )

        records = []

        with path.open(
            "r",
            encoding="utf-8",
        ) as f:
            for line_no, line in enumerate(
                f,
                1,
            ):
                if not line.strip():
                    continue

                record = json.loads(line)

                messages = record["messages"]

                assert isinstance(
                    messages,
                    list,
                )

                assert any(
                    m["role"] == "assistant"
                    for m in messages
                )

                fp = fingerprint(record)

                if fp in fingerprints:
                    duplicate_count += 1
                    print(
                        "DUPLICATE:",
                        filename,
                        "line",
                        line_no,
                        "matches",
                        fingerprints[fp],
                    )
                else:
                    fingerprints[fp] = (
                        f"{filename}:line{line_no}"
                    )

                records.append(record)
                all_records.append(record)

                category_counts[category] += 1

                for message in messages:
                    role = message["role"]
                    content = message["content"]

                    role_counts[role] += 1
                    total_chars += len(content)
                    total_words += len(
                        content.split()
                    )

        print(
            f"{filename:28s}"
            f"{len(records):4d} examples"
        )

    # ------------------------------------------------------------
    # Conversation statistics
    # ------------------------------------------------------------

    assistant_lengths = []
    user_lengths = []
    turns = []

    for record in all_records:
        messages = record["messages"]

        turns.append(len(messages))

        for message in messages:
            length = len(
                message["content"]
            )

            if message["role"] == "assistant":
                assistant_lengths.append(length)

            elif message["role"] == "user":
                user_lengths.append(length)

    print()
    print("===== SUMMARY =====")

    print(
        "examples:",
        len(all_records),
    )

    print(
        "unique examples:",
        len(fingerprints),
    )

    print(
        "duplicates:",
        duplicate_count,
    )

    print(
        "total characters:",
        f"{total_chars:,}",
    )

    print(
        "total words:",
        f"{total_words:,}",
    )

    print()
    print("roles:")

    for role in (
        "system",
        "user",
        "assistant",
    ):
        print(
            f"  {role:10s}: "
            f"{role_counts[role]:,}"
        )

    print()
    print("categories:")

    for category, count in sorted(
        category_counts.items()
    ):
        print(
            f"  {category:14s}: "
            f"{count:,}"
        )

    print()
    print("conversation turns:")

    if turns:
        print(
            "  min:",
            min(turns),
        )
        print(
            "  max:",
            max(turns),
        )
        print(
            "  avg:",
            f"{sum(turns) / len(turns):.2f}",
        )

    print()
    print("assistant response length:")

    if assistant_lengths:
        print(
            "  min:",
            min(assistant_lengths),
        )
        print(
            "  max:",
            max(assistant_lengths),
        )
        print(
            "  avg:",
            f"{sum(assistant_lengths) / len(assistant_lengths):.2f}",
        )

    print()
    print("user message length:")

    if user_lengths:
        print(
            "  min:",
            min(user_lengths),
        )
        print(
            "  max:",
            max(user_lengths),
        )
        print(
            "  avg:",
            f"{sum(user_lengths) / len(user_lengths):.2f}",
        )

    print()
    print("===== VALIDATION =====")

    assert duplicate_count == 0
    assert len(all_records) > 0
    assert role_counts["assistant"] > 0
    assert role_counts["user"] > 0

    print("duplicate check: PASS")
    print("assistant check: PASS")
    print("user check: PASS")
    print("dataset integrity: PASS")


if __name__ == "__main__":
    main()
