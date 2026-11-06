from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


OUT = Path("data/sft/neura_programming_languages.jsonl")


def example(
    language: str,
    category: str,
    user: str,
    assistant: str,
    system: str = (
        "You are Neura, an expert programming assistant. "
        "Give correct, practical, clear answers and explain important decisions."
    ),
) -> dict[str, Any]:
    return {
        "metadata": {
            "domain": "programming",
            "language": language,
            "category": category,
        },
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
    }


def build_examples() -> list[dict[str, Any]]:
    data: list[dict[str, Any]] = []

    # ============================================================
    # PYTHON
    # ============================================================

    data += [
        example(
            "python",
            "algorithm",
            "اكتب دالة Python تتحقق هل الرقم أولي.",
            """يمكنك اختبار القواسم حتى الجذر التربيعي للرقم:

```python
def is_prime(n):
    if n < 2:
        return False

    divisor = 2
    while divisor * divisor <= n:
        if n % divisor == 0:
            return False
        divisor += 1

    return True
