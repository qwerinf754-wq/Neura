from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path("data/sft")
OUTPUT = ROOT / "neura_sft_master_v1.jsonl"
REPORT = ROOT / "neura_sft_master_v1.report.json"

EXCLUDED = {
    OUTPUT.name,
}

def fingerprint(item: dict) -> str:
    payload = json.dumps(
        item["messages"],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate(item: dict) -> tuple[bool, str]:
    if not isinstance(item, dict):
        return False, "not_object"

    messages = item.get("messages")

    if not isinstance(messages, list):
        return False, "missing_messages"

    if len(messages) < 2:
        return False, "too_few_messages"

    roles = [m.get("role") for m in messages]

    if "user" not in roles:
        return False, "missing_user"

    if "assistant" not in roles:
        return False, "missing_assistant"

    for message in messages:
        if message.get("role") not in {
            "system",
            "user",
            "assistant",
        }:
            return False, "invalid_role"

        if not isinstance(message.get("content"), str):
            return False, "invalid_content"

        if not message["content"].strip():
            return False, "empty_content"

    if roles[-1] != "assistant":
        return False, "last_message_not_assistant"

    return True, "ok"


def main() -> None:
    files = sorted(
        ROOT.glob("*.jsonl"),
        key=lambda p: p.name,
    )

    selected = [
        path for path in files
        if path.name not in EXCLUDED
    ]

    print("===== NEURA SFT MASTER BUILDER =====")
    print(f"source files: {len(selected)}")
    print()

    seen: set[str] = set()
    dataset: list[dict] = []

    stats = {
        "files": Counter(),
        "domains": Counter(),
        "categories": Counter(),
        "languages": Counter(),
        "roles": Counter(),
        "validation": Counter(),
    }

    total_input = 0
    duplicates = 0

    for path in selected:
        file_count = 0

        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()

                if not line:
                    continue

                total_input += 1

                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    stats["validation"]["invalid_json"] += 1
                    continue

                valid, reason = validate(item)

                if not valid:
                    stats["validation"][reason] += 1
                    continue

                fp = fingerprint(item)

                if fp in seen:
                    duplicates += 1
                    continue

                seen.add(fp)

                metadata = item.get("metadata", {})

                domain = metadata.get(
                    "domain",
                    "unknown",
                )

                category = metadata.get(
                    "category",
                    "unknown",
                )

                language = metadata.get(
                    "language",
                    "unknown",
                )

                stats["files"][path.name] += 1
                stats["domains"][domain] += 1
                stats["categories"][category] += 1
                stats["languages"][language] += 1

                for message in item["messages"]:
                    stats["roles"][message["role"]] += 1

                dataset.append(item)
                file_count += 1

        print(f"{path.name:<38} {file_count:>5}")

    with OUTPUT.open("w", encoding="utf-8") as f:
        for item in dataset:
            f.write(
                json.dumps(
                    item,
                    ensure_ascii=False,
                )
                + "\n"
            )

    report = {
        "version": "neura-sft-master-v1",
        "total_input_examples": total_input,
        "unique_examples": len(dataset),
        "duplicates_removed": duplicates,
        "output": str(OUTPUT),
        "files": dict(stats["files"]),
        "domains": dict(stats["domains"]),
        "categories": dict(stats["categories"]),
        "languages": dict(stats["languages"]),
        "roles": dict(stats["roles"]),
        "validation": dict(stats["validation"]),
        "status": "PASS",
    }

    REPORT.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("===== MASTER SUMMARY =====")
    print(f"input examples:     {total_input}")
    print(f"unique examples:    {len(dataset)}")
    print(f"duplicates removed: {duplicates}")

    print()
    print("domains:")
    for name, count in sorted(stats["domains"].items()):
        print(f"  {name:<22}: {count}")

    print()
    print("languages:")
    for name, count in sorted(stats["languages"].items()):
        print(f"  {name:<22}: {count}")

    print()
    print("roles:")
    for name, count in sorted(stats["roles"].items()):
        print(f"  {name:<12}: {count}")

    print()
    print(f"dataset: {OUTPUT}")
    print(f"report:  {REPORT}")
    print()
    print("MASTER DATASET: PASS")


if __name__ == "__main__":
    main()
