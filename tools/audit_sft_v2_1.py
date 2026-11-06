import argparse
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

ROLES = {"system", "user", "assistant"}
BUCKETS = [
    (15, 40),
    (41, 100),
    (101, 180),
    (181, 300),
    (301, 500),
    (501, 800),
]


def load_jsonl(path):
    rows = []
    if not path.exists():
        return rows

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception as exc:
                raise ValueError(
                    f"{path}:{line_no}: invalid JSON: {exc}"
                ) from exc
            rows.append((line_no, obj))
    return rows


def normalize_text(text):
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def first_words(text, n=5):
    words = re.findall(r"\S+", str(text).strip().lower())
    return tuple(words[:n])


def assistant_messages(messages):
    return [
        str(m.get("content", ""))
        for m in messages
        if m.get("role") == "assistant"
    ]


def user_messages(messages):
    return [
        str(m.get("content", ""))
        for m in messages
        if m.get("role") == "user"
    ]


def token_estimate(text):
    # Byte-level compatible lower-bound diagnostic.
    # Exact tokenizer statistics are added separately when available.
    return len(text.encode("utf-8"))


def validate_split(name, rows):
    errors = []
    warnings = []

    exact_seen = {}
    normalized_seen = {}
    opening_seen = Counter()

    examples = 0
    assistant_count = 0
    multi_turn = 0

    char_lengths = []
    token_lengths = []
    bucket_counts = Counter()

    for line_no, obj in rows:
        examples += 1

        if not isinstance(obj, dict):
            errors.append(f"{name}:{line_no}: example is not an object")
            continue

        messages = obj.get("messages")

        if not isinstance(messages, list) or not messages:
            errors.append(
                f"{name}:{line_no}: missing/empty messages list"
            )
            continue

        roles = []
        for idx, msg in enumerate(messages):
            if not isinstance(msg, dict):
                errors.append(
                    f"{name}:{line_no}: message[{idx}] is not an object"
                )
                continue

            role = msg.get("role")
            content = msg.get("content")

            roles.append(role)

            if role not in ROLES:
                errors.append(
                    f"{name}:{line_no}: invalid role={role!r}"
                )

            if content is None or not str(content).strip():
                errors.append(
                    f"{name}:{line_no}: empty content at message[{idx}]"
                )

        users = user_messages(messages)
        assistants = assistant_messages(messages)

        if not users:
            errors.append(f"{name}:{line_no}: no user message")

        if not assistants:
            errors.append(f"{name}:{line_no}: no assistant message")

        if messages and messages[-1].get("role") != "assistant":
            warnings.append(
                f"{name}:{line_no}: last message is not assistant"
            )

        if len(users) > 1:
            multi_turn += 1

        assistant_count += len(assistants)

        for answer in assistants:
            chars = len(answer)
            tokens = token_estimate(answer)

            char_lengths.append(chars)
            token_lengths.append(tokens)

            for lo, hi in BUCKETS:
                if lo <= tokens <= hi:
                    bucket_counts[f"{lo}-{hi}"] += 1
                    break

            norm = normalize_text(answer)
            exact_seen[answer] = exact_seen.get(answer, 0) + 1
            normalized_seen[norm] = normalized_seen.get(norm, 0) + 1

            opening_seen[first_words(answer)] += 1

    duplicate_answers = {
        k: v for k, v in exact_seen.items() if v > 1
    }

    normalized_duplicates = {
        k: v for k, v in normalized_seen.items() if v > 1
    }

    repeated_openings = {
        " ".join(k): v
        for k, v in opening_seen.items()
        if k and v > 1
    }

    def stats(values):
        if not values:
            return {
                "min": 0,
                "median": 0,
                "mean": 0,
                "max": 0,
            }

        ordered = sorted(values)
        n = len(ordered)

        if n % 2:
            median = ordered[n // 2]
        else:
            median = (
                ordered[n // 2 - 1] +
                ordered[n // 2]
            ) / 2

        return {
            "min": min(values),
            "median": median,
            "mean": sum(values) / len(values),
            "max": max(values),
        }

    return {
        "split": name,
        "examples": examples,
        "assistant_messages": assistant_count,
        "multi_turn_examples": multi_turn,
        "multi_turn_ratio": (
            multi_turn / examples if examples else 0
        ),
        "assistant_chars": stats(char_lengths),
        "assistant_token_estimate_utf8_bytes": stats(token_lengths),
        "length_buckets": dict(bucket_counts),
        "exact_duplicate_answers": duplicate_answers,
        "normalized_duplicate_answers": normalized_duplicates,
        "repeated_first_5_words": repeated_openings,
        "errors": errors,
        "warnings": warnings,
        "status": "FAIL" if errors else "PASS",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="data/processed/neura_chat_sft_v2_1",
    )
    parser.add_argument(
        "--stage",
        default="seed",
        choices=["seed", "100", "250", "500", "750", "1000"],
    )
    args = parser.parse_args()

    base = ROOT / args.dataset

    paths = {
        "train": base / "train.jsonl",
        "validation": base / "validation.jsonl",
        "test": base / "test.jsonl",
    }

    report = {
        "dataset": str(base.relative_to(ROOT)),
        "stage": args.stage,
        "targets": {
            "seed": {"train": 40, "validation": 0, "test": 0},
            "100": {"train": 100, "validation": 15, "test": 15},
            "250": {"train": 250, "validation": 38, "test": 38},
            "500": {"train": 500, "validation": 75, "test": 75},
            "750": {"train": 750, "validation": 113, "test": 113},
            "1000": {"train": 1000, "validation": 150, "test": 150},
        }[args.stage],
        "splits": {},
    }

    all_ok = True

    for name, path in paths.items():
        rows = load_jsonl(path)
        result = validate_split(name, rows)
        report["splits"][name] = result

        if result["errors"]:
            all_ok = False

    target = report["targets"]

    for name, expected in target.items():
        actual = report["splits"][name]["examples"]

        if actual != expected:
            report["splits"][name]["target_check"] = {
                "expected": expected,
                "actual": actual,
                "status": "FAIL",
            }
            all_ok = False
        else:
            report["splits"][name]["target_check"] = {
                "expected": expected,
                "actual": actual,
                "status": "PASS",
            }

    report["status"] = "PASS" if all_ok else "FAIL"

    out = base / "audit.json"
    out.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 72)
    print("NEURA CHAT SFT V2.1 OFFICIAL AUDIT")
    print("=" * 72)
    print(f"DATASET : {base.relative_to(ROOT)}")
    print(f"STAGE   : {args.stage}")
    print()

    for name in ("train", "validation", "test"):
        r = report["splits"][name]
        tc = r["target_check"]

        print(
            f"{name:12s} "
            f"examples={r['examples']:4d} "
            f"assistant={r['assistant_messages']:4d} "
            f"multi_turn={r['multi_turn_examples']:4d} "
            f"status={r['status']}"
        )

        print(
            f"  target: {tc['actual']} / {tc['expected']} "
            f"[{tc['status']}]"
        )

        if r["errors"]:
            for err in r["errors"][:10]:
                print(f"  ERROR: {err}")

        if r["warnings"]:
            for warning in r["warnings"][:5]:
                print(f"  WARN : {warning}")

    print()
    print(f"AUDIT JSON: {out}")
    print(f"FINAL STATUS: {report['status']}")

    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
