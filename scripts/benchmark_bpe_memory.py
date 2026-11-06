from __future__ import annotations

import gc
import os
import resource
import time

from neura.tokenizer.bpe import BPETrainer


CORPUS = "data/processed/wiki_ar_00006.tokenizer.txt"
MAX_LINES = 10_000
VOCAB_SIZE = 512
MIN_FREQUENCY = 2


def read_lines(path: str, max_lines: int) -> list[str]:
    lines: list[str] = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n\r")
            if not line:
                continue

            lines.append(line)

            if len(lines) >= max_lines:
                break

    return lines


def memory_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)

    # Linux reports ru_maxrss in KiB.
    return usage.ru_maxrss / 1024.0


def main() -> None:
    print("===== BPE MEMORY BENCHMARK =====")

    if not os.path.exists(CORPUS):
        raise FileNotFoundError(CORPUS)

    start = time.perf_counter()
    lines = read_lines(CORPUS, MAX_LINES)
    load_time = time.perf_counter() - start

    chars = sum(len(x) for x in lines)
    byte_count = sum(len(x.encode("utf-8")) for x in lines)

    print(f"lines:       {len(lines):,}")
    print(f"chars:       {chars:,}")
    print(f"bytes:       {byte_count:,}")
    print(f"load_time:   {load_time:.3f}s")
    print(f"RAM_loaded:  {memory_mb():.1f} MB")

    print()
    print("===== TRAINING =====")

    gc.collect()

    trainer = BPETrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQUENCY,
    )

    before = memory_mb()
    start = time.perf_counter()

    trainer.train_texts(lines)

    train_time = time.perf_counter() - start
    after = memory_mb()

    print(f"train_time:  {train_time:.3f}s")
    print(f"RAM_before:  {before:.1f} MB")
    print(f"RAM_after:   {after:.1f} MB")
    print(f"RAM_delta:   {after - before:.1f} MB")
    print(f"vocab_size:  {trainer.vocab_size}")
    print(f"merges:      {len(trainer.merges):,}")

    print()
    print("===== RESULT =====")
    print("BENCHMARK: PASS")


if __name__ == "__main__":
    main()
