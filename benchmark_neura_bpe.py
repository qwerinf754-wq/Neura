from pathlib import Path
import time

from neura.tokenizer.bpe import BPETrainer


CORPUS = Path(
    "data/processed/wiki_ar_00006.tokenizer.txt"
)

SAMPLE_LINES = 1000
VOCAB_SIZE = 512


print("=" * 70)
print("NEURA BPE MERGE PERFORMANCE BENCHMARK")
print("=" * 70)

texts = []

with CORPUS.open(
    "r",
    encoding="utf-8",
    errors="strict",
) as f:

    for line in f:
        line = line.strip()

        if not line:
            continue

        texts.append(line)

        if len(texts) >= SAMPLE_LINES:
            break

print(f"Sample lines : {len(texts):,}")
print(f"Vocab target : {VOCAB_SIZE:,}")
print()

trainer = BPETrainer(
    vocab_size=VOCAB_SIZE
)

start = time.perf_counter()

trainer.train_texts(
    texts,
    checkpoint_interval=0,
)

elapsed = time.perf_counter() - start

print()
print("=" * 70)
print("BENCHMARK RESULT")
print("=" * 70)

print(f"Elapsed time : {elapsed:.3f}s")
print(f"Vocabulary   : {len(trainer.vocab):,}")
print(f"Merges       : {len(trainer.merges):,}")

print()

if len(trainer.merges) == 0:
    print("STATUS: NO MERGES GENERATED")
elif elapsed > 60:
    print("STATUS: BPE MERGE PATH TOO SLOW")
elif elapsed > 10:
    print("STATUS: BPE MERGE PATH NEEDS OPTIMIZATION")
else:
    print("STATUS: BPE MERGE PATH ACCEPTABLE")

print("=" * 70)
