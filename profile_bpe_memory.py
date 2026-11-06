import os
import sys
import time
import resource

from neura.tokenizer.bpe import BPETrainer

CORPUS = "data/processed/neura_arabic_bpe_32k_v4/train_corpus.txt"

def rss_mb():
    return resource.getrusage(
        resource.RUSAGE_SELF
    ).ru_maxrss / 1024.0

print("===== BPE MEMORY PROFILE =====", flush=True)
print("PID:", os.getpid(), flush=True)
print("Corpus:", CORPUS, flush=True)
print(flush=True)

trainer = BPETrainer(
    vocab_size=32768,
    min_frequency=2,
)

print("Loading corpus...", flush=True)

with open(CORPUS, "r", encoding="utf-8") as f:
    texts = [line.rstrip("\n") for line in f]

print("Texts:", len(texts), flush=True)
print(
    "RSS after loading:",
    f"{rss_mb():.1f} MB",
    flush=True,
)

print(flush=True)
print("Calling train_texts()...", flush=True)
print(
    "If the process dies after this point, "
    "the failure is inside BPE training.",
    flush=True,
)

start = time.monotonic()

try:
    result = trainer.train_texts(
        texts,
        checkpoint_interval=0,
    )

    elapsed = time.monotonic() - start

    print(flush=True)
    print("===== SUCCESS =====", flush=True)
    print("Elapsed:", f"{elapsed:.1f}s", flush=True)
    print("RSS:", f"{rss_mb():.1f} MB", flush=True)
    print("Result:", result, flush=True)

except Exception as exc:
    print(flush=True)
    print("===== PYTHON EXCEPTION =====", flush=True)
    print(
        type(exc).__name__,
        str(exc),
        flush=True,
    )
    raise
