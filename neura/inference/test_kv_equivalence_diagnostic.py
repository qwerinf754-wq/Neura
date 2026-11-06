from neura.tokenizer import VOCAB, TOKENIZER
from neura.model.model import NeuraModel
from neura.model.config import NeuraTiny


print("=" * 60)
print("NEURA KV EQUIVALENCE DIAGNOSTIC")
print("=" * 60)

VOCAB.ensure_size(32768)
TOKENIZER.unfreeze()

tokens = TOKENIZER.encode("Hello Neura")

print("Prompt tokens:", tokens)

model = NeuraModel(
    config=NeuraTiny()
)

# ---------------------------------------------------------
# 1. Uncached reference
# ---------------------------------------------------------

uncached = model.forward(
    tokens,
    use_cache=False,
)

print("Uncached forward: PASS")

# ---------------------------------------------------------
# 2. Clear cache
# ---------------------------------------------------------

model.clear_cache()

assert model.cache_size() == 0

print("Cache reset before cached path: PASS")

# ---------------------------------------------------------
# 3. Cached prefill
# ---------------------------------------------------------

cached = model.forward(
    tokens,
    use_cache=True,
)

print("Cached prefill: PASS")

assert len(uncached) == len(tokens)
assert len(cached) == len(tokens)

print("Sequence length equivalence: PASS")

# ---------------------------------------------------------
# 4. Compare every position
# ---------------------------------------------------------

max_difference = 0.0
max_position = None

for position, (reference, candidate) in enumerate(
    zip(uncached, cached)
):

    difference = max(
        abs(a - b)
        for a, b in zip(
            reference,
            candidate
        )
    )

    print(
        f"Position {position}: "
        f"max_difference={difference}"
    )

    if difference > max_difference:
        max_difference = difference
        max_position = position

print()
print(
    "Maximum difference:",
    max_difference
)

print(
    "Maximum-difference position:",
    max_position
)

print(
    "Cache size after prefill:",
    model.cache_size()
)

# ---------------------------------------------------------
# 5. Cache accounting
# ---------------------------------------------------------

assert model.cache_size() > 0

print("Cache population: PASS")

# ---------------------------------------------------------
# 6. Cleanup
# ---------------------------------------------------------

model.clear_cache()

assert model.cache_size() == 0

print("Cache cleanup: PASS")

print("=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)
