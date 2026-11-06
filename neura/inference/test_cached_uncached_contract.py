from neura.tokenizer import VOCAB, TOKENIZER
from neura.model.model import NeuraModel
from neura.model.config import NeuraTiny


print("=" * 60)
print("NEURA CACHED / UNCACHED EQUIVALENCE CONTRACT")
print("=" * 60)

VOCAB.ensure_size(32768)
TOKENIZER.unfreeze()

tokens = TOKENIZER.encode("Hello Neura")

print("Prompt tokens:", tokens)
print("Prompt tokenization: PASS")

# ---------------------------------------------------------
# Uncached reference
# ---------------------------------------------------------

uncached_model = NeuraModel(
    config=NeuraTiny()
)

uncached_logits = uncached_model.forward(
    tokens,
    use_cache=False,
)

print("Uncached forward: PASS")

assert len(uncached_logits) == len(tokens)
assert len(uncached_logits[-1]) == 32768

print("Uncached shape: PASS")

# ---------------------------------------------------------
# Cached prefill
# ---------------------------------------------------------

cached_model = NeuraModel(
    config=NeuraTiny()
)

prefill_logits = cached_model.forward(
    tokens,
    use_cache=True,
)

print("Cached prefill: PASS")

assert len(prefill_logits) == len(tokens)
assert len(prefill_logits[-1]) == 32768

print("Cached prefill shape: PASS")

cache_after_prefill = cached_model.cache_size()

assert cache_after_prefill > 0

print("Prefill cache population: PASS")

# ---------------------------------------------------------
# Compare final prefill logits
# ---------------------------------------------------------

reference = uncached_logits[-1]
cached = prefill_logits[-1]

assert len(reference) == len(cached)

max_difference = max(
    abs(a - b)
    for a, b in zip(reference, cached)
)

print(
    "Prefill max logit difference:",
    max_difference
)

assert max_difference < 1e-9

print("Prefill cached/uncached equivalence: PASS")

# ---------------------------------------------------------
# Incremental decode
# ---------------------------------------------------------

next_token = 5

incremental_logits = cached_model.forward(
    [next_token],
    use_cache=True,
)

print("Incremental decode: PASS")

assert len(incremental_logits) == 1
assert len(incremental_logits[0]) == 32768

print("Incremental output shape: PASS")

cache_after_decode = cached_model.cache_size()

assert cache_after_decode > cache_after_prefill

print("Incremental cache growth: PASS")

# ---------------------------------------------------------
# Cleanup
# ---------------------------------------------------------

cached_model.clear_cache()

assert cached_model.cache_size() == 0

print("Cache cleanup: PASS")

print("=" * 60)
print("RESULT: CACHED / UNCACHED EQUIVALENCE CONTRACT PASSED")
print("=" * 60)
