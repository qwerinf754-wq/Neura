from neura.tokenizer import VOCAB, TOKENIZER
from neura.model.model import NeuraModel
from neura.model.config import NeuraTiny


print("=" * 60)
print("NEURA INCREMENTAL DECODE EQUIVALENCE CONTRACT")
print("=" * 60)

VOCAB.ensure_size(32768)
TOKENIZER.unfreeze()

prompt = TOKENIZER.encode("Hello Neura")

assert len(prompt) >= 2

next_token = 5
full_sequence = prompt + [next_token]

print("Prompt:", prompt)
print("Next token:", next_token)
print("Full sequence:", full_sequence)

model = NeuraModel(
    config=NeuraTiny()
)

print("Model creation: PASS")

# ---------------------------------------------------------
# Reference: uncached full sequence
# ---------------------------------------------------------

full_logits = model.forward(
    full_sequence,
    use_cache=False,
)

print("Full forward: PASS")

assert len(full_logits) == len(full_sequence)

reference = full_logits[-1]

# ---------------------------------------------------------
# Reset cache explicitly
# ---------------------------------------------------------

model.clear_cache()

assert model.cache_size() == 0

print("Cache reset: PASS")

# ---------------------------------------------------------
# Cached prefill
# ---------------------------------------------------------

prefill_logits = model.forward(
    prompt,
    use_cache=True,
)

print("Cached prefill: PASS")

assert len(prefill_logits) == len(prompt)

prefill_cache_size = model.cache_size()

assert prefill_cache_size > 0

print(
    "Prefill cache size:",
    prefill_cache_size,
)

# ---------------------------------------------------------
# Incremental decode
# ---------------------------------------------------------

decode_logits = model.forward(
    [next_token],
    use_cache=True,
)

print("Incremental decode: PASS")

assert len(decode_logits) == 1

candidate = decode_logits[0]

# ---------------------------------------------------------
# Compare exact logits
# ---------------------------------------------------------

max_difference = max(
    abs(a - b)
    for a, b in zip(
        reference,
        candidate,
    )
)

print(
    "Incremental max logit difference:",
    max_difference,
)

if max_difference == 0.0:
    print(
        "Full / incremental equivalence: PASS"
    )
else:
    print(
        "Full / incremental equivalence: FAIL"
    )

# ---------------------------------------------------------
# Cache growth
# ---------------------------------------------------------

final_cache_size = model.cache_size()

print(
    "Final cache size:",
    final_cache_size,
)

assert final_cache_size > prefill_cache_size

print("Cache growth: PASS")

# ---------------------------------------------------------
# Cleanup
# ---------------------------------------------------------

model.clear_cache()

assert model.cache_size() == 0

print("Cache cleanup: PASS")

print("=" * 60)

if max_difference < 1e-9:
    print(
        "RESULT: INCREMENTAL DECODE EQUIVALENCE PASSED"
    )
else:
    print(
        "RESULT: INCREMENTAL DECODE EQUIVALENCE FAILED"
    )
    raise SystemExit(1)

print("=" * 60)
