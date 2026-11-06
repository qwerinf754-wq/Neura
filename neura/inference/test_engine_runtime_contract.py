from neura.tokenizer import VOCAB, TOKENIZER
from neura.model.model import NeuraModel
from neura.model.config import NeuraTiny

from neura.inference.engine import (
    InferenceEngine,
    InferenceRequest,
)


print("=" * 60)
print("NEURA OFFICIAL INFERENCE ENGINE CONTRACT")
print("=" * 60)


# ------------------------------------------------------------
# Vocabulary
# ------------------------------------------------------------

VOCAB.ensure_size(32768)
TOKENIZER.unfreeze()

tokens = TOKENIZER.encode(
    "Hello Neura"
)

assert tokens
print("Prompt tokenization: PASS")


# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

model = NeuraModel(
    config=NeuraTiny()
)

print("Model creation: PASS")


# ------------------------------------------------------------
# Engine
# ------------------------------------------------------------

engine = InferenceEngine(model)

print("InferenceEngine creation: PASS")


# ------------------------------------------------------------
# Request
# ------------------------------------------------------------

request = InferenceRequest(
    prompt_tokens=tokens,
    max_tokens=8,
)

print("Request creation: PASS")


# ------------------------------------------------------------
# Generation
# ------------------------------------------------------------

result = engine.generate(request)

print(
    "Prompt tokens:",
    result.prompt_tokens
)

print(
    "Generated tokens:",
    result.generated_tokens
)

print(
    "Finish reason:",
    result.finish_reason
)


# ------------------------------------------------------------
# Prompt preservation
# ------------------------------------------------------------

assert result.prompt_tokens == tokens

print(
    "Prompt preservation: PASS"
)


# ------------------------------------------------------------
# Token validity
# ------------------------------------------------------------

assert all(
    0 <= token < 32768
    for token in result.generated_tokens
)

print(
    "Generated token validity: PASS"
)


# ------------------------------------------------------------
# Special token protection
# ------------------------------------------------------------

generated = result.generated_tokens

assert 0 not in generated
assert 1 not in generated
assert 2 not in generated

print(
    "Blocked-token protection: PASS"
)


# ------------------------------------------------------------
# Finish reason
# ------------------------------------------------------------

assert result.finish_reason in {
    "eos",
    "length",
}

print(
    "Finish reason contract: PASS"
)


# ------------------------------------------------------------
# Cache cleanup
# ------------------------------------------------------------

assert model.kv_cache.total_size() == 0

print(
    "Cache cleanup: PASS"
)


# ------------------------------------------------------------
# Request isolation
# ------------------------------------------------------------

second = engine.generate(
    InferenceRequest(
        prompt_tokens=tokens,
        max_tokens=4,
    )
)

assert second.prompt_tokens == tokens

print(
    "Second request isolation: PASS"
)

assert model.kv_cache.total_size() == 0

print(
    "Final cache cleanup: PASS"
)


print("=" * 60)
print(
    "RESULT: OFFICIAL INFERENCE ENGINE CONTRACT PASSED"
)
print("=" * 60)
