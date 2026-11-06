from neura.tokenizer import VOCAB, TOKENIZER
from neura.model.model import NeuraModel
from neura.model.config import NeuraTiny

from neura.inference import (
    InferenceEngine,
    InferenceError,
    InferenceRequest,
    InferenceResponse,
    InferenceState,
)


print("=" * 60)
print("NEURA RUNTIME CORE CONTRACT")
print("=" * 60)

VOCAB.ensure_size(32768)
TOKENIZER.unfreeze()

tokens = TOKENIZER.encode("Hello Neura")

print("Prompt tokenization: PASS")

model = NeuraModel(
    config=NeuraTiny()
)

engine = InferenceEngine(model)

assert engine.state == InferenceState.CREATED
assert engine.active is False

print("Initial runtime state: PASS")

request = InferenceRequest(
    prompt_tokens=tokens,
    max_tokens=4,
)

print("Request creation: PASS")

# ---------------------------------------------------------
# PREFILL
# ---------------------------------------------------------

logits = engine.prefill(request)

assert engine.active is True
assert engine.state == InferenceState.DECODING
assert len(logits) == len(tokens)
assert engine.prompt_tokens == tokens
assert engine.generated_tokens == []

print("Prefill execution: PASS")
print("Prefill state: PASS")
print("Prefill logits: PASS")
print("Prompt preservation: PASS")

assert model.cache_size() > 0

print("Prefill KV population: PASS")

# ---------------------------------------------------------
# DECODE
# ---------------------------------------------------------

token, response = engine.decode()

assert isinstance(token, int)
assert 0 <= token < VOCAB.capacity

print("Single-token decode: PASS")
print("Decoded token validity: PASS")

assert len(engine.generated_tokens) == 1

print("Generated-state update: PASS")

# Continue until completion.
while engine.active:
    token, response = engine.decode()

    if response is not None:
        break

assert isinstance(response, InferenceResponse)

print("Decode lifecycle completion: PASS")
print("Response type: PASS")

assert response.prompt_tokens == tokens

print("Response prompt preservation: PASS")

assert response.tokens == (
    response.prompt_tokens
    + response.generated_tokens
)

print("Response token composition: PASS")

assert response.finish_reason in (
    "eos",
    "length",
)

print("Finish reason: PASS")

assert engine.active is False
assert engine.state == InferenceState.FINISHED

print("Runtime completion state: PASS")

assert model.cache_size() == 0

print("Cache cleanup: PASS")

# ---------------------------------------------------------
# SECOND REQUEST ISOLATION
# ---------------------------------------------------------

second = InferenceRequest(
    prompt_tokens=tokens,
    max_tokens=1,
)

response2 = engine.generate(second)

assert isinstance(response2, InferenceResponse)
assert response2.prompt_tokens == tokens
assert engine.active is False
assert model.cache_size() == 0

print("Second request isolation: PASS")
print("Second request cleanup: PASS")

# ---------------------------------------------------------
# ERROR CONTRACT
# ---------------------------------------------------------

try:
    engine.decode()
    raise AssertionError(
        "decode() should fail without active request"
    )
except InferenceError:
    pass

print("Invalid decode lifecycle protection: PASS")

print("=" * 60)
print("RESULT: RUNTIME CORE CONTRACT PASSED")
print("=" * 60)
