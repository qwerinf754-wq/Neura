from neura.tokenizer import VOCAB, TOKENIZER
from neura.model.model import NeuraModel
from neura.model.config import NeuraTiny

from neura.inference.engine import (
    InferenceEngine,
    InferenceError,
    InferenceRequest,
    InferenceResponse,
    InferenceState,
)


print("=" * 60)
print("NEURA PREFILL / DECODE RUNTIME CONTRACT")
print("=" * 60)

VOCAB.ensure_size(32768)
TOKENIZER.unfreeze()

tokens = TOKENIZER.encode(
    "Hello Neura"
)

print("Prompt tokenization: PASS")

model = NeuraModel(
    config=NeuraTiny()
)

engine = InferenceEngine(
    model
)

print("InferenceEngine creation: PASS")

request = InferenceRequest(
    prompt_tokens=tokens,
    max_tokens=4,
)

print("Request creation: PASS")

# ---------------------------------------------------------
# PREFILL
# ---------------------------------------------------------

state = engine.prefill(
    request
)

assert isinstance(
    state,
    InferenceState
)

print("Prefill state type: PASS")

assert state.prompt_tokens == tokens

print("Prompt preservation: PASS")

assert state.generated_tokens == []

print("Initial generated state: PASS")

assert state.last_logits is not None

assert len(state.last_logits) == 32768

print("Prefill logits: PASS")

assert model.cache_size() > 0

print("Prefill KV population: PASS")

# ---------------------------------------------------------
# DECODE
# ---------------------------------------------------------

token = engine.decode(
    state
)

print("First decode: PASS")

assert isinstance(
    token,
    int
)

assert 0 <= token < 32768

print("Decoded token validity: PASS")

assert len(
    state.generated_tokens
) == 1

print("Generated-state update: PASS")

assert state.last_logits is not None

print("Incremental logits update: PASS")

# ---------------------------------------------------------
# Continue decoding until finished
# ---------------------------------------------------------

while not state.finished:
    engine.decode(
        state
    )

print("Decode lifecycle completion: PASS")

assert state.finished

assert state.finish_reason in (
    "eos",
    "length",
)

print("Finish reason: PASS")

# ---------------------------------------------------------
# RESPONSE
# ---------------------------------------------------------

response = engine.response(
    state
)

assert isinstance(
    response,
    InferenceResponse
)

print("Response type: PASS")

assert response.prompt_tokens == tokens

print("Response prompt preservation: PASS")

assert response.tokens == (
    tokens
    + response.generated_tokens
)

print("Response token composition: PASS")

# ---------------------------------------------------------
# Cache cleanup
# ---------------------------------------------------------

engine._finish_request()

assert model.cache_size() == 0

print("Cache cleanup: PASS")

assert not engine.active

print("Request lifecycle cleanup: PASS")

# ---------------------------------------------------------
# BACKWARD COMPATIBILITY
# ---------------------------------------------------------

model2 = NeuraModel(
    config=NeuraTiny()
)

engine2 = InferenceEngine(
    model2
)

result = engine2.generate(
    InferenceRequest(
        prompt_tokens=tokens,
        max_tokens=2,
    )
)

assert isinstance(
    result,
    InferenceResponse
)

print(
    "generate() compatibility: PASS"
)

assert result.prompt_tokens == tokens

print(
    "generate() prompt preservation: PASS"
)

assert model2.cache_size() == 0

print(
    "generate() cache cleanup: PASS"
)

# ---------------------------------------------------------
# API contracts
# ---------------------------------------------------------

assert issubclass(
    InferenceError,
    RuntimeError,
)

print("InferenceError contract: PASS")

print("=" * 60)
print(
    "RESULT: PREFILL / DECODE RUNTIME CONTRACT PASSED"
)
print("=" * 60)
