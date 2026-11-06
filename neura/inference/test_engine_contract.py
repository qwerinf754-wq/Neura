"""
Neura Inference Engine Contract
"""

from neura.tokenizer import VOCAB, TOKENIZER
from neura.model.model import NeuraModel
from neura.model.config import NeuraTiny
from neura.generation.generator import Generator
from neura.inference.engine import (
    InferenceEngine,
    InferenceRequest,
)


print("=" * 60)
print("NEURA REAL INFERENCE ENGINE CONTRACT")
print("=" * 60)


# ---------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------

VOCAB.ensure_size(32768)
TOKENIZER.unfreeze()

prompt = "Hello Neura"

tokens = TOKENIZER.encode(
    prompt
)

assert tokens
print("Prompt tokenization: PASS")


# ---------------------------------------------------------
# Model
# ---------------------------------------------------------

model = NeuraModel(
    config=NeuraTiny()
)

print("Model creation: PASS")


# ---------------------------------------------------------
# Generator
# ---------------------------------------------------------

generator = Generator(
    model
)

print("Generator creation: PASS")


# ---------------------------------------------------------
# Engine
# ---------------------------------------------------------

engine = InferenceEngine(
    model=model,
    generator=generator,
    tokenizer=TOKENIZER,
)

print("InferenceEngine creation: PASS")


# ---------------------------------------------------------
# Request
# ---------------------------------------------------------

request = InferenceRequest(
    tokens=tokens,
    max_tokens=8,
)

print("Request creation: PASS")


# ---------------------------------------------------------
# Generation
# ---------------------------------------------------------

response = engine.generate(
    request
)

print(
    "Prompt tokens:",
    response.prompt_tokens,
)

print(
    "Generated tokens:",
    response.generated_tokens,
)

print(
    "Finish reason:",
    response.finish_reason,
)


# ---------------------------------------------------------
# Prompt preservation
# ---------------------------------------------------------

assert (
    response.prompt_tokens
    == tokens
)

print(
    "Prompt preservation: PASS"
)


# ---------------------------------------------------------
# Token validity
# ---------------------------------------------------------

assert all(
    isinstance(token, int)
    for token in response.generated_tokens
)

assert all(
    0 <= token < TOKENIZER.vocab_size
    for token in response.generated_tokens
)

print(
    "Generated token validity: PASS"
)


# ---------------------------------------------------------
# Blocked token protection
# ---------------------------------------------------------

blocked = set(
    VOCAB.blocked_token_ids()
)

assert not any(
    token in blocked
    for token in response.generated_tokens
)

print(
    "Blocked-token protection: PASS"
)


# ---------------------------------------------------------
# EOS
# ---------------------------------------------------------

if 3 in response.generated_tokens:

    eos_index = (
        response.generated_tokens.index(3)
    )

    assert all(
        token == 3
        for token in
        response.generated_tokens[
            eos_index:
        ]
    )

    assert (
        response.finish_reason
        == "eos"
    )

    print(
        "EOS termination: PASS"
    )

else:

    assert (
        response.finish_reason
        == "length"
    )

    print(
        "Length termination: PASS"
    )


# ---------------------------------------------------------
# Cache lifecycle
# ---------------------------------------------------------

engine.clear()

assert (
    model.kv_cache.total_size()
    == 0
)

print(
    "Cache cleanup: PASS"
)


# ---------------------------------------------------------
# Request isolation
# ---------------------------------------------------------

second_tokens = TOKENIZER.encode(
    "Second request"
)

second_request = InferenceRequest(
    tokens=second_tokens,
    max_tokens=4,
)

second_response = engine.generate(
    second_request
)

assert (
    second_response.prompt_tokens
    == second_tokens
)

print(
    "Second request isolation: PASS"
)

engine.clear()

assert (
    model.kv_cache.total_size()
    == 0
)

print(
    "Final cache cleanup: PASS"
)


print("=" * 60)
print(
    "RESULT: REAL INFERENCE ENGINE CONTRACT PASSED"
)
print("=" * 60)
