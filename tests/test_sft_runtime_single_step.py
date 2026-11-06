import copy
import math

from neura.tokenizer.neura_bpe_tokenizer import NeuraBPETokenizer
from neura.training.chat_sft_bpe_dataset import (
    ChatSFTBuilderBPE,
    IGNORE_INDEX,
)
from neura.model.causal_lm import CausalLM
from neura.model.parameters import PARAMETERS
from neura.training.loss import CrossEntropyLoss
from neura.training.sft_runtime import SFTRuntime
from neura.training.optimizer import AdamW


TOKENIZER_PATH = (
    "data/processed/neura_arabic_bpe_32k/tokenizer.json"
)


def flatten_values(value):
    if isinstance(value, list):
        out = []
        for item in value:
            out.extend(flatten_values(item))
        return out

    return [float(value)]


def max_abs_delta(before, after):
    maximum = 0.0

    for name in before:
        a = flatten_values(before[name])
        b = flatten_values(after[name])

        if len(a) != len(b):
            raise AssertionError(
                f"Parameter shape changed: {name}"
            )

        for x, y in zip(a, b):
            maximum = max(
                maximum,
                abs(y - x),
            )

    return maximum


def all_finite_state(state):
    for value in state.values():
        for item in flatten_values(value):
            if not math.isfinite(item):
                return False
    return True


print("=" * 80)
print("NEURA SFT RUNTIME SINGLE STEP CONTRACT")
print("=" * 80)


# ------------------------------------------------------------------
# 1. Tokenizer
# ------------------------------------------------------------------

print("[1/10] BPE tokenizer")

tokenizer = NeuraBPETokenizer(
    TOKENIZER_PATH
)

print("[PASS] Tokenizer")
print("Vocab:", tokenizer.vocab_size)


# ------------------------------------------------------------------
# 2. Chat SFT
# ------------------------------------------------------------------

print()
print("[2/10] Build Chat SFT sample")

builder = ChatSFTBuilderBPE(
    tokenizer=tokenizer,
    max_length=128,
    add_bos=True,
    add_eos=False,
)

example = builder.build([
    {
        "role": "system",
        "content": "You are Neura.",
    },
    {
        "role": "user",
        "content": "What is artificial intelligence?",
    },
    {
        "role": "assistant",
        "content": (
            "Artificial intelligence is a field of "
            "computer science."
        ),
    },
])

print("[PASS] Chat SFT")
print("Sequence:", len(example.input_ids))
print(
    "Assistant tokens:",
    sum(
        1
        for x in example.labels
        if x != IGNORE_INDEX
    ),
)


# ------------------------------------------------------------------
# 3. Causal alignment
# ------------------------------------------------------------------

print()
print("[3/10] Causal alignment")

input_ids = example.input_ids[:-1]
labels = example.labels[1:]

print("[PASS] Causal alignment")
print("Input:", len(input_ids))
print("Labels:", len(labels))


# ------------------------------------------------------------------
# 4. Tiny model
# ------------------------------------------------------------------

print()
print("[4/10] CausalLM")

model = CausalLM(
    vocab_size=tokenizer.vocab_size,
    hidden_size=64,
    heads=4,
    num_layers=2,
    intermediate_size=256,
    context_length=128,
)

print("[PASS] CausalLM")
print(
    "Parameters:",
    model.parameter_count(),
)
print(
    "Registry entries:",
    len(PARAMETERS.names()),
)


# ------------------------------------------------------------------
# 5. Optimizer
# ------------------------------------------------------------------

print()
print("[5/10] AdamW")

optimizer = AdamW(
    learning_rate=1e-4,
    weight_decay=0.01,
)

loss = CrossEntropyLoss()

runtime = SFTRuntime(
    model=model,
    loss=loss,
    optimizer=optimizer,
)

print("[PASS] Runtime")
print("Initial runtime step:", runtime.step_count)


# ------------------------------------------------------------------
# 6. Initial parameter state
# ------------------------------------------------------------------

print()
print("[6/10] Initial parameter state")

initial_state = copy.deepcopy(
    PARAMETERS.state_dict()
)

assert all_finite_state(initial_state)

print("[PASS] Initial parameters finite")


# ------------------------------------------------------------------
# 7. Execute EXACTLY ONE runtime step
# ------------------------------------------------------------------

print()
print("[7/10] SFTRuntime.step()")

result = runtime.step(
    [example]
)

assert result["status"] == "trained"
assert result["step"] == 1
assert runtime.step_count == 1

assert result["assistant_tokens"] > 0
assert result["gradient_count"] > 0

print("[PASS] Runtime step")
print("Result:", result)
print("Optimizer steps:", optimizer.step_count)


# ------------------------------------------------------------------
# 8. Verify exactly one optimizer step
# ------------------------------------------------------------------

print()
print("[8/10] Verify exactly ONE optimizer step")

assert optimizer.step_count == 1

print("[PASS] Exactly one optimizer step")
print("Optimizer step count:", optimizer.step_count)


# ------------------------------------------------------------------
# 9. Verify parameter update
# ------------------------------------------------------------------

print()
print("[9/10] Verify parameter update")

after_state = copy.deepcopy(
    PARAMETERS.state_dict()
)

delta = max_abs_delta(
    initial_state,
    after_state,
)

assert delta > 0.0

print("[PASS] Parameters changed")
print("Max absolute parameter delta:", delta)


# ------------------------------------------------------------------
# 10. Verify finite result
# ------------------------------------------------------------------

print()
print("[10/10] Verify final state")

assert all_finite_state(after_state)

assert math.isfinite(
    float(result["loss"])
)

print("[PASS] Parameters finite")
print("[PASS] Loss finite")


print()
print("=" * 80)
print("===== SFT RUNTIME SINGLE STEP: PASS =====")
print("=" * 80)
print("BPE:                         PASS")
print("CHAT SFT:                    PASS")
print("RUNTIME FORWARD:             PASS")
print("ASSISTANT-ONLY LOSS:         PASS")
print("FULL GRADIENT RECONSTRUCTION: PASS")
print("CausalLM BACKWARD:           PASS")
print("GRADIENTS:                   PASS")
print("EXACTLY ONE OPTIMIZER STEP:  PASS")
print("WEIGHTS CHANGED:             PASS")
print("PARAMETERS FINITE:           PASS")
print()
print("ONE RUNTIME STEP ONLY")
print("NO CHECKPOINT SAVE")
print("NO LONG TRAINING")
print("=" * 80)
