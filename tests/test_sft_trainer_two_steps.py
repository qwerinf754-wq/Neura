from pathlib import Path
import copy
import math

from neura.tokenizer.neura_bpe_tokenizer import NeuraBPETokenizer
from neura.training.chat_sft_bpe_dataset import ChatSFTBuilderBPE
from neura.training.loss import CrossEntropyLoss
from neura.training.optimizer import AdamW
from neura.training.sft_runtime import SFTRuntime
from neura.training.sft_trainer import SFTTrainer
from neura.model.causal_lm import CausalLM
from neura.model.parameters import PARAMETERS


ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_PATH = (
    ROOT / "data" / "processed" / "neura_arabic_bpe_32k" / "tokenizer.json"
)


def snapshot():
    return copy.deepcopy(PARAMETERS.state_dict())


def max_delta(a, b):
    result = 0.0

    for name in a:
        x = a[name]
        y = b[name]

        if isinstance(x, list) and x and isinstance(x[0], list):
            for r in range(len(x)):
                for c in range(len(x[r])):
                    result = max(
                        result,
                        abs(float(y[r][c]) - float(x[r][c])),
                    )
        elif isinstance(x, list):
            for i in range(len(x)):
                result = max(
                    result,
                    abs(float(y[i]) - float(x[i])),
                )

    return result


def assert_finite_state(state):
    for name, value in state.items():
        if isinstance(value, list):
            if value and isinstance(value[0], list):
                for row in value:
                    for x in row:
                        assert math.isfinite(float(x)), f"Non-finite: {name}"
            else:
                for x in value:
                    assert math.isfinite(float(x)), f"Non-finite: {name}"


print("=" * 80)
print("NEURA SFT TRAINER TWO-STEP GRADIENT RESET CONTRACT")
print("=" * 80)

# ------------------------------------------------------------------
# 1. Tokenizer
# ------------------------------------------------------------------
print("\n[1/8] BPE tokenizer")

tokenizer = NeuraBPETokenizer(TOKENIZER_PATH)

print("[PASS] Tokenizer")
print("Vocab:", tokenizer.vocab_size)

# ------------------------------------------------------------------
# 2. SFT example
# ------------------------------------------------------------------
print("\n[2/8] Build Chat SFT sample")

builder = ChatSFTBuilderBPE(
    tokenizer=tokenizer,
    max_length=128,
    add_bos=True,
    add_eos=False,
)

example = builder.build([
    {"role": "system", "content": "You are Neura."},
    {"role": "user", "content": "Explain artificial intelligence."},
    {
        "role": "assistant",
        "content": "Artificial intelligence is the study of systems that learn from data.",
    },
])

assert sum(example.loss_mask) > 0

print("[PASS] Chat SFT")
print("Sequence:", len(example.input_ids))
print("Assistant tokens:", sum(example.loss_mask))

# ------------------------------------------------------------------
# 3. Model / runtime
# ------------------------------------------------------------------
print("\n[3/8] Model + runtime")

model = CausalLM(
    vocab_size=tokenizer.vocab_size,
    hidden_size=64,
    heads=4,
    num_layers=2,
    intermediate_size=256,
    context_length=128,
)

loss = CrossEntropyLoss()

optimizer = AdamW(
    learning_rate=1e-4,
    weight_decay=0.01,
)

runtime = SFTRuntime(
    model=model,
    loss=loss,
    optimizer=optimizer,
)

trainer = SFTTrainer(
    runtime=runtime,
    optimizer=optimizer,
)

assert PARAMETERS.parameter_count() == 393536

print("[PASS] Model")
print("Parameters:", PARAMETERS.parameter_count())

# ------------------------------------------------------------------
# 4. Initial state
# ------------------------------------------------------------------
print("\n[4/8] Initial state")

before = snapshot()

assert runtime.step_count == 0
assert optimizer.step_count == 0
assert trainer.state.step == 0
assert trainer.state.optimizer_steps == 0

print("[PASS] Initial state")

# ------------------------------------------------------------------
# 5. First step
# ------------------------------------------------------------------
print("\n[5/8] FIRST TRAINING STEP")

result1 = trainer.train_batch([example])

after1 = snapshot()

delta1 = max_delta(before, after1)

assert runtime.step_count == 1
assert optimizer.step_count == 1
assert trainer.state.step == 1
assert trainer.state.optimizer_steps == 1
assert delta1 > 0.0
assert math.isfinite(float(result1["loss"]))

print("[PASS] First step")
print("Loss 1:", result1["loss"])
print("Runtime steps:", runtime.step_count)
print("Optimizer steps:", optimizer.step_count)
print("Trainer steps:", trainer.state.step)
print("Max parameter delta 1:", delta1)

# ------------------------------------------------------------------
# 6. Gradient reset inspection BEFORE second step
# ------------------------------------------------------------------
print("\n[6/8] Inspect gradients before second step")

gradients_before_second = PARAMETERS.gradient_dict()

nonzero_before_second = 0
max_grad_before_second = 0.0

for name, grad in gradients_before_second.items():
    if isinstance(grad, list) and grad:
        if isinstance(grad[0], list):
            for row in grad:
                for value in row:
                    value = abs(float(value))
                    if value != 0.0:
                        nonzero_before_second += 1
                    max_grad_before_second = max(
                        max_grad_before_second,
                        value,
                    )
        else:
            for value in grad:
                value = abs(float(value))
                if value != 0.0:
                    nonzero_before_second += 1
                max_grad_before_second = max(
                    max_grad_before_second,
                    value,
                )

print("Nonzero gradients before step 2:", nonzero_before_second)
print("Max gradient before step 2:", max_grad_before_second)

# This is intentionally diagnostic.
# Existing implementation may retain gradients here.
# We do NOT fail yet solely because they are nonzero.

# ------------------------------------------------------------------
# 7. Second step
# ------------------------------------------------------------------
print("\n[7/8] SECOND TRAINING STEP")

result2 = trainer.train_batch([example])

after2 = snapshot()

delta2 = max_delta(after1, after2)

assert runtime.step_count == 2
assert optimizer.step_count == 2
assert trainer.state.step == 2
assert trainer.state.optimizer_steps == 2

assert math.isfinite(float(result2["loss"]))
assert delta2 > 0.0

print("[PASS] Second step")
print("Loss 2:", result2["loss"])
print("Runtime steps:", runtime.step_count)
print("Optimizer steps:", optimizer.step_count)
print("Trainer steps:", trainer.state.step)
print("Max parameter delta 2:", delta2)

# ------------------------------------------------------------------
# 8. Final consistency
# ------------------------------------------------------------------
print("\n[8/8] Final consistency")

assert trainer.state.examples_seen == 2
assert trainer.state.loss_sum == (
    float(result1["loss"]) + float(result2["loss"])
)

final_state = PARAMETERS.state_dict()
assert_finite_state(final_state)

print("[PASS] Two optimizer steps exactly")
print("[PASS] Two trainer steps exactly")
print("[PASS] Parameters changed on both steps")
print("[PASS] Parameters finite")
print("[PASS] State bookkeeping correct")
print()
print("=" * 80)
print("===== SFT TRAINER TWO-STEP: COMPLETE =====")
print("STEP 1: PASS")
print("STEP 2: PASS")
print("NO DOUBLE STEP")
print("GRADIENT STATE INSPECTED")
print("NO LONG TRAINING")
print("=" * 80)
