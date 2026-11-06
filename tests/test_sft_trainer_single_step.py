from pathlib import Path
import math
import copy

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


def assert_finite_state(state):
    for name, value in state.items():
        if isinstance(value, list):
            for row in value:
                if isinstance(row, list):
                    for x in row:
                        assert math.isfinite(float(x)), f"Non-finite: {name}"
                else:
                    assert math.isfinite(float(row)), f"Non-finite: {name}"


print("=" * 80)
print("NEURA SFT TRAINER SINGLE STEP CONTRACT")
print("=" * 80)

# ------------------------------------------------------------------
# 1. Tokenizer
# ------------------------------------------------------------------
print("\n[1/9] BPE tokenizer")

tokenizer = NeuraBPETokenizer(TOKENIZER_PATH)

print("[PASS] Tokenizer")
print("Vocab:", tokenizer.vocab_size)

# ------------------------------------------------------------------
# 2. Chat SFT example
# ------------------------------------------------------------------
print("\n[2/9] Build Chat SFT sample")

builder = ChatSFTBuilderBPE(
    tokenizer=tokenizer,
    max_length=128,
    add_bos=True,
    add_eos=False,
)

example = builder.build([
    {"role": "system", "content": "You are Neura."},
    {"role": "user", "content": "What is artificial intelligence?"},
    {
        "role": "assistant",
        "content": "Artificial intelligence is a field of computer science.",
    },
])

assert len(example.input_ids) == len(example.labels)
assert len(example.input_ids) == len(example.loss_mask)
assert sum(example.loss_mask) > 0

print("[PASS] Chat SFT")
print("Sequence:", len(example.input_ids))
print("Assistant tokens:", sum(example.loss_mask))

# ------------------------------------------------------------------
# 3. Model
# ------------------------------------------------------------------
print("\n[3/9] CausalLM")

model = CausalLM(
    vocab_size=tokenizer.vocab_size,
    hidden_size=64,
    heads=4,
    num_layers=2,
    intermediate_size=256,
    context_length=128,
)

print("[PASS] CausalLM")
print("Parameters:", PARAMETERS.parameter_count())
print("Registry entries:", len(PARAMETERS.names()))

# ------------------------------------------------------------------
# 4. Optimizer + runtime
# ------------------------------------------------------------------
print("\n[4/9] Runtime + optimizer")

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

assert runtime.step_count == 0
assert optimizer.step_count == 0
assert trainer.state.step == 0
assert trainer.state.optimizer_steps == 0

print("[PASS] Initial state")
print("Runtime step:", runtime.step_count)
print("Optimizer steps:", optimizer.step_count)
print("Trainer step:", trainer.state.step)

# ------------------------------------------------------------------
# 5. Initial weights
# ------------------------------------------------------------------
print("\n[5/9] Snapshot initial parameters")

before = copy.deepcopy(PARAMETERS.state_dict())

assert before
assert_finite_state(before)

print("[PASS] Initial parameters finite")
print("Parameter tensors:", len(before))

# ------------------------------------------------------------------
# 6. Trainer single step
# ------------------------------------------------------------------
print("\n[6/9] SFTTrainer.train_batch()")

result = trainer.train_batch([example])

print("[PASS] train_batch()")
print("Result:", result)

# ------------------------------------------------------------------
# 7. Exactly one step
# ------------------------------------------------------------------
print("\n[7/9] Verify exactly ONE optimization step")

assert runtime.step_count == 1, (
    f"Expected runtime.step_count=1, got {runtime.step_count}"
)

assert optimizer.step_count == 1, (
    f"Expected optimizer.step_count=1, got {optimizer.step_count}"
)

assert trainer.state.step == 1, (
    f"Expected trainer.state.step=1, got {trainer.state.step}"
)

assert trainer.state.optimizer_steps == 1, (
    "Expected trainer.state.optimizer_steps=1, "
    f"got {trainer.state.optimizer_steps}"
)

print("[PASS] Runtime step count:", runtime.step_count)
print("[PASS] Optimizer step count:", optimizer.step_count)
print("[PASS] Trainer step:", trainer.state.step)
print("[PASS] Trainer optimizer steps:", trainer.state.optimizer_steps)

# ------------------------------------------------------------------
# 8. Verify weights changed exactly through the step
# ------------------------------------------------------------------
print("\n[8/9] Verify parameter update")

after = PARAMETERS.state_dict()

changed = 0
max_delta = 0.0

for name in before:
    old = before[name]
    new = after[name]

    if isinstance(old, list):
        if old and isinstance(old[0], list):
            for r in range(len(old)):
                for c in range(len(old[r])):
                    delta = abs(float(new[r][c]) - float(old[r][c]))
                    if delta > 0.0:
                        changed += 1
                    max_delta = max(max_delta, delta)
        else:
            for i in range(len(old)):
                delta = abs(float(new[i]) - float(old[i]))
                if delta > 0.0:
                    changed += 1
                max_delta = max(max_delta, delta)

assert changed > 0, "No parameters changed"
print("[PASS] Parameters changed")
print("Changed scalar parameters:", changed)
print("Max absolute delta:", max_delta)

# ------------------------------------------------------------------
# 9. Final consistency
# ------------------------------------------------------------------
print("\n[9/9] Final state")

assert math.isfinite(float(result["loss"]))
assert result["step"] == 1
assert result["optimizer_steps"] == 1
assert result["loss"] > 0.0

after = PARAMETERS.state_dict()
assert_finite_state(after)

assert trainer.state.examples_seen == 1
assert trainer.state.loss_sum == result["loss"]

print("[PASS] Loss finite")
print("[PASS] Parameters finite")
print("[PASS] Examples seen:", trainer.state.examples_seen)
print("[PASS] Loss bookkeeping")
print()
print("=" * 80)
print("===== SFT TRAINER SINGLE STEP: PASS =====")
print("BPE PASS")
print("CHAT SFT PASS")
print("RUNTIME PASS")
print("TRAINER PASS")
print("EXACTLY ONE OPTIMIZER STEP PASS")
print("WEIGHTS CHANGED PASS")
print("STATE BOOKKEEPING PASS")
print("NO DOUBLE STEP")
print("NO CHECKPOINT SAVE")
print("NO LONG TRAINING")
print("=" * 80)
