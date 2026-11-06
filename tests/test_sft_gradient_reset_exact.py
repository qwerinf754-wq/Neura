from pathlib import Path
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


def gradient_abs_sum():
    total = 0.0
    nonzero = 0

    for grad in PARAMETERS.gradient_dict().values():
        if isinstance(grad, list) and grad:
            if isinstance(grad[0], list):
                for row in grad:
                    for x in row:
                        value = abs(float(x))
                        total += value
                        if value != 0.0:
                            nonzero += 1
            else:
                for x in grad:
                    value = abs(float(x))
                    total += value
                    if value != 0.0:
                        nonzero += 1

    return total, nonzero


print("=" * 80)
print("NEURA EXACT GRADIENT RESET CONTRACT")
print("=" * 80)

tokenizer = NeuraBPETokenizer(TOKENIZER_PATH)

builder = ChatSFTBuilderBPE(
    tokenizer=tokenizer,
    max_length=128,
    add_bos=True,
    add_eos=False,
)

example = builder.build([
    {"role": "system", "content": "You are Neura."},
    {"role": "user", "content": "Explain AI."},
    {
        "role": "assistant",
        "content": "Artificial intelligence is a field of computer science.",
    },
])

model = CausalLM(
    vocab_size=tokenizer.vocab_size,
    hidden_size=64,
    heads=4,
    num_layers=2,
    intermediate_size=256,
    context_length=128,
)

optimizer = AdamW(
    learning_rate=1e-4,
    weight_decay=0.01,
)

runtime = SFTRuntime(
    model=model,
    loss=CrossEntropyLoss(),
    optimizer=optimizer,
)

trainer = SFTTrainer(
    runtime=runtime,
    optimizer=optimizer,
)

print("\n[1/6] Initial gradient state")

total0, nonzero0 = gradient_abs_sum()

print("Gradient abs sum:", total0)
print("Nonzero gradients:", nonzero0)

assert total0 == 0.0
assert nonzero0 == 0

print("[PASS] Initial gradients are zero")

print("\n[2/6] First training step")

result1 = trainer.train_batch([example])

total1, nonzero1 = gradient_abs_sum()

print("Loss 1:", result1["loss"])
print("Gradient abs sum after step 1:", total1)
print("Nonzero gradients after step 1:", nonzero1)

assert nonzero1 > 0
assert total1 > 0.0

print("[PASS] First step produced gradients")

print("\n[3/6] Instrument optimizer.zero_grad()")

original_zero_grad = optimizer.zero_grad

observed = {
    "called": False,
    "before_sum": None,
    "before_nonzero": None,
    "after_sum": None,
    "after_nonzero": None,
}


def traced_zero_grad(parameters):
    before_sum, before_nonzero = gradient_abs_sum()

    observed["called"] = True
    observed["before_sum"] = before_sum
    observed["before_nonzero"] = before_nonzero

    result = original_zero_grad(parameters)

    after_sum, after_nonzero = gradient_abs_sum()

    observed["after_sum"] = after_sum
    observed["after_nonzero"] = after_nonzero

    return result


optimizer.zero_grad = traced_zero_grad

print("[PASS] zero_grad instrumentation installed")

print("\n[4/6] Second training step")

result2 = trainer.train_batch([example])

print("Loss 2:", result2["loss"])
print("zero_grad called:", observed["called"])
print("Gradients BEFORE zero_grad:", observed["before_nonzero"])
print("Gradient sum BEFORE zero_grad:", observed["before_sum"])
print("Gradients AFTER zero_grad:", observed["after_nonzero"])
print("Gradient sum AFTER zero_grad:", observed["after_sum"])

assert observed["called"] is True

# There must have been stale gradients from step 1.
assert observed["before_nonzero"] > 0
assert observed["before_sum"] > 0.0

# zero_grad must clear them before the second forward/backward.
assert observed["after_nonzero"] == 0
assert observed["after_sum"] == 0.0

print("[PASS] zero_grad was called at start of step 2")
print("[PASS] Previous gradients were present before reset")
print("[PASS] Previous gradients were completely cleared")

print("\n[5/6] Verify exactly two optimizer steps")

assert runtime.step_count == 2
assert optimizer.step_count == 2
assert trainer.state.step == 2
assert trainer.state.optimizer_steps == 2

print("[PASS] Runtime steps:", runtime.step_count)
print("[PASS] Optimizer steps:", optimizer.step_count)
print("[PASS] Trainer steps:", trainer.state.step)

print("\n[6/6] Final gradient state")

total_final, nonzero_final = gradient_abs_sum()

print("Final gradient abs sum:", total_final)
print("Final nonzero gradients:", nonzero_final)

# Gradients should exist after the second backward.
assert nonzero_final > 0
assert total_final > 0.0

print("[PASS] Second backward produced fresh gradients")

print()
print("=" * 80)
print("===== EXACT GRADIENT RESET: PASS =====")
print("STEP 1 GRADIENTS: PASS")
print("ZERO_GRAD CALLED: PASS")
print("OLD GRADIENTS CLEARED: PASS")
print("STEP 2 GRADIENTS FRESH: PASS")
print("EXACTLY TWO OPTIMIZER STEPS: PASS")
print("NO LONG TRAINING")
print("=" * 80)
