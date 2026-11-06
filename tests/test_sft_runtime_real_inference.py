from __future__ import annotations

from pathlib import Path

from neura.inference.sft_runtime import SFTRuntime


CHECKPOINT = Path("checkpoints/neura_chat_sft_best.pt")


print("=== REAL SFT INFERENCE TEST ===")

assert CHECKPOINT.exists(), (
    f"Missing checkpoint: {CHECKPOINT}"
)

runtime = SFTRuntime(
    checkpoint_path=CHECKPOINT,
    device="cpu",
)

assert runtime.loaded
assert runtime.model is not None
assert runtime.tokenizer is not None

print("CHECKPOINT LOAD: PASS")
print("VOCAB SIZE:", runtime.vocab_size)
print("CONTEXT LENGTH:", runtime.context_length)

prompt = "Hello Neura"

tokens = runtime.encode(prompt)

assert tokens
assert all(
    isinstance(token, int)
    and 0 <= token < runtime.vocab_size
    for token in tokens
)

print("TOKENIZATION: PASS")
print("PROMPT TOKENS:", len(tokens))

result = runtime.generate(
    prompt,
    max_new_tokens=4,
)

assert result.prompt_tokens == tokens
assert len(result.generated_tokens) <= 4
assert result.finish_reason in {
    "eos",
    "length",
}

generated_text = result.text

assert isinstance(generated_text, str)

print("REAL GENERATION: PASS")
print("GENERATED TOKENS:", len(result.generated_tokens))
print("GENERATED TEXT:", repr(generated_text))
print("FINISH:", result.finish_reason)

print("=== REAL SFT INFERENCE: PASS ===")
