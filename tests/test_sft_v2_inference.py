from pathlib import Path

from neura.brain.sft_v2_inference import (
    EXPECTED_VERSION,
    NeuraSFTV2Runtime,
)


CHECKPOINT = Path(
    "checkpoints/chat_sft_v2/neura_chat_sft_last.pt"
)


def main():
    print("=" * 70)
    print("SFT V2 INFERENCE TEST")
    print("=" * 70)

    assert CHECKPOINT.exists(), (
        f"missing checkpoint: {CHECKPOINT}"
    )

    runtime = NeuraSFTV2Runtime(CHECKPOINT)

    assert runtime.model is not None
    assert runtime.model.training is False
    assert runtime.tokenizer.vocab_size == 2048
    assert runtime.context_length == 512
    assert EXPECTED_VERSION == "chat-sft-v2-pytorch"

    prompt = runtime.build_prompt(
        "مرحبا",
        context={
            "system": "أنت مساعد مفيد.",
        },
    )

    assert "<|system|>\n" in prompt
    assert "<|user|>\n" in prompt
    assert "<|assistant|>\n" in prompt
    assert "\n<|end|>\n" in prompt

    result = runtime.generate(
        "مرحبا",
        context={
            "system": "أنت مساعد مفيد.",
        },
        max_new_tokens=4,
    )

    assert isinstance(result, str)

    print("checkpoint:", CHECKPOINT)
    print("step:", runtime.step)
    print("device:", runtime.device)
    print("vocab_size:", runtime.tokenizer.vocab_size)
    print("context_length:", runtime.context_length)
    print("generated:", repr(result))
    print("SFT V2 INFERENCE: PASS")


if __name__ == "__main__":
    main()
