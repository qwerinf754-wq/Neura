import torch
from pathlib import Path

from neura.brain.sft_v2_inference import NeuraSFTV2Runtime

CHECKPOINTS = [
    "checkpoints/chat_sft_v2/neura_chat_sft_best.pt",
    "checkpoints/chat_sft_v2_repaired/neura_chat_sft_best.pt",
    "checkpoints/chat_sft_v2_repaired/neura_chat_sft_last.pt",
    "checkpoints/chat_sft_v2_600/neura_chat_sft_best.pt",
    "checkpoints/chat_sft_v2_600/neura_chat_sft_last.pt",
]

PROMPT = "مرحبا"

print("=== SFT CHECKPOINT COMPARISON ===")
print("prompt:", PROMPT)

for ckpt in CHECKPOINTS:
    print()
    print("=" * 80)
    print("CHECKPOINT:", ckpt)

    rt = NeuraSFTV2Runtime(ckpt, device="cpu")

    print("step:", rt.step)

    prompt = rt.build_prompt(PROMPT)

    input_ids = rt.tokenizer.encode(
        prompt,
        add_bos=True,
        add_eos=False,
    )

    with torch.no_grad():
        x = torch.tensor(
            [input_ids],
            dtype=torch.long,
            device=rt.device,
        )

        logits = rt.model(x)
        last_logits = logits[0, -1]

        values, indices = torch.topk(
            torch.softmax(last_logits, dim=-1),
            k=10,
        )

    print("prompt tokens:", len(input_ids))
    print("top-10 first assistant tokens:")

    for rank, (prob, idx) in enumerate(
        zip(values.tolist(), indices.tolist()),
        start=1,
    ):
        idx = int(idx)

        try:
            token_text = rt.tokenizer.decode(
                [idx],
                skip_special_tokens=False,
            )
        except Exception as exc:
            token_text = f"<decode-error:{exc}>"

        print(
            f"{rank:2d}. "
            f"id={idx:4d} "
            f"prob={prob:.8f} "
            f"text={token_text!r}"
        )

    print("greedy first token:", int(indices[0]))

print()
print("DONE")
