import torch

from neura.brain.sft_v2_inference import NeuraSFTV2Runtime

CKPT = "checkpoints/chat_sft_v2/neura_chat_sft_last.pt"

rt = NeuraSFTV2Runtime(CKPT, device="cpu")
model = rt.model
tok = rt.tokenizer

prompt = rt.build_prompt("مرحبا")

print("=== SFT GENERATION TRACE ===")
print("checkpoint:", CKPT)
print("step:", rt.step)
print("prompt:")
print(repr(prompt))

input_ids = tok.encode(
    prompt,
    add_bos=True,
    add_eos=False,
)

print("prompt token count:", len(input_ids))
print("last 20 prompt ids:", input_ids[-20:])

assistant_prefix_ids = tok.encode(
    "<|assistant|>\n",
    add_bos=False,
    add_eos=False,
)

print("assistant prefix ids:", assistant_prefix_ids)

with torch.no_grad():
    for step in range(20):
        x = torch.tensor(
            [input_ids],
            dtype=torch.long,
            device=rt.device,
        )

        logits = model(x)
        last_logits = logits[0, -1]

        probs = torch.softmax(last_logits, dim=-1)
        values, indices = torch.topk(probs, k=10)

        print()
        print(f"--- generation step {step + 1} ---")
        print("context length:", len(input_ids))

        for rank, (prob, idx) in enumerate(
            zip(values.tolist(), indices.tolist()),
            start=1,
        ):
            try:
                text = tok.decode(
                    [int(idx)],
                    skip_special_tokens=False,
                )
            except Exception as exc:
                text = f"<decode-error:{exc}>"

            print(
                f"{rank:2d}. "
                f"id={int(idx):4d} "
                f"prob={prob:.6f} "
                f"text={text!r}"
            )

        next_id = int(indices[0].item())
        input_ids.append(next_id)

        try:
            generated_text = tok.decode(
                input_ids[len(
                    tok.encode(
                        prompt,
                        add_bos=True,
                        add_eos=False,
                    )
                ):],
                skip_special_tokens=False,
            )
        except Exception as exc:
            generated_text = f"<decode-error:{exc}>"

        print("chosen id:", next_id)
        print("generated so far:", repr(generated_text))

        if next_id == tok.EOS_ID:
            print("EOS reached.")
            break

print()
print("=== FINAL GENERATED TOKENS ===")

prompt_ids = tok.encode(
    prompt,
    add_bos=True,
    add_eos=False,
)

generated_ids = input_ids[len(prompt_ids):]

print("ids:", generated_ids)

try:
    print(
        "text:",
        repr(
            tok.decode(
                generated_ids,
                skip_special_tokens=False,
            )
        ),
    )
except Exception as exc:
    print("decode error:", repr(exc))
