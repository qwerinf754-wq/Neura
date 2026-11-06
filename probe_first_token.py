import json
import torch

from neura.brain.sft_v2_inference import NeuraSFTV2Runtime

CKPT = "checkpoints/chat_sft_v2_600/neura_chat_sft_best.pt"
VAL = "data/processed/neura_chat_sft_v2/validation.jsonl"

rt = NeuraSFTV2Runtime(CKPT, device="cpu")
model = rt.model
tok = rt.tokenizer
model.eval()

with open(VAL, "r", encoding="utf-8") as f:
    rows = [json.loads(line) for line in f if line.strip()]

print("checkpoint:", CKPT)
print("validation rows:", len(rows))

for i, row in enumerate(rows):
    messages = row.get("messages", [])

    system = None
    history = []
    user_text = None
    assistant_text = None

    for msg in messages:
        role = msg.get("role")
        content = str(msg.get("content", ""))

        if role == "system" and system is None:
            system = content
        elif role == "user":
            user_text = content
        elif role == "assistant":
            assistant_text = content

    if user_text is None or assistant_text is None:
        print(f"[{i}] SKIP")
        continue

    # Previous conversational turns, excluding final user/assistant pair.
    for msg in messages:
        role = msg.get("role")
        content = str(msg.get("content", ""))

        if role in {"user", "assistant"} and content != user_text:
            history.append({
                "role": role,
                "content": content,
            })

    context = {
        "system": system,
        "history": history,
    }

    prompt = rt.build_prompt(
        user_text,
        context=context,
    )

    prompt_ids = tok.encode(
        prompt,
        add_bos=True,
        add_eos=False,
    )

    answer_ids = tok.encode(
        assistant_text,
        add_bos=False,
        add_eos=False,
    )

    if not answer_ids:
        print(f"[{i}] EMPTY ANSWER")
        continue

    input_tensor = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
    )

    with torch.no_grad():
        logits = model(input_tensor)

    last_logits = logits[0, -1]

    top_values, top_ids = torch.topk(
        last_logits,
        k=10,
    )

    target_id = answer_ids[0]
    target_logit = float(last_logits[target_id])
    target_rank = int(
        (last_logits > last_logits[target_id]).sum().item()
    ) + 1

    print()
    print("=" * 70)
    print("ROW:", i)
    print("USER:", repr(user_text))
    print("EXPECTED:", repr(assistant_text))
    print("PROMPT TOKENS:", len(prompt_ids))
    print("EXPECTED FIRST ID:", target_id)
    print(
        "EXPECTED FIRST TOKEN:",
        repr(tok.decode([target_id], skip_special_tokens=False))
    )
    print("EXPECTED LOGIT:", round(target_logit, 5))
    print("EXPECTED RANK:", target_rank)
    print()
    print("TOP 10:")

    for rank, (value, token_id) in enumerate(
        zip(top_values.tolist(), top_ids.tolist()),
        1,
    ):
        text = tok.decode(
            [token_id],
            skip_special_tokens=False,
        )
        print(
            f"{rank:2d}. "
            f"id={token_id:4d} "
            f"logit={value:9.4f} "
            f"text={text!r}"
        )
