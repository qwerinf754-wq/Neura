import json
from neura.brain.sft_v2_inference import NeuraSFTV2Runtime

VAL = "data/processed/neura_chat_sft_v2/validation.jsonl"

CHECKPOINTS = [
    ("496", "checkpoints/chat_sft_v2_repaired/neura_chat_sft_best.pt"),
    ("569", "checkpoints/chat_sft_v2_600/neura_chat_sft_best.pt"),
]

with open(VAL, encoding="utf-8") as f:
    rows = [json.loads(line) for line in f if line.strip()]


def get_turns(row):
    messages = row["messages"]

    system = next(
        (m["content"] for m in messages if m["role"] == "system"),
        None,
    )

    # نختبر آخر سؤال للمستخدم، ونمرر ما قبله من الحوار كسياق.
    last_user_index = max(
        i for i, m in enumerate(messages)
        if m["role"] == "user"
    )

    user = messages[last_user_index]["content"]

    history = [
        {"role": m["role"], "content": m["content"]}
        for m in messages[:last_user_index]
        if m["role"] in ("user", "assistant")
    ]

    return system, history, user


for label, checkpoint in CHECKPOINTS:
    print("\n" + "=" * 90)
    print("CHECKPOINT:", label)
    print(checkpoint)
    print("=" * 90, flush=True)

    runtime = NeuraSFTV2Runtime(checkpoint, device="cpu")

    for i, row in enumerate(rows, 1):
        system, history, user = get_turns(row)

        context = {"history": history}
        if system is not None:
            context["system"] = system

        print(f"\n[{i}] USER: {user}")

        try:
            answer = runtime.generate(
                user,
                context=context,
                route="chat",
                max_new_tokens=80,
            )
            print("NEURA:", repr(answer))
        except Exception as e:
            print("ERROR:", type(e).__name__, repr(e))

print("\nCOMPARISON COMPLETE")
