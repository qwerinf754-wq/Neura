import json
import traceback
import torch
from neura.brain.sft_v2_inference import NeuraSFTV2Runtime

TRAIN = "data/processed/neura_chat_sft_v2/train.jsonl"

CHECKPOINTS = [
    ("010", "checkpoints/chat_sft_v2_repaired/steps/step_010.pt"),
    ("020", "checkpoints/chat_sft_v2_repaired/steps/step_020.pt"),
    ("050", "checkpoints/chat_sft_v2_repaired/steps/step_050.pt"),
    ("100", "checkpoints/chat_sft_v2_repaired/steps/step_100.pt"),
    ("200", "checkpoints/chat_sft_v2_repaired/steps/step_200.pt"),
    ("496", "checkpoints/chat_sft_v2_repaired/neura_chat_sft_best.pt"),
    ("569", "checkpoints/chat_sft_v2_600/neura_chat_sft_best.pt"),
]

with open(TRAIN, encoding="utf-8") as f:
    rows = [json.loads(x) for x in f if x.strip()]

def evaluate(ckpt):
    runtime = NeuraSFTV2Runtime(ckpt, device="cpu")
    runtime.model.eval()

    total_nll = 0.0
    total_tokens = 0
    top1 = top5 = top10 = 0

    with torch.no_grad():
        for row in rows:
            messages = row["messages"]

            system_msg = next(
                (m["content"] for m in messages if m["role"] == "system"),
                None,
            )

            user = next(
                m["content"] for m in messages
                if m["role"] == "user"
            )

            assistant = next(
                m["content"] for m in messages
                if m["role"] == "assistant"
            )

            context = {"history": []}

            if system_msg is not None:
                context["system"] = system_msg

            prompt = runtime.build_prompt(
                user,
                context=context,
            )

            prompt_ids = runtime.tokenizer.encode(
                prompt, add_bos=True, add_eos=False
            )

            target_ids = runtime.tokenizer.encode(
                assistant, add_bos=False, add_eos=False
            )

            x = torch.tensor(
                [prompt_ids + target_ids],
                dtype=torch.long,
            )

            logits = runtime.model(x)[0]

            for i, target_id in enumerate(target_ids):
                pos = len(prompt_ids) + i - 1
                scores = logits[pos]

                log_probs = torch.log_softmax(scores, dim=-1)
                total_nll += -log_probs[target_id].item()
                total_tokens += 1

                rank = int(
                    (scores > scores[target_id]).sum().item()
                ) + 1

                if rank <= 1:
                    top1 += 1
                if rank <= 5:
                    top5 += 1
                if rank <= 10:
                    top10 += 1

    return (
        total_nll / total_tokens,
        top1 / total_tokens,
        top5 / total_tokens,
        top10 / total_tokens,
        total_tokens,
    )

print("=" * 86)
print("NEURA — TRAIN CONTENT LEARNING CURVE")
print("=" * 86)
print(f"{'STEP':>6} {'NLL':>10} {'TOP1':>10} {'TOP5':>10} {'TOP10':>10}")
print("-" * 86)

for step, ckpt in CHECKPOINTS:
    print(f"{step:>6} loading...", flush=True)

    try:
        nll, a, b, c, tokens = evaluate(ckpt)

        print(
            f"{step:>6} "
            f"{nll:>10.4f} "
            f"{a*100:>9.2f}% "
            f"{b*100:>9.2f}% "
            f"{c*100:>9.2f}% "
            f"({tokens} tokens)"
        )

    except BaseException as e:
        print(f"{step:>6} ERROR {type(e).__name__}: {repr(e)}")
        traceback.print_exc()

print("=" * 86)
