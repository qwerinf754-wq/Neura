import json
import traceback
import torch
from neura.brain.sft_v2_inference import NeuraSFTV2Runtime

TRAIN = "data/processed/neura_chat_sft_v2/train.jsonl"
VAL = "data/processed/neura_chat_sft_v2/validation.jsonl"

CHECKPOINTS = [
    ("010", "checkpoints/chat_sft_v2_repaired/steps/step_010.pt"),
    ("020", "checkpoints/chat_sft_v2_repaired/steps/step_020.pt"),
    ("050", "checkpoints/chat_sft_v2_repaired/steps/step_050.pt"),
    ("100", "checkpoints/chat_sft_v2_repaired/steps/step_100.pt"),
    ("200", "checkpoints/chat_sft_v2_repaired/steps/step_200.pt"),
    ("496", "checkpoints/chat_sft_v2_repaired/neura_chat_sft_best.pt"),
    ("569", "checkpoints/chat_sft_v2_600/neura_chat_sft_best.pt"),
]

def load_rows(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]

train_rows = load_rows(TRAIN)
val_rows = load_rows(VAL)

def evaluate(runtime, rows):
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
                prompt,
                add_bos=True,
                add_eos=False,
            )

            target_ids = runtime.tokenizer.encode(
                assistant,
                add_bos=False,
                add_eos=False,
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

                if rank == 1:
                    top1 += 1
                if rank <= 5:
                    top5 += 1
                if rank <= 10:
                    top10 += 1

    return {
        "nll": total_nll / total_tokens,
        "top1": top1 / total_tokens,
        "top5": top5 / total_tokens,
        "top10": top10 / total_tokens,
        "tokens": total_tokens,
    }

print("=" * 115)
print("NEURA — TRAIN vs VALIDATION CONTENT LEARNING CURVE")
print("=" * 115)

print(
    f"{'STEP':>6} | "
    f"{'TRAIN NLL':>10} {'T1':>7} {'T5':>7} {'T10':>7} | "
    f"{'VAL NLL':>10} {'V1':>7} {'V5':>7} {'V10':>7}"
)

print("-" * 115)

for step, ckpt in CHECKPOINTS:
    print(f"{step:>6} | loading...", flush=True)

    try:
        runtime = NeuraSFTV2Runtime(ckpt, device="cpu")

        tr = evaluate(runtime, train_rows)
        va = evaluate(runtime, val_rows)

        print(
            f"{step:>6} | "
            f"{tr['nll']:>10.4f} "
            f"{tr['top1']*100:>6.2f}% "
            f"{tr['top5']*100:>6.2f}% "
            f"{tr['top10']*100:>6.2f}% | "
            f"{va['nll']:>10.4f} "
            f"{va['top1']*100:>6.2f}% "
            f"{va['top5']*100:>6.2f}% "
            f"{va['top10']*100:>6.2f}%"
        )

    except BaseException as e:
        print(
            f"{step:>6} | ERROR "
            f"{type(e).__name__}: {repr(e)}"
        )
        traceback.print_exc()

print("=" * 115)
