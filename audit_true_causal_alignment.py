import torch

from neura.training.sft_checkpoint import SFTCheckpoint
from neura.tokenizer import TOKENIZER
from neura.model.torch_model import NeuraTorchModel


CHECKPOINT = "checkpoints/neura_chat_sft_overfit_v1_best.pt"


def token_name(token_id):
    try:
        return TOKENIZER.vocab.decode(int(token_id))
    except Exception:
        return f"<ID:{token_id}>"


checkpoint = SFTCheckpoint().load_state(
    CHECKPOINT,
    map_location="cpu",
)

config = checkpoint["model_config"]
vocab_size = int(config["vocab_size"])

model = NeuraTorchModel(
    vocab_size=vocab_size,
    hidden_size=config["hidden_size"],
    layers=config["layers"],
    heads=config["heads"],
    context_length=config["context_length"],
    dropout=config["dropout"],
)

model.load_state_dict(
    checkpoint["model_state_dict"],
    strict=True,
)

model.eval()

from neura.chat import ChatTemplate

messages = [
    {
        "role": "system",
        "content": "You are Neura.",
    },
    {
        "role": "user",
        "content": "Hello",
    },
]

template = ChatTemplate()
serialized = template.render_for_generation(messages)

print("=" * 60)
print("TRUE CAUSAL ALIGNMENT AUDIT")
print("=" * 60)

print("SERIALIZED:")
print(repr(serialized))
print()

input_ids = TOKENIZER.encode(serialized)

print("IDS:")
print(input_ids)
print()

print("TOKENS:")
for i, token_id in enumerate(input_ids):
    print(
        f"{i:02d}: "
        f"id={token_id:02d} "
        f"{token_name(token_id)!r}"
    )

x = torch.tensor(
    [input_ids],
    dtype=torch.long,
)

with torch.no_grad():
    output = model(x)

if hasattr(output, "logits"):
    logits = output.logits
elif isinstance(output, dict):
    logits = output["logits"]
elif torch.is_tensor(output):
    logits = output
else:
    raise TypeError(
        f"unsupported model output type: {type(output)}"
    )

print()
print("=" * 60)
print("CAUSAL PAIRS")
print("=" * 60)

# The actual training relationship:
#
# logits[t] -> labels[t + 1]
#
for t in range(len(input_ids) - 1):

    target_id = input_ids[t + 1]

    probs = torch.softmax(
        logits[0, t],
        dim=-1,
    )

    top_probs, top_ids = torch.topk(
        probs,
        k=min(5, vocab_size),
    )

    print()
    print(f"POSITION {t}")
    print(
        "input :",
        repr(token_name(input_ids[t])),
    )
    print(
        "target:",
        f"id={target_id}",
        repr(token_name(target_id)),
    )

    print("TOP 5:")

    for rank, (p, token_id) in enumerate(
        zip(top_probs.tolist(), top_ids.tolist()),
        1,
    ):
        print(
            f"{rank}. "
            f"id={token_id:02d} "
            f"p={p:.6f} "
            f"token={token_name(token_id)!r}"
        )

    print(
        "TARGET PROBABILITY:",
        float(probs[target_id]),
    )

print()
print("=" * 60)
print("CRITICAL GENERATION POSITION")
print("=" * 60)

assistant_positions = [
    i
    for i, token_id in enumerate(input_ids)
    if token_name(token_id) == "<|assistant|>"
]

if assistant_positions:
    pos = assistant_positions[-1]
    probs = torch.softmax(
        logits[0, pos],
        dim=-1,
    )

    top_probs, top_ids = torch.topk(
        probs,
        k=min(10, vocab_size),
    )

    print("assistant position:", pos)

    for rank, (p, token_id) in enumerate(
        zip(top_probs.tolist(), top_ids.tolist()),
        1,
    ):
        print(
            f"{rank:02d}. "
            f"id={token_id:02d} "
            f"p={p:.6f} "
            f"token={token_name(token_id)!r}"
        )
