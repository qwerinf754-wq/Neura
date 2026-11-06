import torch

from neura.training.sft_checkpoint import SFTCheckpoint
from neura.tokenizer import TOKENIZER
from neura.model.torch_model import NeuraTorchModel


CHECKPOINT = (
    "checkpoints/neura_chat_sft_overfit_v1_best.pt"
)


def token_name(token_id):
    try:
        return TOKENIZER.vocab.decode(int(token_id))
    except Exception:
        return f"<ID:{token_id}>"


print("=" * 60)
print("NEURA CHAT SFT GENERATION TRACE")
print("=" * 60)

checkpoint = SFTCheckpoint().load_state(
    CHECKPOINT,
    map_location="cpu",
)

# ---------------------------------------------------------
# RESTORE CHECKPOINT VOCABULARY
# ---------------------------------------------------------
# load_state() validates and returns the checkpoint only.
# The generation trace must explicitly restore the tokenizer
# vocabulary before encoding/decoding any chat tokens.

SFTCheckpoint._restore_vocabulary(
    checkpoint["vocabulary"]
)

config = checkpoint["model_config"]

if TOKENIZER.vocab.size() != int(config["vocab_size"]):
    raise RuntimeError(
        "runtime tokenizer/model vocabulary mismatch: "
        f"tokenizer={TOKENIZER.vocab.size()} "
        f"model={config['vocab_size']}"
    )

print("runtime vocabulary restored: PASS")
print("runtime vocab size:", TOKENIZER.vocab.size())

print("checkpoint step:", checkpoint["training"]["step"])
print("checkpoint loss:", checkpoint["training"]["loss"])
print("vocab:", config["vocab_size"])

model = NeuraTorchModel(
    vocab_size=config["vocab_size"],
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

print("model loaded: PASS")
print()

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

# ---------------------------------------------------------
# BUILD PROMPT
# ---------------------------------------------------------

from neura.chat import ChatTemplate
from neura.tokenizer import TOKENIZER

# Generation must use the canonical generation serializer.
# SFT dataset construction uses render(), while inference must use
# render_for_generation(), which appends the empty assistant turn.

template = ChatTemplate()

rendered_prompt = template.render_for_generation(
    messages
)

print("SERIALIZED GENERATION PROMPT:")
print(repr(rendered_prompt))
print()

print(rendered_prompt)

input_ids = TOKENIZER.encode(
    rendered_prompt
)

print("PROMPT IDS:")
print(input_ids)
print()

print("PROMPT TOKENS:")

for i, token_id in enumerate(input_ids):
    print(
        f"{i:03d} "
        f"id={token_id:03d} "
        f"token={token_name(token_id)!r}"
    )

print()
print("=" * 60)
print("AUTOREGRESSIVE TRACE")
print("=" * 60)

generated = []

# ---------------------------------------------------------
# MANUAL GREEDY GENERATION
# ---------------------------------------------------------

for step in range(12):

    ids = input_ids + generated

    x = torch.tensor(
        [ids],
        dtype=torch.long,
    )

    with torch.no_grad():
        output = model(x)

    if hasattr(output, "logits"):
        logits = output.logits
    elif isinstance(output, dict):
        logits = output["logits"]
    else:
        logits = output

    next_logits = logits[:, -1, :]

    probabilities = torch.softmax(
        next_logits,
        dim=-1,
    )

    top_values, top_indices = torch.topk(
        probabilities,
        k=min(10, probabilities.shape[-1]),
        dim=-1,
    )

    print()
    print(
        f"STEP {step + 1}"
    )

    print(
        "context last token:",
        f"id={ids[-1]}",
        f"token={token_name(ids[-1])!r}",
    )

    print("TOP PREDICTIONS:")

    for rank, (value, index) in enumerate(
        zip(
            top_values[0].tolist(),
            top_indices[0].tolist(),
        ),
        start=1,
    ):
        print(
            f"{rank:02d}. "
            f"id={index:03d} "
            f"p={value:.6f} "
            f"token={token_name(index)!r}"
        )

    next_id = int(
        torch.argmax(
            next_logits,
            dim=-1,
        ).item()
    )

    next_token = token_name(next_id)

    print(
        "SELECTED:",
        f"id={next_id}",
        f"token={next_token!r}",
    )

    generated.append(next_id)

    if next_token == "<|end|>":
        break

print()
print("=" * 60)
print("GENERATED")
print("=" * 60)

print(
    [
        token_name(token_id)
        for token_id in generated
    ]
)

print()
print("TEXT:")
print(
    " ".join(
        token_name(token_id)
        for token_id in generated
    )
)

print("=" * 60)
