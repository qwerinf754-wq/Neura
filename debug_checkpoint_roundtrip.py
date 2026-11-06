import json
import torch

from neura.brain.sft_v2_inference import NeuraSFTV2Runtime
from neura.training.chat_sft_bpe_dataset import ChatSFTBuilderBPE

BASE_CKPT = "checkpoints/chat_sft_v2_repaired/neura_chat_sft_best.pt"
DEBUG_CKPT = "checkpoints/debug_sft_overfit/one_example_overfit.pt"
TRAIN = "data/processed/neura_chat_sft_v2/train.jsonl"

device = torch.device("cpu")

print("=" * 80)
print("CHECKPOINT LOAD ROUNDTRIP")
print("=" * 80)

# ------------------------------------------------------------
# Same input used by the one-example test.
# ------------------------------------------------------------

with open(TRAIN, "r", encoding="utf-8") as f:
    row = json.loads(next(f))

messages = row["messages"]

# ------------------------------------------------------------
# Tokenizer.
# ------------------------------------------------------------

rt_a = NeuraSFTV2Runtime(BASE_CKPT, device="cpu")
tok = rt_a.tokenizer

builder = ChatSFTBuilderBPE(
    tokenizer=tok,
    max_length=512,
    add_bos=True,
    add_eos=True,
)

ex = builder.build(messages)

input_ids = torch.tensor(
    [ex.input_ids],
    dtype=torch.long,
    device=device,
)

# ------------------------------------------------------------
# Load debug checkpoint.
# ------------------------------------------------------------

payload = torch.load(
    DEBUG_CKPT,
    map_location="cpu",
)

state = payload["model"]

print("debug checkpoint:", DEBUG_CKPT)
print("step:", payload.get("step"))
print("parameters:", len(state))

# ------------------------------------------------------------
# Model A: load debug weights.
# ------------------------------------------------------------

rt_a = NeuraSFTV2Runtime(BASE_CKPT, device="cpu")
model_a = rt_a.model.to(device)

result_a = model_a.load_state_dict(
    state,
    strict=True,
)

print("model A load:", result_a)

model_a.eval()

with torch.no_grad():
    logits_a = model_a(input_ids)

# ------------------------------------------------------------
# Model B: independently construct and load same debug weights.
# ------------------------------------------------------------

rt_b = NeuraSFTV2Runtime(BASE_CKPT, device="cpu")
model_b = rt_b.model.to(device)

result_b = model_b.load_state_dict(
    state,
    strict=True,
)

print("model B load:", result_b)

model_b.eval()

with torch.no_grad():
    logits_b = model_b(input_ids)

# ------------------------------------------------------------
# Compare.
# ------------------------------------------------------------

diff = (logits_a - logits_b).abs()

print()
print("logits shape:", tuple(logits_a.shape))
print("max_diff:", diff.max().item())
print("mean_diff:", diff.mean().item())
print("exact_equal:", torch.equal(logits_a, logits_b))

if torch.equal(logits_a, logits_b):
    print("RESULT: CHECKPOINT LOAD OK")
else:
    print("RESULT: CHECKPOINT LOAD MISMATCH")
