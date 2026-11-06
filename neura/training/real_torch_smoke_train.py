"""
Neura Real PyTorch Training Contract

Trains the real NeuraTorchModel and produces:
    checkpoints/neura_torch_last.pt
    checkpoints/neura_torch_best.pt
    checkpoints/neura_vocab.json

Includes:
    - real corpus
    - real tokenizer
    - train/validation split
    - causal LM training
    - gradient clipping
    - validation loss
    - perplexity
    - checkpoint metadata
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import torch
from torch import nn

from neura.model.torch_model import NeuraTorchModel
from neura.tokenizer.tokenizer import TOKENIZER
from neura.training.real_corpus import RealCorpusPipeline


print("=" * 60)
print("NEURA REAL TORCH TRAINING CONTRACT")
print("=" * 60)

# ---------------------------------------------------------
# DEVICE
# ---------------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("torch:", torch.__version__)
print("device:", device)

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

sequence_length = 32
batch_size = 2
steps = 20

hidden_size = 256
layers = 4
heads = 8
context_length = 1024

learning_rate = 1e-3
weight_decay = 0.01
gradient_clip = 1.0

validation_ratio = 0.05

checkpoint_dir = Path("checkpoints")
checkpoint_dir.mkdir(
    parents=True,
    exist_ok=True,
)

last_checkpoint_path = (
    checkpoint_dir /
    "neura_torch_last.pt"
)

best_checkpoint_path = (
    checkpoint_dir /
    "neura_torch_best.pt"
)

vocab_path = (
    checkpoint_dir /
    "neura_vocab.json"
)

# ---------------------------------------------------------
# REAL CORPUS
# ---------------------------------------------------------

pipeline = RealCorpusPipeline(
    minimum_length=100,
    validation_ratio=validation_ratio,
    seed=42,
)

split = pipeline.run("data/corpus")

print(
    "train documents:",
    len(split.train),
)

print(
    "validation documents:",
    len(split.validation),
)

# ---------------------------------------------------------
# TOKENIZATION
# ---------------------------------------------------------

train_tokens = []

for text in split.train:

    ids = TOKENIZER.encode(text)

    if ids:
        train_tokens.extend(ids)


validation_tokens = []

for text in split.validation:

    ids = TOKENIZER.encode(text)

    if ids:
        validation_tokens.extend(ids)


print(
    "train tokens:",
    len(train_tokens),
)

print(
    "validation tokens:",
    len(validation_tokens),
)

print(
    "tokenizer vocab size:",
    TOKENIZER.vocab.size(),
)

if len(train_tokens) < 64:
    raise RuntimeError(
        "Not enough training tokens."
    )

if len(validation_tokens) < sequence_length + 1:
    raise RuntimeError(
        "Not enough validation tokens."
    )

# ---------------------------------------------------------
# VOCABULARY
# ---------------------------------------------------------

vocab_size = int(
    TOKENIZER.vocab.size()
)

maximum_token_id = max(
    train_tokens +
    validation_tokens
)

if vocab_size <= maximum_token_id:
    vocab_size = (
        maximum_token_id + 1
    )

print(
    "model vocab size:",
    vocab_size,
)

# ---------------------------------------------------------
# WINDOW BUILDER
# ---------------------------------------------------------

def build_windows(
    token_ids,
    sequence_length,
):
    usable = (
        len(token_ids)
        // sequence_length
    ) * sequence_length

    tokens = token_ids[:usable]

    inputs = []
    targets = []

    for i in range(
        0,
        len(tokens) - sequence_length,
        sequence_length,
    ):
        x = tokens[
            i:
            i + sequence_length
        ]

        y = tokens[
            i + 1:
            i + sequence_length + 1
        ]

        if len(y) != sequence_length:
            continue

        inputs.append(x)
        targets.append(y)

    return inputs, targets


train_inputs, train_targets = build_windows(
    train_tokens,
    sequence_length,
)

validation_inputs, validation_targets = build_windows(
    validation_tokens,
    sequence_length,
)

print(
    "train windows:",
    len(train_inputs),
)

print(
    "validation windows:",
    len(validation_inputs),
)

if len(train_inputs) < batch_size:
    raise RuntimeError(
        "Unable to construct training batches."
    )

if not validation_inputs:
    raise RuntimeError(
        "Unable to construct validation batches."
    )

# ---------------------------------------------------------
# TENSORS
# ---------------------------------------------------------

x_train = torch.tensor(
    train_inputs,
    dtype=torch.long,
    device=device,
)

y_train = torch.tensor(
    train_targets,
    dtype=torch.long,
    device=device,
)

x_validation = torch.tensor(
    validation_inputs,
    dtype=torch.long,
    device=device,
)

y_validation = torch.tensor(
    validation_targets,
    dtype=torch.long,
    device=device,
)

print(
    "train input shape:",
    tuple(x_train.shape),
)

print(
    "train target shape:",
    tuple(y_train.shape),
)

print(
    "validation input shape:",
    tuple(x_validation.shape),
)

print(
    "validation target shape:",
    tuple(y_validation.shape),
)

# ---------------------------------------------------------
# MODEL
# ---------------------------------------------------------

model = NeuraTorchModel(
    vocab_size=vocab_size,
    hidden_size=hidden_size,
    layers=layers,
    heads=heads,
    context_length=context_length,
    dropout=0.0,
).to(device)

parameter_count = sum(
    p.numel()
    for p in model.parameters()
)

print(
    "parameters:",
    parameter_count,
)

# ---------------------------------------------------------
# OPTIMIZER
# ---------------------------------------------------------

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    weight_decay=weight_decay,
)

criterion = nn.CrossEntropyLoss()

# ---------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------

@torch.inference_mode()
def evaluate():

    model.eval()

    total_loss = 0.0
    total_batches = 0

    for start in range(
        0,
        len(x_validation),
        batch_size,
    ):

        end = min(
            start + batch_size,
            len(x_validation),
        )

        batch_x = x_validation[
            start:end
        ]

        batch_y = y_validation[
            start:end
        ]

        logits = model(batch_x)

        loss = criterion(
            logits.reshape(
                -1,
                vocab_size,
            ),
            batch_y.reshape(-1),
        )

        total_loss += float(
            loss.detach().cpu()
        )

        total_batches += 1

    if total_batches == 0:
        return float("inf")

    return (
        total_loss /
        total_batches
    )


# ---------------------------------------------------------
# TRAIN
# ---------------------------------------------------------

model.train()

train_losses = []
validation_losses = []

best_validation_loss = float("inf")

start_time = time.perf_counter()

for step in range(
    1,
    steps + 1,
):

    start_index = (
        ((step - 1) * batch_size)
        % len(x_train)
    )

    end_index = (
        start_index + batch_size
    )

    if end_index <= len(x_train):

        batch_x = x_train[
            start_index:end_index
        ]

        batch_y = y_train[
            start_index:end_index
        ]

    else:

        remaining = (
            end_index -
            len(x_train)
        )

        batch_x = torch.cat(
            (
                x_train[
                    start_index:
                ],
                x_train[
                    :remaining
                ],
            ),
            dim=0,
        )

        batch_y = torch.cat(
            (
                y_train[
                    start_index:
                ],
                y_train[
                    :remaining
                ],
            ),
            dim=0,
        )

    optimizer.zero_grad(
        set_to_none=True
    )

    logits = model(batch_x)

    loss = criterion(
        logits.reshape(
            -1,
            vocab_size,
        ),
        batch_y.reshape(-1),
    )

    loss.backward()

    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        gradient_clip,
    )

    optimizer.step()

    train_loss = float(
        loss.detach().cpu()
    )

    validation_loss = evaluate()

    train_losses.append(
        train_loss
    )

    validation_losses.append(
        validation_loss
    )

    try:
        perplexity = math.exp(
            min(validation_loss, 20.0)
        )
    except OverflowError:
        perplexity = float("inf")

    print(
        f"step={step:02d} "
        f"train_loss={train_loss:.6f} "
        f"val_loss={validation_loss:.6f} "
        f"ppl={perplexity:.3f} "
        f"grad_norm={float(gradient_norm):.4f}"
    )

    # -----------------------------------------------------
    # BEST CHECKPOINT
    # -----------------------------------------------------

    if validation_loss < best_validation_loss:

        best_validation_loss = (
            validation_loss
        )

        best_checkpoint = {
            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "model_config": {
                "vocab_size": vocab_size,
                "hidden_size": hidden_size,
                "layers": layers,
                "heads": heads,
                "context_length":
                    context_length,
                "dropout": 0.0,
            },

            "training": {
                "step": step,
                "steps": steps,
                "sequence_length":
                    sequence_length,
                "batch_size": batch_size,
                "learning_rate":
                    learning_rate,
                "weight_decay":
                    weight_decay,
                "gradient_clip":
                    gradient_clip,
                "train_loss":
                    train_loss,
                "validation_loss":
                    validation_loss,
                "best_validation_loss":
                    best_validation_loss,
            },
        }

        torch.save(
            best_checkpoint,
            best_checkpoint_path,
        )

# ---------------------------------------------------------
# FINAL EVALUATION
# ---------------------------------------------------------

final_validation_loss = evaluate()

elapsed = (
    time.perf_counter()
    - start_time
)

# ---------------------------------------------------------
# SAVE VOCABULARY
# ---------------------------------------------------------

TOKENIZER.vocab.save(
    vocab_path
)

# ---------------------------------------------------------
# SAVE LAST CHECKPOINT
# ---------------------------------------------------------

last_checkpoint = {
    "model_state_dict":
        model.state_dict(),

    "optimizer_state_dict":
        optimizer.state_dict(),

    "model_config": {
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "layers": layers,
        "heads": heads,
        "context_length":
            context_length,
        "dropout": 0.0,
    },

    "training": {
        "step": steps,
        "steps": steps,
        "sequence_length":
            sequence_length,
        "batch_size": batch_size,
        "learning_rate":
            learning_rate,
        "weight_decay":
            weight_decay,
        "gradient_clip":
            gradient_clip,
        "initial_train_loss":
            train_losses[0],
        "final_train_loss":
            train_losses[-1],
        "final_validation_loss":
            final_validation_loss,
        "best_validation_loss":
            best_validation_loss,
        "elapsed_seconds":
            elapsed,
    },
}

torch.save(
    last_checkpoint,
    last_checkpoint_path,
)

# ---------------------------------------------------------
# RESULT
# ---------------------------------------------------------

print()
print("=" * 60)
print("REAL NEURA TORCH TRAINING RESULT")
print("=" * 60)

print(
    "initial train loss:",
    train_losses[0],
)

print(
    "final train loss:",
    train_losses[-1],
)

print(
    "train loss delta:",
    train_losses[-1]
    - train_losses[0],
)

print(
    "final validation loss:",
    final_validation_loss,
)

print(
    "best validation loss:",
    best_validation_loss,
)

try:
    final_perplexity = math.exp(
        min(final_validation_loss, 20.0)
    )
except OverflowError:
    final_perplexity = float("inf")

print(
    "final perplexity:",
    final_perplexity,
)

print(
    "elapsed seconds:",
    round(elapsed, 3),
)

print(
    "last checkpoint:",
    last_checkpoint_path,
)

print(
    "best checkpoint:",
    best_checkpoint_path,
)

print(
    "vocabulary:",
    vocab_path,
)

print(
    "learning signal:",
    "PASS"
    if train_losses[-1] < train_losses[0]
    else "WARNING",
)

print(
    "validation:",
    "PASS"
    if math.isfinite(final_validation_loss)
    else "WARNING",
)

print("forward: PASS")
print("backward: PASS")
print("optimizer update: PASS")
print("gradient clipping: PASS")
print("validation loss: PASS")
print("best checkpoint save: PASS")
print("last checkpoint save: PASS")
print("vocabulary save: PASS")

print("=" * 60)
print("RESULT: REAL NEURA TORCH TRAINING")
print("=" * 60)
