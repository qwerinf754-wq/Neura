"""
Neura Chat SFT V2
Real PyTorch multi-step conversational SFT.

Data:
    data/processed/neura_chat_sft_v2/train.jsonl
    data/processed/neura_chat_sft_v2/validation.jsonl

Tokenizer:
    Neura Arabic BPE
    vocab_size=2048

Training:
    assistant-only loss
    PyTorch AdamW
    gradient clipping
    validation loss
    best/last checkpoints
    checkpoint resume
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

# Android/Termux CPU determinism:
# MultiheadAttention is nondeterministic with the current 8-thread CPU path.
# Keep SFT training single-threaded for reproducible numerical behavior.
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
from torch import nn

from neura.model.torch_model import NeuraTorchModel
from neura.tokenizer.neura_bpe_tokenizer import NeuraBPETokenizer
from neura.training.chat_sft_bpe_dataset import (
    ChatSFTBuilderBPE,
    IGNORE_INDEX,
)


def main():
    # =========================================================
    # ARGUMENTS
    # =========================================================

    parser = argparse.ArgumentParser(
        description="Neura Chat SFT V2"
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--resume",
        action="store_true",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/chat_sft_v2_repaired/neura_chat_sft_last.pt",
    )

    args = parser.parse_args()

    if args.steps <= 0:
        raise ValueError("--steps must be greater than zero")


    # =========================================================
    # PATHS
    # =========================================================

    TOKENIZER_PATH = Path(
        "data/processed/neura_arabic_bpe_32k_repaired/tokenizer.json"
    )

    TRAIN_PATH = Path(
        "data/processed/neura_chat_sft_v2/train.jsonl"
    )

    VALIDATION_PATH = Path(
        "data/processed/neura_chat_sft_v2/validation.jsonl"
    )

    OUTPUT_DIR = Path(
        "checkpoints/chat_sft_v2_600"
    )

    SCHEDULED_DIR = OUTPUT_DIR / "steps"

    BEST_PATH = OUTPUT_DIR / "neura_chat_sft_best.pt"
    LAST_PATH = OUTPUT_DIR / "neura_chat_sft_last.pt"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SCHEDULED_DIR.mkdir(parents=True, exist_ok=True)

    SCHEDULED_CHECKPOINT_STEPS = {10, 20, 50, 100, 200}


    # =========================================================
    # CONFIG
    # =========================================================

    hidden_size = 128
    layers = 2
    heads = 4
    context_length = 512

    learning_rate = 1e-4
    weight_decay = 0.01
    gradient_clip = 1.0

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 70)
    print("NEURA CHAT SFT V2")
    print("=" * 70)
    print("torch:", torch.__version__)
    print("device:", device)
    print("steps:", args.steps)


    # =========================================================
    # TOKENIZER
    # =========================================================

    tokenizer = NeuraBPETokenizer(
        str(TOKENIZER_PATH)
    )

    print()
    print("TOKENIZER")
    print("vocab:", tokenizer.vocab_size)
    print("BOS:", tokenizer.BOS_ID)
    print("EOS:", tokenizer.EOS_ID)
    print("PAD:", tokenizer.PAD_ID)

    if tokenizer.vocab_size != 2048:
        raise RuntimeError(
            f"expected BPE vocab 2048, got {tokenizer.vocab_size}"
        )


    # =========================================================
    # DATA LOADING
    # =========================================================

    def load_jsonl(path: Path) -> list[list[dict]]:
        records = []

        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                line = line.strip()

                if not line:
                    continue

                record = json.loads(line)

                if "messages" not in record:
                    raise RuntimeError(
                        f"{path}:{line_number} missing messages"
                    )

                records.append(record["messages"])

        if not records:
            raise RuntimeError(f"empty dataset: {path}")

        return records


    train_records = load_jsonl(TRAIN_PATH)
    validation_records = load_jsonl(VALIDATION_PATH)

    print()
    print("DATA")
    print("train examples:", len(train_records))
    print("validation examples:", len(validation_records))


    # =========================================================
    # CHAT SFT BUILDER
    # =========================================================

    builder = ChatSFTBuilderBPE(
        tokenizer=tokenizer,
        max_length=context_length,
        add_bos=True,
        add_eos=True,
    )


    def build_examples(records):
        examples = []

        for index, messages in enumerate(records):
            example = builder.build(messages)

            if len(example.input_ids) > context_length:
                raise RuntimeError(
                    f"example {index} exceeds context length"
                )

            assistant_tokens = sum(
                label != IGNORE_INDEX
                for label in example.labels
            )

            if assistant_tokens <= 0:
                raise RuntimeError(
                    f"example {index} has no assistant loss tokens"
                )

            if len(example.input_ids) != len(example.labels):
                raise RuntimeError(
                    f"example {index} length mismatch"
                )

            examples.append(example)

        return examples


    train_examples = build_examples(train_records)
    validation_examples = build_examples(validation_records)

    print("train sequences:",
          sum(len(x.input_ids) for x in train_examples))

    print("validation sequences:",
          sum(len(x.input_ids) for x in validation_examples))

    print("train assistant tokens:",
          sum(
              sum(label != IGNORE_INDEX for label in x.labels)
              for x in train_examples
          ))

    print("validation assistant tokens:",
          sum(
              sum(label != IGNORE_INDEX for label in x.labels)
              for x in validation_examples
          ))


    # =========================================================
    # MODEL
    # =========================================================

    model = NeuraTorchModel(
        vocab_size=tokenizer.vocab_size,
        hidden_size=hidden_size,
        layers=layers,
        heads=heads,
        context_length=context_length,
        dropout=0.0,
    ).to(device)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print()
    print("MODEL")
    print("parameters:", parameter_count)


    # =========================================================
    # OPTIMIZER / LOSS
    # =========================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    criterion = nn.CrossEntropyLoss(
        ignore_index=IGNORE_INDEX,
    )


    # =========================================================
    # CHECKPOINT
    # =========================================================

    start_step = 0
    best_loss = float("inf")
    loss_history = []


    def checkpoint_payload(step: int, train_loss: float, validation_loss: float):
        return {
            "version": "chat-sft-v2-pytorch",
            "step": step,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "best_loss": best_loss,
            "loss_history": list(loss_history),
            "model_config": {
                "vocab_size": tokenizer.vocab_size,
                "hidden_size": hidden_size,
                "layers": layers,
                "heads": heads,
                "context_length": context_length,
                "dropout": 0.0,
            },
            "sft_config": {
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "gradient_clip": gradient_clip,
                "steps": args.steps,
                "ignore_index": IGNORE_INDEX,
                "policy": "assistant_only",
            },
            "tokenizer": {
                "type": "NeuraBPETokenizer",
                "artifact": str(TOKENIZER_PATH),
                "vocab_size": tokenizer.vocab_size,
                "BOS_ID": tokenizer.BOS_ID,
                "EOS_ID": tokenizer.EOS_ID,
                "PAD_ID": tokenizer.PAD_ID,
                "UNK_ID": tokenizer.UNK_ID,
            },
            "optimizer_state_dict": optimizer.state_dict(),
            "model_state_dict": model.state_dict(),
        }


    # =========================================================
    # RESUME
    # =========================================================

    if args.resume:
        checkpoint_path = Path(args.checkpoint)

        print()
        print("RESUME")
        print("checkpoint:", checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"resume checkpoint not found: {checkpoint_path}"
            )

        state = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )

        if state.get("version") != "chat-sft-v2-pytorch":
            raise RuntimeError(
                "checkpoint version mismatch"
            )

        saved_config = state["model_config"]

        if saved_config != {
            "vocab_size": tokenizer.vocab_size,
            "hidden_size": hidden_size,
            "layers": layers,
            "heads": heads,
            "context_length": context_length,
            "dropout": 0.0,
        }:
            raise RuntimeError(
                "checkpoint model config mismatch"
            )

        model.load_state_dict(
            state["model_state_dict"]
        )

        optimizer.load_state_dict(
            state["optimizer_state_dict"]
        )

        start_step = int(state["step"])
        loss_history = list(
            state.get("loss_history", [])
        )

        best_loss = float(
            state.get("best_loss", float("inf"))
        )

        print("resumed step:", start_step)
        print("best loss:", best_loss)

        if start_step >= args.steps:
            print("requested steps already reached")
            raise SystemExit(0)

    else:
        print()
        print("starting fresh SFT training")


    # =========================================================
    # LOSS FUNCTION
    # =========================================================

    def compute_loss(example):
        input_tensor = torch.tensor(
            [example.input_ids],
            dtype=torch.long,
            device=device,
        )

        label_tensor = torch.tensor(
            [example.labels],
            dtype=torch.long,
            device=device,
        )

        logits = model(input_tensor)

        shifted_logits = logits[:, :-1, :]
        shifted_labels = label_tensor[:, 1:]

        active = int(
            (shifted_labels != IGNORE_INDEX).sum().item()
        )

        if active <= 0:
            raise RuntimeError(
                "batch contains no assistant loss tokens"
            )

        loss = criterion(
            shifted_logits.reshape(
                -1,
                tokenizer.vocab_size,
            ),
            shifted_labels.reshape(-1),
        )

        return loss, active


    # =========================================================
    # VALIDATION
    # =========================================================

    @torch.no_grad()
    def evaluate():
        model.eval()

        total_loss = 0.0
        total_tokens = 0

        for example in validation_examples:
            loss, active = compute_loss(example)

            total_loss += float(loss.item()) * active
            total_tokens += active

        if total_tokens <= 0:
            raise RuntimeError(
                "validation has no assistant tokens"
            )

        return total_loss / total_tokens


    # =========================================================
    # TRAIN
    # =========================================================

    model.train()

    for step in range(start_step + 1, args.steps + 1):

        # Deterministic cycling through the validated train set.
        example = train_examples[
            (step - 1) % len(train_examples)
        ]

        optimizer.zero_grad(
            set_to_none=True
        )

        loss, active_tokens = compute_loss(
            example
        )

        loss.backward()

        pre_clip_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            gradient_clip,
        )

        optimizer.step()

        train_loss = float(
            loss.detach().cpu()
        )

        validation_loss = evaluate()

        model.train()

        loss_history.append(train_loss)

        current_lr = float(
            optimizer.param_groups[0]["lr"]
        )

        try:
            perplexity = math.exp(
                min(train_loss, 20.0)
            )
        except OverflowError:
            perplexity = float("inf")

        print(
            f"step={step:03d} "
            f"train_loss={train_loss:.6f} "
            f"val_loss={validation_loss:.6f} "
            f"ppl={perplexity:.3f} "
            f"active={active_tokens} "
            f"grad_norm={float(pre_clip_norm):.4f} "
            f"lr={current_lr:.8f}"
        )

        # =====================================================
        # BEST
        # =====================================================

        if validation_loss < best_loss:
            best_loss = validation_loss

            torch.save(
                checkpoint_payload(
                    step,
                    train_loss,
                    validation_loss,
                ),
                BEST_PATH,
            )

            print(
                "BEST CHECKPOINT:",
                BEST_PATH,
            )

        # =====================================================
        # LAST
        # =====================================================

        torch.save(
            checkpoint_payload(
                step,
                train_loss,
                validation_loss,
            ),
            LAST_PATH,
        )

        # =====================================================
        # SCHEDULED CHECKPOINT
        # =====================================================

        if step in SCHEDULED_CHECKPOINT_STEPS:
            scheduled_path = (
                SCHEDULED_DIR / f"step_{step:03d}.pt"
            )

            torch.save(
                checkpoint_payload(
                    step,
                    train_loss,
                    validation_loss,
                ),
                scheduled_path,
            )

            print(
                "SCHEDULED CHECKPOINT:",
                scheduled_path,
            )


    # =========================================================
    # RESULT
    # =========================================================

    if not loss_history:
        raise RuntimeError(
            "training produced no loss history"
        )

    initial_loss = loss_history[0]
    final_loss = loss_history[-1]

    print()
    print("=" * 70)
    print("NEURA CHAT SFT V2 RESULT")
    print("=" * 70)

    print("steps:", len(loss_history))
    print("initial train loss:", initial_loss)
    print("final train loss:", final_loss)
    print("best validation loss:", best_loss)
    print("train loss delta:", final_loss - initial_loss)

    print(
        "TRAINING SIGNAL:",
        "PASS" if final_loss < initial_loss else "WARNING",
    )

    print("BEST CHECKPOINT:", BEST_PATH)
    print("LAST CHECKPOINT:", LAST_PATH)

    print()
    print("NO RESPONSE QUALITY CLAIM")
    print("SFT infrastructure/training signal only.")


if __name__ == "__main__":
    main()
