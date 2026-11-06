"""
Neura TXT -> Real PyTorch Training

Example:

python scripts/train_from_txt.py \
    --data-dir datasets/txt \
    --output-dir checkpoints/txt_training \
    --epochs 1 \
    --batch-size 2 \
    --sequence-length 32
"""

from __future__ import annotations

import argparse
import math
import json
from pathlib import Path

import torch
from torch import nn

from neura.model.torch_model import NeuraTorchModel
from neura.tokenizer.tokenizer import TOKENIZER
from neura.training.txt_dataset import TXTCorpusDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Neura from TXT files."
    )

    parser.add_argument(
        "--data-dir",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        default="checkpoints/txt_training",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--sequence-length",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--gradient-clip",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--hidden-size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--layers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--heads",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--context-length",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume training from a TXT training checkpoint.",
    )

    return parser.parse_args()


def make_batches(x, y, batch_size):
    count = x.size(0)

    for start in range(
        0,
        count,
        batch_size,
    ):
        end = min(
            start + batch_size,
            count,
        )

        yield (
            x[start:end],
            y[start:end],
        )


def evaluate(
    model,
    x,
    y,
    batch_size,
    criterion,
):
    model.eval()

    total_loss = 0.0
    batches = 0

    with torch.no_grad():
        for xb, yb in make_batches(
            x,
            y,
            batch_size,
        ):
            logits = model(xb)

            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                yb.reshape(-1),
            )

            total_loss += float(loss.item())
            batches += 1

    model.train()

    if batches == 0:
        return float("inf")

    return total_loss / batches


def main():
    args = parse_args()

    if args.epochs <= 0:
        raise ValueError("epochs must be positive")

    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")

    if args.sequence_length <= 0:
        raise ValueError(
            "sequence-length must be positive"
        )

    if args.sequence_length >= args.context_length:
        raise ValueError(
            "sequence-length must be smaller than "
            "context-length"
        )

    torch.manual_seed(args.seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 60)
    print("NEURA TXT TRAINING")
    print("=" * 60)
    print("device:", device)
    print("torch:", torch.__version__)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # DATA
    # -----------------------------------------------------

    corpus = TXTCorpusDataset(
        args.data_dir,
        minimum_length=20,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
    )

    data = corpus.load()

    print("source files:", data.source_files)
    print("train documents:", data.train_documents)
    print("validation documents:", data.validation_documents)
    print("train tokens:", len(data.train_tokens))
    print("validation tokens:", len(data.validation_tokens))

    if len(data.train_tokens) <= args.sequence_length:
        raise RuntimeError(
            "Training corpus is too small for sequence length."
        )

    stride = args.stride

    if stride is None:
        stride = args.sequence_length

    from neura.training.txt_dataset import tensors_from_tokens

    x_train, y_train = tensors_from_tokens(
        data.train_tokens,
        args.sequence_length,
        stride,
        device,
    )

    print("train windows:", x_train.size(0))

    x_validation = None
    y_validation = None

    if len(data.validation_tokens) > args.sequence_length:
        x_validation, y_validation = tensors_from_tokens(
            data.validation_tokens,
            args.sequence_length,
            stride,
            device,
        )

        print(
            "validation windows:",
            x_validation.size(0),
        )
    else:
        print(
            "validation windows: 0"
        )

    # -----------------------------------------------------
    # VOCABULARY
    # -----------------------------------------------------

    vocab_size = int(
        TOKENIZER.vocab.size()
    )

    maximum_token_id = max(
        data.train_tokens
        + data.validation_tokens
    )

    if maximum_token_id >= vocab_size:
        vocab_size = maximum_token_id + 1

    print("vocab size:", vocab_size)

    # -----------------------------------------------------
    # MODEL
    # -----------------------------------------------------

    model = NeuraTorchModel(
        vocab_size=vocab_size,
        hidden_size=args.hidden_size,
        layers=args.layers,
        heads=args.heads,
        context_length=args.context_length,
        dropout=0.0,
    ).to(device)

    parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print("parameters:", parameters)

    # -----------------------------------------------------
    # OPTIMIZER
    # -----------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    criterion = nn.CrossEntropyLoss()

    total_steps = (
        args.epochs
        * math.ceil(
            x_train.size(0)
            / args.batch_size
        )
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(total_steps, 1),
    )

    start_epoch = 1
    best_loss = float("inf")

    # -----------------------------------------------------
    # RESUME
    # -----------------------------------------------------

    if args.resume:
        resume_path = Path(args.resume)

        if not resume_path.exists():
            raise FileNotFoundError(
                f"Resume checkpoint not found: {resume_path}"
            )

        print("RESUME CHECKPOINT:", resume_path)

        resume_checkpoint = torch.load(
            resume_path,
            map_location=device,
            weights_only=False,
        )

        required_keys = {
            "model_state_dict",
            "optimizer_state_dict",
            "scheduler_state_dict",
            "epoch",
        }

        missing = required_keys.difference(
            resume_checkpoint.keys()
        )

        if missing:
            raise RuntimeError(
                "Resume checkpoint missing keys: "
                + ", ".join(sorted(missing))
            )

        model.load_state_dict(
            resume_checkpoint["model_state_dict"]
        )

        optimizer.load_state_dict(
            resume_checkpoint["optimizer_state_dict"]
        )

        scheduler.load_state_dict(
            resume_checkpoint["scheduler_state_dict"]
        )

        # Keep the scheduler horizon consistent with the
        # current requested total training duration.
        scheduler.T_max = max(total_steps, 1)

        # The checkpoint is saved after scheduler.step(), so the
        # optimizer LR in the checkpoint may represent the old
        # training horizon. Re-anchor the scheduler at the number
        # of completed optimizer steps using the NEW total_steps.
        completed_steps = int(resume_checkpoint["epoch"]) * math.ceil(
            x_train.size(0) / args.batch_size
        )
        completed_steps = min(
            completed_steps,
            max(total_steps, 1),
        )

        # Rebuild the scheduler position without using the
        # deprecated epoch argument to scheduler.step().
        #
        # The optimizer checkpoint contains the old LR. For resume,
        # compute the LR belonging to the completed step in the NEW
        # training horizon and apply it directly to the optimizer.
        if total_steps > 0:
            import math as _math

            progress = min(
                completed_steps / total_steps,
                1.0,
            )

            for group in optimizer.param_groups:
                base_lr = float(group.get("initial_lr", args.learning_rate))
                eta_min = float(scheduler.eta_min)
                lr = eta_min + (
                    base_lr - eta_min
                ) * (
                    1.0 + _math.cos(_math.pi * progress)
                ) / 2.0
                group["lr"] = lr

        scheduler.last_epoch = completed_steps
        scheduler._step_count = completed_steps + 1

        start_epoch = int(
            resume_checkpoint["epoch"]
        ) + 1

        previous_validation = resume_checkpoint.get(
            "training",
            {},
        ).get(
            "validation_loss",
            float("inf"),
        )

        best_loss = float(
            previous_validation
        )

        # If a dedicated best checkpoint exists, preserve
        # its metric when it is better than the last checkpoint.
        best_candidate = (
            Path(args.output_dir)
            / "neura_txt_best.pt"
        )

        if best_candidate.exists():
            best_checkpoint = torch.load(
                best_candidate,
                map_location="cpu",
                weights_only=False,
            )

            best_candidate_loss = float(
                best_checkpoint.get(
                    "training",
                    {},
                ).get(
                    "validation_loss",
                    best_loss,
                )
            )

            best_loss = min(
                best_loss,
                best_candidate_loss,
            )

        print(
            "RESUME EPOCH:",
            resume_checkpoint["epoch"],
        )

        print(
            "START EPOCH:",
            start_epoch,
        )

        print(
            f"RESUME BEST LOSS: {best_loss:.6f}"
        )

        if start_epoch > args.epochs:
            print(
                "=== TXT TRAINING: NOTHING TO RESUME ==="
            )
            return

    # -----------------------------------------------------
    # TRAIN
    # -----------------------------------------------------

    for epoch in range(
        start_epoch,
        args.epochs + 1,
    ):
        model.train()

        permutation = torch.randperm(
            x_train.size(0),
            device=device,
        )

        epoch_loss = 0.0
        batches = 0

        for start in range(
            0,
            x_train.size(0),
            args.batch_size,
        ):
            indices = permutation[
                start:
                start + args.batch_size
            ]

            xb = x_train[indices]
            yb = y_train[indices]

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(xb)

            loss = criterion(
                logits.reshape(
                    -1,
                    logits.size(-1),
                ),
                yb.reshape(-1),
            )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss: {loss.item()}"
                )

            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                args.gradient_clip,
            )

            current_lr = optimizer.param_groups[0]["lr"]

            optimizer.step()
            scheduler.step()

            epoch_loss += float(
                loss.item()
            )

            batches += 1

            print(
                f"epoch={epoch} "
                f"step={batches} "
                f"loss={loss.item():.6f} "
                f"lr={current_lr:.8f} "
                f"grad_norm={float(grad_norm):.6f}"
            )

        train_loss = (
            epoch_loss / batches
        )

        if (
            x_validation is not None
            and y_validation is not None
        ):
            validation_loss = evaluate(
                model,
                x_validation,
                y_validation,
                args.batch_size,
                criterion,
            )
        else:
            validation_loss = float("nan")

        print(
            f"Epoch {epoch} complete | "
            f"train_loss={train_loss:.6f} | "
            f"validation_loss={validation_loss:.6f}"
        )

        if math.isfinite(validation_loss):
            metric = validation_loss
        else:
            metric = train_loss

        is_best = metric < best_loss

        if is_best:
            best_loss = metric

        checkpoint = {
            "version": "txt-training-v1",
            "epoch": epoch,
            "model_config": {
                "vocab_size": vocab_size,
                "hidden_size": args.hidden_size,
                "layers": args.layers,
                "heads": args.heads,
                "context_length": args.context_length,
                "dropout": 0.0,
            },
            "training": {
                "sequence_length": args.sequence_length,
                "stride": stride,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "best_loss": best_loss,
            },
            "vocabulary": TOKENIZER.vocab.state_dict(),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        }

        last_path = (
            output_dir / "neura_txt_last.pt"
        )

        torch.save(
            checkpoint,
            last_path,
        )


        if is_best:

            best_path = (
                output_dir / "neura_txt_best.pt"
            )

            torch.save(
                checkpoint,
                best_path,
            )

            print(
                "BEST CHECKPOINT:",
                best_path,
            )

    metadata = {
        "data_dir": str(
            Path(args.data_dir).resolve()
        ),
        "source_files": data.source_files,
        "train_documents": data.train_documents,
        "validation_documents": data.validation_documents,
        "train_tokens": len(data.train_tokens),
        "validation_tokens": len(data.validation_tokens),
        "vocab_size": vocab_size,
        "parameters": parameters,
        "device": str(device),
    }

    (output_dir / "metadata.json").write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=== TXT TRAINING: PASS ===")


if __name__ == "__main__":
    main()
