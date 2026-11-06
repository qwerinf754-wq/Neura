"""
Neura Real Training Entry
"""

import argparse
from pathlib import Path

from neura.tokenizer.tokenizer import TOKENIZER
from neura.training.dataset import TextDataset
from neura.data.pipeline import DataPipeline
from neura.training.loss import CrossEntropyLoss
from neura.training.optimizer import AdamW
from neura.training.trainer import Trainer

from neura.model.model import NeuraModel
from neura.model.config import DEFAULT_MODEL_CONFIG


def parse_args():

    parser = argparse.ArgumentParser(
        prog="neura.training.train"
    )

    parser.add_argument(
        "--dataset",
        required=True,
        type=str
    )

    parser.add_argument(
        "--epochs",
        default=1,
        type=int
    )

    return parser.parse_args()


def load_text(path):
    pipeline = DataPipeline()

    processed = pipeline.process(path)

    print(
        "Texts loaded:",
        processed["loaded"]
    )

    print(
        "Texts cleaned:",
        processed["cleaned"]
    )

    print(
        "Texts unique:",
        processed["unique"]
    )

    return processed["texts"]


def main():

    args = parse_args()

    dataset_path = Path(
        args.dataset
    )

    if not dataset_path.exists():
        raise FileNotFoundError(
            dataset_path
        )


    print("=" * 60)
    print("Neura Training")
    print("=" * 60)

    print(
        "Dataset:",
        dataset_path
    )

    print(
        "Epochs:",
        args.epochs
    )


    texts = load_text(
        dataset_path
    )


    print(
        "Texts loaded:",
        len(texts)
    )


    dataset = TextDataset(
        texts,
        TOKENIZER,
        sequence_length=4
    )


    print(
        "Samples:",
        len(dataset)
    )


    model = NeuraModel(
        config=DEFAULT_MODEL_CONFIG
    )


    loss = CrossEntropyLoss()

    optimizer = AdamW()


    trainer = Trainer(
        model,
        dataset,
        loss,
        optimizer
    )


    print(
        "Trainer initialized"
    )

    print(
        "Starting training..."
    )


    trainer.train(
        epochs=args.epochs
    )


    print(
        "Training finished"
    )


if __name__ == "__main__":
    main()
