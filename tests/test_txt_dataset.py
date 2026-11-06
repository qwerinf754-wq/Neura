from pathlib import Path
from tempfile import TemporaryDirectory

from neura.training.txt_dataset import (
    TXTCorpusDataset,
    build_windows,
)


def main():
    with TemporaryDirectory() as directory:
        root = Path(directory)

        (root / "a.txt").write_text(
            (
                "Neura is an artificial intelligence project. "
                "Python is used for machine learning and software development."
            ),
            encoding="utf-8",
        )

        (root / "b.txt").write_text(
            (
                "Transformers process sequences of tokens. "
                "Language models learn to predict the next token."
            ),
            encoding="utf-8",
        )

        dataset = TXTCorpusDataset(
            root,
            minimum_length=20,
            validation_ratio=0.5,
            seed=42,
        )

        data = dataset.load()

        assert data.source_files == 2
        assert data.train_documents >= 1
        assert data.train_tokens

        x, y = build_windows(
            data.train_tokens,
            sequence_length=8,
            stride=8,
        )

        assert x
        assert y
        assert len(x) == len(y)
        assert len(x[0]) == 8
        assert len(y[0]) == 8

        print("TXT DATASET: PASS")
        print("SOURCE FILES:", data.source_files)
        print("TRAIN DOCUMENTS:", data.train_documents)
        print("VALIDATION DOCUMENTS:", data.validation_documents)
        print("TRAIN TOKENS:", len(data.train_tokens))
        print("VALIDATION TOKENS:", len(data.validation_tokens))
        print("WINDOWS:", len(x))


if __name__ == "__main__":
    main()
