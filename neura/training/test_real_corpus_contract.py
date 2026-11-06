from pathlib import Path
import tempfile

from neura.training.real_corpus import (
    RealCorpusPipeline,
    CorpusStats,
    RealCorpusError,
)


print("=" * 60)
print("NEURA REAL CORPUS CONTRACT")
print("=" * 60)


# ---------------------------------------------------------
# Pipeline creation
# ---------------------------------------------------------

pipeline = RealCorpusPipeline(
    minimum_length=10,
    validation_ratio=0.25,
    seed=42,
)

print("Pipeline creation: PASS")


# ---------------------------------------------------------
# Synthetic real-style corpus
# ---------------------------------------------------------

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)

    (root / "a.txt").write_text(
        "Neura is an artificial intelligence model "
        "designed for language understanding.",
        encoding="utf-8",
    )

    (root / "b.txt").write_text(
        "Transformers process sequences using "
        "attention mechanisms and learned representations.",
        encoding="utf-8",
    )

    (root / "duplicate.txt").write_text(
        "Neura is an artificial intelligence model "
        "designed for language understanding.",
        encoding="utf-8",
    )

    (root / "empty.txt").write_text(
        "   \n\n   ",
        encoding="utf-8",
    )

    (root / "short.txt").write_text(
        "tiny",
        encoding="utf-8",
    )

    paths = pipeline.discover(root)

    assert len(paths) == 5

    print("Corpus discovery: PASS")

    documents = pipeline.load_files(paths)

    assert len(documents) == 5

    print("Text loading: PASS")

    normalized = pipeline.normalize(
        "  Neura   AI \n\n model   "
    )

    assert normalized == "Neura AI model"

    print("Whitespace normalization: PASS")

    texts, counters = pipeline.prepare(
        documents
    )

    (
        discovered,
        accepted,
        rejected_empty,
        rejected_short,
        duplicates,
    ) = counters

    assert discovered == 5
    assert accepted == 2
    assert rejected_empty == 1
    assert rejected_short == 1
    assert duplicates == 1

    assert len(texts) == 2

    print("Corpus filtering: PASS")
    print("Duplicate removal: PASS")


    # -----------------------------------------------------
    # Split
    # -----------------------------------------------------

    split = pipeline.split(texts)

    assert len(split.train) == 1
    assert len(split.validation) == 1

    print("Train/validation split: PASS")


    # -----------------------------------------------------
    # Determinism
    # -----------------------------------------------------

    pipeline_again = RealCorpusPipeline(
        minimum_length=10,
        validation_ratio=0.25,
        seed=42,
    )

    split_again = pipeline_again.split(
        texts
    )

    assert split.train == split_again.train
    assert (
        split.validation
        == split_again.validation
    )

    print("Deterministic split: PASS")


    # -----------------------------------------------------
    # Statistics
    # -----------------------------------------------------

    split_result = pipeline.run(root)

    assert isinstance(
        pipeline.stats,
        CorpusStats,
    )

    assert pipeline.stats.discovered == 5
    assert pipeline.stats.accepted == 2
    assert pipeline.stats.duplicates == 1
    assert pipeline.stats.train_count == 1
    assert pipeline.stats.validation_count == 1

    print("Corpus statistics: PASS")


    # -----------------------------------------------------
    # Empty corpus protection
    # -----------------------------------------------------

    empty_root = root / "empty_corpus"
    empty_root.mkdir()

    (empty_root / "empty.txt").write_text(
        "",
        encoding="utf-8",
    )

    try:
        pipeline.run(empty_root)

        raise AssertionError(
            "Empty corpus was accepted."
        )

    except RealCorpusError:
        pass

    print("Empty-corpus protection: PASS")


print("=" * 60)
print("RESULT: REAL CORPUS CONTRACT PASSED")
print("=" * 60)
