"""
neura.core.checkpoint

Checkpoint management for the Neura AI Platform.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


DEFAULT_CHECKPOINT_DIR = "checkpoints"


def ensure_checkpoint_dir(
    directory: str | Path = DEFAULT_CHECKPOINT_DIR,
) -> Path:
    """
    Create the checkpoint directory if it does not exist.
    """

    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)

    return path


def get_checkpoint_path(
    name: str,
    directory: str | Path = DEFAULT_CHECKPOINT_DIR,
) -> Path:
    """
    Return the complete path for a checkpoint file.
    """

    if not isinstance(name, str):
        raise TypeError("name must be a string.")

    if not name.strip():
        raise ValueError("name cannot be empty.")

    checkpoint_dir = ensure_checkpoint_dir(directory)

    if not name.endswith(".pt"):
        name = f"{name}.pt"

    return checkpoint_dir / name


def build_checkpoint(
    model_state: Any = None,
    optimizer_state: Any = None,
    scheduler_state: Any = None,
    epoch: int = 0,
    step: int = 0,
    loss: float | None = None,
) -> dict[str, Any]:
    """
    Build a standardized Neura checkpoint dictionary.
    """

    if epoch < 0:
        raise ValueError("epoch must be non-negative.")

    if step < 0:
        raise ValueError("step must be non-negative.")

    return {
        "model_state": model_state,
        "optimizer_state": optimizer_state,
        "scheduler_state": scheduler_state,
        "epoch": epoch,
        "step": step,
        "loss": loss,
    }
def save_checkpoint(
    checkpoint: dict[str, Any],
    name: str,
    directory: str | Path = DEFAULT_CHECKPOINT_DIR,
) -> Path:
    """
    Save a checkpoint to disk.

    PyTorch is used when available.
    Otherwise, the checkpoint is saved using pickle.
    """

    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint must be a dictionary.")

    path = get_checkpoint_path(name, directory)

    try:
        import torch

        torch.save(checkpoint, path)

    except ImportError:
        import pickle

        with path.open("wb") as file:
            pickle.dump(
                checkpoint,
                file,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    return path


def load_checkpoint(
    name: str,
    directory: str | Path = DEFAULT_CHECKPOINT_DIR,
) -> dict[str, Any]:
    """
    Load a checkpoint from disk.
    """

    path = get_checkpoint_path(name, directory)

    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}"
        )

    try:
        import torch

        checkpoint = torch.load(
            path,
            map_location="cpu",
        )

    except ImportError:
        import pickle

        with path.open("rb") as file:
            checkpoint = pickle.load(file)

    if not isinstance(checkpoint, dict):
        raise ValueError(
            "Invalid checkpoint format."
        )

    return checkpoint


def checkpoint_exists(
    name: str,
    directory: str | Path = DEFAULT_CHECKPOINT_DIR,
) -> bool:
    """
    Check whether a checkpoint exists.
    """

    return get_checkpoint_path(
        name,
        directory,
    ).exists()


if __name__ == "__main__":
    checkpoint = build_checkpoint(
        epoch=1,
        step=100,
        loss=2.5,
    )

    path = save_checkpoint(
        checkpoint,
        "test_checkpoint",
    )

    loaded = load_checkpoint(
        "test_checkpoint",
    )

    print("Checkpoint:", path)
    print("Epoch:", loaded["epoch"])
    print("Step:", loaded["step"])
    print("Loss:", loaded["loss"])
