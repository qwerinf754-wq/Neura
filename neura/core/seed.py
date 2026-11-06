"""
neura.core.seed

Reproducibility utilities for the Neura AI Platform.
"""

from __future__ import annotations

import os
import random


DEFAULT_SEED = 42


def set_python_seed(seed: int = DEFAULT_SEED) -> None:
    """
    Set the Python and OS-level random seeds.
    """

    if not isinstance(seed, int):
        raise TypeError("seed must be an integer.")

    if seed < 0:
        raise ValueError("seed must be non-negative.")

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
def set_numpy_seed(seed: int = DEFAULT_SEED) -> None:
    """
    Set the NumPy random seed when NumPy is available.
    """

    if not isinstance(seed, int):
        raise TypeError("seed must be an integer.")

    if seed < 0:
        raise ValueError("seed must be non-negative.")

    try:
        import numpy as np
    except ImportError:
        return

    np.random.seed(seed)


def set_torch_seed(seed: int = DEFAULT_SEED) -> None:
    """
    Set PyTorch random seeds when PyTorch is available.
    """

    if not isinstance(seed, int):
        raise TypeError("seed must be an integer.")

    if seed < 0:
        raise ValueError("seed must be non-negative.")

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def set_seed(seed: int = DEFAULT_SEED) -> None:
    """
    Set all supported random seeds.

    This function is safe to call even when optional
    numerical libraries are not installed.
    """

    set_python_seed(seed)
    set_numpy_seed(seed)
    set_torch_seed(seed)


if __name__ == "__main__":
    set_seed()

    print("Neura Seed Manager")
    print(f"Seed: {DEFAULT_SEED}")
    print("Reproducibility setup completed.")
def set_global_seed(seed: int = DEFAULT_SEED) -> None:
    """
    Set all supported random seeds.

    This is the public high-level seed function.
    """

    set_python_seed(seed)
    set_numpy_seed(seed)
    set_torch_seed(seed)


if __name__ == "__main__":
    set_global_seed()

    print("Neura Seed Manager")
    print(f"Seed: {DEFAULT_SEED}")
    print("Reproducibility setup completed.")
