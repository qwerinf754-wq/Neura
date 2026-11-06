"""
Neura Gradient Clipping
=======================

Global L2 gradient clipping.
"""

from __future__ import annotations

import math


def global_gradient_norm(grads):
    """Calculate the global L2 norm of all gradients."""

    total = 0.0

    for gradient in grads.values():
        for row in gradient:
            if isinstance(row, list):
                for value in row:
                    total += value * value
            else:
                total += row * row

    return math.sqrt(total)


def clip_gradients(grads, max_norm: float):
    """
    Clip gradients using global L2 norm.

    Returns:
        (original_norm, clipped)
    """

    if max_norm <= 0:
        raise ValueError(
            "max_norm must be positive."
        )

    norm = global_gradient_norm(grads)

    if norm == 0.0:
        return norm, False

    if norm <= max_norm:
        return norm, False

    scale = max_norm / norm

    for gradient in grads.values():
        for i, row in enumerate(gradient):
            if isinstance(row, list):
                for j in range(len(row)):
                    row[j] *= scale
            else:
                gradient[i] *= scale

    return norm, True


if __name__ == "__main__":
    gradients = {
        "layer.weight": [
            [3.0, 4.0],
        ],
    }

    norm, clipped = clip_gradients(
        gradients,
        max_norm=1.0,
    )

    print("Neura Gradient Clipping")
    print("=======================")
    print("Original norm :", norm)
    print("Clipped       :", clipped)
    print("Gradient      :", gradients)

    new_norm = global_gradient_norm(
        gradients
    )

    print("New norm      :", new_norm)

    assert abs(norm - 5.0) < 1e-12
    assert clipped is True
    assert abs(new_norm - 1.0) < 1e-12

    print()
    print("GRADIENT CLIPPING TEST: PASS")
