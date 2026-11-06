#!/usr/bin/env python3

"""
NAA Systolic Array Reference Test

This file is a pure Python reference model for the
Neura AI Accelerator (NAA) systolic matrix engine.

It verifies matrix multiplication against the logical
systolic dataflow for several matrix sizes.
"""

from __future__ import annotations

import random


def matmul_reference(
    a: list[list[int]],
    b: list[list[int]],
) -> list[list[int]]:
    """Ordinary matrix multiplication reference."""

    rows = len(a)
    inner = len(a[0])
    cols = len(b[0])

    assert all(len(row) == inner for row in a)
    assert all(len(row) == cols for row in b)
    assert len(b) == inner

    c = [[0 for _ in range(cols)] for _ in range(rows)]

    for i in range(rows):
        for j in range(cols):
            total = 0

            for k in range(inner):
                total += a[i][k] * b[k][j]

            c[i][j] = total

    return c


def systolic_reference(
    a: list[list[int]],
    b: list[list[int]],
) -> list[list[int]]:
    """
    Logical reference model of an N x N systolic array.

    Each PE(i,j) computes:

        C[i][j] = sum(A[i][k] * B[k][j])

    The implementation models the mathematical dataflow,
    not RTL clock timing.
    """

    n = len(a)

    assert n > 0
    assert len(b) == n

    assert all(len(row) == n for row in a)
    assert all(len(row) == n for row in b)

    accum = [[0 for _ in range(n)] for _ in range(n)]

    for k in range(n):
        for i in range(n):
            for j in range(n):
                accum[i][j] += a[i][k] * b[k][j]

    return accum


def random_matrix(
    rows: int,
    cols: int,
    low: int = -4,
    high: int = 4,
) -> list[list[int]]:
    """Generate a deterministic random integer matrix."""

    return [
        [random.randint(low, high) for _ in range(cols)]
        for _ in range(rows)
    ]


def run_test(n: int) -> None:
    """Run one N x N matrix multiplication test."""

    a = random_matrix(n, n)
    b = random_matrix(n, n)

    expected = matmul_reference(a, b)
    actual = systolic_reference(a, b)

    if actual != expected:
        print(f"[FAIL] {n}x{n}")

        print("A:")
        for row in a:
            print(row)

        print("B:")
        for row in b:
            print(row)

        print("Expected:")
        for row in expected:
            print(row)

        print("Actual:")
        for row in actual:
            print(row)

        raise AssertionError(
            f"Systolic reference mismatch for {n}x{n}"
        )

    print(f"[PASS] {n}x{n} systolic reference")


def main() -> None:
    """Run the complete reference test suite."""

    random.seed(42)

    print("=" * 60)
    print("NAA SYSTOLIC ARRAY REFERENCE TEST")
    print("=" * 60)

    for n in (2, 4, 8, 16):
        run_test(n)

    print("-" * 60)
    print("All systolic reference tests passed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
