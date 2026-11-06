import math

from neura.model.rope import RoPE


def main():
    head_dim = 4

    rope = RoPE(
        head_dim=head_dim,
        max_length=16,
    )

    # --------------------------------------------------
    # Position 0 must not rotate the vector.
    # --------------------------------------------------

    vector = [
        1.0,
        2.0,
        3.0,
        4.0,
    ]

    result = rope.apply(
        vector,
        position=0,
    )

    error = max(
        abs(a - b)
        for a, b in zip(vector, result)
    )

    print("RoPE Test")
    print("=========")
    print()
    print("Position 0")
    print("Input :", vector)
    print("Output:", result)
    print(f"Error : {error:.12e}")

    if error >= 1e-12:
        print("POSITION 0: FAIL")
        return 1

    print("POSITION 0: PASS")
    print()

    # --------------------------------------------------
    # Rotation must preserve vector norm.
    # --------------------------------------------------

    position = 3

    result = rope.apply(
        vector,
        position=position,
    )

    input_norm = math.sqrt(
        sum(x * x for x in vector)
    )

    output_norm = math.sqrt(
        sum(x * x for x in result)
    )

    norm_error = abs(
        input_norm - output_norm
    )

    print("Position", position)
    print("Input norm :", input_norm)
    print("Output norm:", output_norm)
    print(
        f"Norm error : {norm_error:.12e}"
    )

    if norm_error >= 1e-12:
        print("NORM PRESERVATION: FAIL")
        return 1

    print("NORM PRESERVATION: PASS")
    print()

    # --------------------------------------------------
    # Different positions must produce different
    # rotations.
    # --------------------------------------------------

    result_1 = rope.apply(
        vector,
        position=1,
    )

    result_2 = rope.apply(
        vector,
        position=2,
    )

    difference = max(
        abs(a - b)
        for a, b in zip(
            result_1,
            result_2,
        )
    )

    print("Position separation")
    print(
        f"Max difference: {difference:.12e}"
    )

    if difference <= 1e-12:
        print("POSITION SEPARATION: FAIL")
        return 1

    print("POSITION SEPARATION: PASS")
    print()

    print("ROPE TEST: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
