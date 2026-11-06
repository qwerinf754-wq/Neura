def test_mac():
    tests = [
        (2, 3, 0, 6),
        (4, 5, 10, 30),
        (-2, 3, 0, -6),
        (-4, -5, 2, 22),
    ]

    for a, b, acc, expected in tests:
        result = acc + (a * b)

        assert result == expected, (
            f"FAIL: {acc} + ({a} * {b}) = "
            f"{result}, expected {expected}"
        )

    print("NAA MAC reference tests: PASS")


if __name__ == "__main__":
    test_mac()
