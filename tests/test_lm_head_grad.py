import random

from neura.model.lm_head import LMHead
from neura.model.parameters import PARAMETERS


def loss(head, hidden, score_gradients):
    logits = head.forward(hidden)

    return sum(
        logits[i] * score_gradients[i]
        for i in range(len(logits))
    )


def numerical_weight_gradient(
    head,
    hidden,
    score_gradients,
    row,
    column,
    epsilon=1e-6,
):
    weight = PARAMETERS.get(head.weight_name)

    original = weight[row][column]

    weight[row][column] = original + epsilon

    plus = loss(
        head,
        hidden,
        score_gradients,
    )

    weight[row][column] = original - epsilon

    minus = loss(
        head,
        hidden,
        score_gradients,
    )

    weight[row][column] = original

    return (plus - minus) / (2.0 * epsilon)


def numerical_hidden_gradient(
    head,
    hidden,
    score_gradients,
    index,
    epsilon=1e-6,
):
    original = hidden[index]

    hidden[index] = original + epsilon

    plus = loss(
        head,
        hidden,
        score_gradients,
    )

    hidden[index] = original - epsilon

    minus = loss(
        head,
        hidden,
        score_gradients,
    )

    hidden[index] = original

    return (plus - minus) / (2.0 * epsilon)


def main():
    random.seed(42)
    PARAMETERS.clear()

    head = LMHead(
        hidden_size=8,
        vocab_size=16,
        name="lm_head",
    )

    hidden = [
        0.13,
        -0.27,
        0.41,
        -0.19,
        0.33,
        -0.52,
        0.71,
        -0.44,
    ]

    score_gradients = [
        0.0
        for _ in range(head.vocab_size)
    ]

    values = {
        1: 0.7,
        4: -1.2,
        7: 0.35,
        11: 0.9,
        15: -0.6,
    }

    for token_id, value in values.items():
        score_gradients[token_id] = value

    head.forward(hidden)

    analytical_hidden = head.backward(
        score_gradients,
        hidden,
    )

    analytical_weight = PARAMETERS.gradient(
        head.weight_name
    )

    weight_indices = [
        (1, 0),
        (4, 3),
        (7, 6),
        (11, 2),
        (15, 5),
    ]

    max_error = 0.0

    print("LM Head Numerical Gradient Test")
    print("================================")
    print()

    for row, column in weight_indices:

        analytical = analytical_weight[
            row
        ][
            column
        ]

        numerical = numerical_weight_gradient(
            head,
            hidden,
            score_gradients,
            row,
            column,
        )

        error = abs(
            analytical - numerical
        )

        max_error = max(
            max_error,
            error,
        )

        status = (
            "PASS"
            if error < 1e-6
            else "FAIL"
        )

        print(
            f"{head.weight_name}[{row},{column}] "
            f"analytical={analytical:.12e} "
            f"numerical={numerical:.12e} "
            f"error={error:.12e} {status}"
        )

    print()

    for index in range(head.hidden_size):

        analytical = analytical_hidden[
            index
        ]

        numerical = numerical_hidden_gradient(
            head,
            hidden,
            score_gradients,
            index,
        )

        error = abs(
            analytical - numerical
        )

        max_error = max(
            max_error,
            error,
        )

        status = (
            "PASS"
            if error < 1e-6
            else "FAIL"
        )

        print(
            f"dX[{index}] "
            f"analytical={analytical:.12e} "
            f"numerical={numerical:.12e} "
            f"error={error:.12e} {status}"
        )

    print()
    print(
        f"Max error: {max_error:.12e}"
    )

    if max_error >= 1e-6:
        print(
            "LM HEAD GRADIENT CHECK: FAIL"
        )
        raise SystemExit(1)

    print(
        "LM HEAD GRADIENT CHECK: PASS"
    )


if __name__ == "__main__":
    main()
