from copy import deepcopy

from neura.model.attention import SelfAttention
from neura.model.parameters import PARAMETERS


HIDDEN_SIZE = 4
HEADS = 2
SEQ_LEN = 3
EPSILON = 1e-5
TOLERANCE = 1e-5


X = [
    [0.10, 0.20, 0.30, 0.40],
    [0.20, 0.30, 0.40, 0.50],
    [0.30, 0.40, 0.50, 0.60],
]


def create_model():
    PARAMETERS.weights.clear()
    PARAMETERS.grads.clear()
    PARAMETERS.count = 0

    return SelfAttention(
        hidden_size=HIDDEN_SIZE,
        heads=HEADS,
        layer_id=0,
    )


def save_weights(model):
    return {
        "Wq": deepcopy(model.Wq),
        "Wk": deepcopy(model.Wk),
        "Wv": deepcopy(model.Wv),
        "Wo": deepcopy(model.Wo),
    }


def set_weights(model, weights):
    model.Wq = deepcopy(weights["Wq"])
    model.Wk = deepcopy(weights["Wk"])
    model.Wv = deepcopy(weights["Wv"])
    model.Wo = deepcopy(weights["Wo"])


def forward_loss(model, inputs):
    output = model.forward(inputs)

    return sum(
        value
        for row in output
        for value in row
    )


def numerical_parameter_gradient(
    weights,
    parameter,
    row,
    column,
):
    plus_weights = deepcopy(weights)
    minus_weights = deepcopy(weights)

    original = weights[parameter][row][column]

    plus_weights[parameter][row][column] = (
        original + EPSILON
    )

    minus_weights[parameter][row][column] = (
        original - EPSILON
    )

    plus_model = SelfAttention(
        hidden_size=HIDDEN_SIZE,
        heads=HEADS,
        layer_id=0,
    )

    minus_model = SelfAttention(
        hidden_size=HIDDEN_SIZE,
        heads=HEADS,
        layer_id=0,
    )

    set_weights(
        plus_model,
        plus_weights,
    )

    set_weights(
        minus_model,
        minus_weights,
    )

    plus = forward_loss(
        plus_model,
        X,
    )

    minus = forward_loss(
        minus_model,
        X,
    )

    return (
        plus - minus
    ) / (2.0 * EPSILON)


def numerical_input_gradient(
    weights,
    row,
    column,
):
    plus_x = deepcopy(X)
    minus_x = deepcopy(X)

    original = X[row][column]

    plus_x[row][column] = (
        original + EPSILON
    )

    minus_x[row][column] = (
        original - EPSILON
    )

    plus_model = SelfAttention(
        hidden_size=HIDDEN_SIZE,
        heads=HEADS,
        layer_id=0,
    )

    minus_model = SelfAttention(
        hidden_size=HIDDEN_SIZE,
        heads=HEADS,
        layer_id=0,
    )

    set_weights(
        plus_model,
        weights,
    )

    set_weights(
        minus_model,
        weights,
    )

    plus = forward_loss(
        plus_model,
        plus_x,
    )

    minus = forward_loss(
        minus_model,
        minus_x,
    )

    return (
        plus - minus
    ) / (2.0 * EPSILON)


def main():
    print("Attention Numerical Gradient Test")
    print("=================================")
    print()

    # --------------------------------------------------
    # Analytical gradients
    # --------------------------------------------------

    model = create_model()

    weights = save_weights(model)

    gradient = [
        [1.0] * HIDDEN_SIZE
        for _ in range(SEQ_LEN)
    ]

    model.forward(X)
    model.backward(gradient)

    analytical = {
        "Wq": deepcopy(
            PARAMETERS.require_gradient(
                "layers.0.attention.Wq"
            )
        ),
        "Wk": deepcopy(
            PARAMETERS.require_gradient(
                "layers.0.attention.Wk"
            )
        ),
        "Wv": deepcopy(
            PARAMETERS.require_gradient(
                "layers.0.attention.Wv"
            )
        ),
        "Wo": deepcopy(
            PARAMETERS.require_gradient(
                "layers.0.attention.Wo"
            )
        ),
    }

    # --------------------------------------------------
    # Parameter gradient checks
    # --------------------------------------------------

    all_passed = True

    for parameter in (
        "Wq",
        "Wk",
        "Wv",
        "Wo",
    ):
        analytical_value = (
            analytical[parameter][0][0]
        )

        numerical_value = (
            numerical_parameter_gradient(
                weights,
                parameter,
                0,
                0,
            )
        )

        error = abs(
            analytical_value
            - numerical_value
        )

        passed = error < TOLERANCE

        if not passed:
            all_passed = False

        print(
            f"{parameter:>2} "
            f"analytical={analytical_value:.12f} "
            f"numerical={numerical_value:.12f} "
            f"error={error:.12e} "
            f"{'PASS' if passed else 'FAIL'}"
        )

    print()

    # --------------------------------------------------
    # Input gradient
    # --------------------------------------------------

    model = SelfAttention(
        hidden_size=HIDDEN_SIZE,
        heads=HEADS,
        layer_id=0,
    )

    set_weights(model, weights)

    model.forward(X)

    analytical_dx = model.backward(
        gradient
    )

    max_dx_error = 0.0

    for row in range(SEQ_LEN):
        for column in range(HIDDEN_SIZE):

            numerical_value = (
                numerical_input_gradient(
                    weights,
                    row,
                    column,
                )
            )

            error = abs(
                analytical_dx[row][column]
                - numerical_value
            )

            max_dx_error = max(
                max_dx_error,
                error,
            )

    dx_passed = (
        max_dx_error < TOLERANCE
    )

    if not dx_passed:
        all_passed = False

    print(
        "dX "
        f"max_error={max_dx_error:.12e} "
        f"{'PASS' if dx_passed else 'FAIL'}"
    )

    print()

    # --------------------------------------------------
    # Final result
    # --------------------------------------------------

    if all_passed:
        print(
            "ATTENTION GRADIENT CHECK: PASS"
        )
        return 0

    print(
        "ATTENTION GRADIENT CHECK: FAIL"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
