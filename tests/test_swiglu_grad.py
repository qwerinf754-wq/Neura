from neura.model.swiglu import SwiGLU
from neura.model.parameters import PARAMETERS


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def loss(layer, x, upstream):
    y = layer.forward(x)
    return dot(y, upstream)


def numerical_weight_gradient(
    layer,
    x,
    upstream,
    matrix_name,
    i,
    j,
    eps=1e-6,
):
    matrix = getattr(layer, matrix_name)

    original = matrix[i][j]

    matrix[i][j] = original + eps
    plus = loss(layer, x, upstream)

    matrix[i][j] = original - eps
    minus = loss(layer, x, upstream)

    matrix[i][j] = original

    return (plus - minus) / (2.0 * eps)


def numerical_input_gradient(
    layer,
    x,
    upstream,
    index,
    eps=1e-6,
):
    original = x[index]

    x[index] = original + eps
    plus = loss(layer, x, upstream)

    x[index] = original - eps
    minus = loss(layer, x, upstream)

    x[index] = original

    return (plus - minus) / (2.0 * eps)


def run_test():
    PARAMETERS.weights.clear()
    PARAMETERS.grads.clear()
    PARAMETERS.count = 0

    layer = SwiGLU(
        hidden_size=4,
        intermediate_size=8,
        layer_id=0,
    )

    x = [
        0.1,
        -0.2,
        0.3,
        -0.4,
    ]

    upstream = [
        0.7,
        -0.5,
        0.2,
        0.9,
    ]

    # Analytical gradients
    layer.forward(x)
    grad_input = layer.backward(upstream)

    print("SwiGLU Numerical Gradient Test")
    print("==============================")
    print()

    weight_tests = [
        ("W_gate", 0, 0),
        ("W_gate", 2, 5),
        ("W_gate", 3, 7),

        ("W_up", 0, 1),
        ("W_up", 1, 3),
        ("W_up", 3, 7),

        ("W_down", 0, 1),
        ("W_down", 4, 2),
        ("W_down", 7, 3),
    ]

    max_error = 0.0

    # ---------------------------------------------------------
    # WEIGHT GRADIENTS
    # ---------------------------------------------------------

    for matrix_name, i, j in weight_tests:

        parameter_name = (
            f"layers.0.feed_forward.{matrix_name}"
        )

        analytical = PARAMETERS.grads[
            parameter_name
        ][i][j]

        numerical = numerical_weight_gradient(
            layer,
            x,
            upstream,
            matrix_name,
            i,
            j,
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
            if error < 1e-7
            else "FAIL"
        )

        print(
            f"{matrix_name}[{i},{j}] "
            f"analytical={analytical:.12e} "
            f"numerical={numerical:.12e} "
            f"error={error:.12e} "
            f"{status}"
        )

    # ---------------------------------------------------------
    # INPUT GRADIENT
    # ---------------------------------------------------------

    print()

    input_max_error = 0.0

    for index in range(len(x)):

        numerical = numerical_input_gradient(
            layer,
            x,
            upstream,
            index,
        )

        analytical = grad_input[index]

        error = abs(
            analytical - numerical
        )

        input_max_error = max(
            input_max_error,
            error,
        )

        status = (
            "PASS"
            if error < 1e-7
            else "FAIL"
        )

        print(
            f"dX[{index}] "
            f"analytical={analytical:.12e} "
            f"numerical={numerical:.12e} "
            f"error={error:.12e} "
            f"{status}"
        )

    max_error = max(
        max_error,
        input_max_error,
    )

    print()
    print(
        f"Max error: {max_error:.12e}"
    )

    if max_error < 1e-7:
        print(
            "SWIGLU GRADIENT CHECK: PASS"
        )
    else:
        print(
            "SWIGLU GRADIENT CHECK: FAIL"
        )
        raise SystemExit(1)


if __name__ == "__main__":
    run_test()
