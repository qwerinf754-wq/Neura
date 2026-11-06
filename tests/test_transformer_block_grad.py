from neura.model.parameters import PARAMETERS
from neura.model.transformer_block import TransformerBlock


def clone_matrix(x):
    return [list(row) for row in x]


def loss(output):
    total = 0.0

    for row in output:
        for value in row:
            total += value

    return total


def numerical_gradient(block, sequence, parameter_name, index_i, index_j=None, eps=1e-5):
    parameter = PARAMETERS.require(parameter_name)

    if index_j is None:
        original = parameter[index_i]

        parameter[index_i] = original + eps
        plus = loss(block.forward(clone_matrix(sequence)))

        parameter[index_i] = original - eps
        minus = loss(block.forward(clone_matrix(sequence)))

        parameter[index_i] = original

    else:
        original = parameter[index_i][index_j]

        parameter[index_i][index_j] = original + eps
        plus = loss(block.forward(clone_matrix(sequence)))

        parameter[index_i][index_j] = original - eps
        minus = loss(block.forward(clone_matrix(sequence)))

        parameter[index_i][index_j] = original

    return (plus - minus) / (2.0 * eps)


def run_check():
    PARAMETERS.weights.clear()
    PARAMETERS.grads.clear()
    PARAMETERS.count = 0

    block = TransformerBlock(
        hidden_size=8,
        heads=2,
        layer_id=0,
        intermediate_size=16,
    )

    sequence = [
        [0.10, -0.20, 0.30, -0.40, 0.50, -0.60, 0.70, -0.80],
        [0.20, -0.10, 0.40, -0.30, 0.60, -0.50, 0.80, -0.70],
        [-0.30, 0.20, -0.50, 0.40, -0.70, 0.60, -0.90, 0.80],
    ]

    output = block.forward(sequence)

    gradient = [
        [1.0 for _ in row]
        for row in output
    ]

    block.backward(gradient)

    checks = [
        ("layers.0.attention.Wq", 0, 0),
        ("layers.0.attention.Wk", 1, 2),
        ("layers.0.attention.Wv", 2, 3),
        ("layers.0.attention.Wo", 3, 1),

        ("layers.0.norm1.weight", 0, None),
        ("layers.0.norm2.weight", 3, None),

        ("layers.0.feed_forward.W_gate", 0, 0),
        ("layers.0.feed_forward.W_up", 2, 3),
        ("layers.0.feed_forward.W_down", 1, 4),
    ]

    print("TransformerBlock Numerical Gradient Test")
    print("=========================================")
    print()

    max_error = 0.0

    for name, i, j in checks:
        analytical = PARAMETERS.gradient(name)

        if j is None:
            analytical_value = analytical[i]
        else:
            analytical_value = analytical[i][j]

        numerical = numerical_gradient(
            block,
            sequence,
            name,
            i,
            j,
        )

        error = abs(analytical_value - numerical)

        max_error = max(max_error, error)

        status = "PASS" if error < 1e-6 else "FAIL"

        print(
            f"{name}[{i}"
            + (f",{j}" if j is not None else "")
            + f"] analytical={analytical_value:.12e} "
              f"numerical={numerical:.12e} "
              f"error={error:.12e} {status}"
        )

    print()
    print(f"Max error: {max_error:.12e}")

    if max_error < 1e-6:
        print("TRANSFORMER BLOCK GRADIENT: PASS")
    else:
        print("TRANSFORMER BLOCK GRADIENT: FAIL")
        raise SystemExit(1)


if __name__ == "__main__":
    run_check()
