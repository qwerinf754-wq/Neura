from neura.model.parameters import PARAMETERS
from neura.model.transformer import Transformer


def clone_sequence(x):
    return [list(row) for row in x]


def loss(logits):
    total = 0.0

    for row in logits:
        for value in row:
            total += value

    return total


def numerical_gradient(
    model,
    tokens,
    parameter_name,
    i,
    j=None,
    eps=1e-5,
):
    parameter = PARAMETERS.require(parameter_name)

    if j is None:
        original = parameter[i]

        parameter[i] = original + eps
        plus = loss(model.forward(tokens))

        parameter[i] = original - eps
        minus = loss(model.forward(tokens))

        parameter[i] = original

    else:
        original = parameter[i][j]

        parameter[i][j] = original + eps
        plus = loss(model.forward(tokens))

        parameter[i][j] = original - eps
        minus = loss(model.forward(tokens))

        parameter[i][j] = original

    return (plus - minus) / (2.0 * eps)


def run_check():
    PARAMETERS.weights.clear()
    PARAMETERS.grads.clear()
    PARAMETERS.count = 0

    model = Transformer(
        vocab_size=16,
        hidden_size=8,
        heads=2,
        num_layers=2,
        intermediate_size=16,
        context_length=16,
    )

    tokens = [3, 7, 2]

    logits = model.forward(tokens)

    gradient = [
        [1.0 for _ in row]
        for row in logits
    ]

    model.backward(gradient)

    checks = [
        ("transformer.embedding.weight", 3, 2),

        ("layers.0.attention.Wq", 0, 0),
        ("layers.0.attention.Wk", 1, 2),
        ("layers.0.attention.Wv", 2, 3),
        ("layers.0.attention.Wo", 3, 1),

        ("layers.0.norm1.weight", 0, None),
        ("layers.0.norm2.weight", 3, None),

        ("layers.0.feed_forward.W_gate", 0, 0),
        ("layers.0.feed_forward.W_up", 2, 3),
        ("layers.0.feed_forward.W_down", 1, 4),

        ("layers.1.attention.Wq", 0, 0),
        ("layers.1.attention.Wk", 1, 2),
        ("layers.1.attention.Wv", 2, 3),
        ("layers.1.attention.Wo", 3, 1),

        ("layers.1.norm1.weight", 0, None),
        ("layers.1.norm2.weight", 3, None),

        ("layers.1.feed_forward.W_gate", 0, 0),
        ("layers.1.feed_forward.W_up", 2, 3),
        ("layers.1.feed_forward.W_down", 1, 4),

        ("transformer.final_norm.weight", 0, None),
        ("transformer.final_norm.weight", 3, None),

        ("transformer.lm_head.weight", 2, 3),
    ]

    print("Transformer Numerical Gradient Test")
    print("====================================")
    print()

    max_error = 0.0

    for name, i, j in checks:
        analytical_gradient = PARAMETERS.gradient(name)

        if j is None:
            analytical = analytical_gradient[i]
        else:
            analytical = analytical_gradient[i][j]

        numerical = numerical_gradient(
            model,
            tokens,
            name,
            i,
            j,
        )

        error = abs(analytical - numerical)

        if error > max_error:
            max_error = error

        status = "PASS" if error < 1e-6 else "FAIL"

        location = f"[{i}]"

        if j is not None:
            location = f"[{i},{j}]"

        print(
            f"{name}{location} "
            f"analytical={analytical:.12e} "
            f"numerical={numerical:.12e} "
            f"error={error:.12e} "
            f"{status}"
        )

    print()
    print(f"Max error: {max_error:.12e}")

    if max_error < 1e-6:
        print("TRANSFORMER GRADIENT CHECK: PASS")
    else:
        print("TRANSFORMER GRADIENT CHECK: FAIL")
        raise SystemExit(1)


if __name__ == "__main__":
    run_check()
