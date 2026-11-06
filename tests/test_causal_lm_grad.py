import math

from neura.model.causal_lm import CausalLM
from neura.model.parameters import PARAMETERS


def loss_for(model, tokens):
    PARAMETERS.grads.clear()
    model._cache.clear()
    return model.loss(tokens)


def main():
    PARAMETERS.weights.clear()
    PARAMETERS.grads.clear()
    PARAMETERS.count = 0

    model = CausalLM(
        vocab_size=8,
        hidden_size=4,
        heads=2,
        num_layers=1,
        intermediate_size=8,
        context_length=8,
    )

    tokens = [1, 3, 2, 6]

    # Analytical gradient
    PARAMETERS.grads.clear()

    loss = model.loss(tokens)
    model.backward()

    analytical = PARAMETERS.gradient(
        "transformer.lm_head.weight"
    )[2][1]

    # Numerical gradient
    epsilon = 1e-5

    weight = PARAMETERS.require(
        "transformer.lm_head.weight"
    )

    original = weight[2][1]

    weight[2][1] = original + epsilon
    plus = loss_for(model, tokens)

    weight[2][1] = original - epsilon
    minus = loss_for(model, tokens)

    weight[2][1] = original

    numerical = (plus - minus) / (2.0 * epsilon)

    error = abs(analytical - numerical)

    print("CausalLM Numerical Gradient Test")
    print("================================")
    print()
    print(
        "LM Head analytical=%.12e numerical=%.12e error=%.12e %s"
        % (
            analytical,
            numerical,
            error,
            "PASS" if error < 1e-6 else "FAIL",
        )
    )

    if not math.isfinite(loss):
        raise RuntimeError("Loss is not finite.")

    if error >= 1e-6:
        raise RuntimeError(
            "CausalLM gradient check failed."
        )

    print()
    print("Loss:", loss)
    print()
    print("CAUSAL LM GRADIENT CHECK: PASS")


if __name__ == "__main__":
    main()
