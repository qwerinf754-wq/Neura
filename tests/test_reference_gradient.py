"""
Neura Reference Backend - Numerical Gradient Check

Checks analytic backpropagation against finite differences.
"""

import math

from neura.model.parameters import PARAMETERS
from neura.model.transformer import Transformer


EPSILON = 1e-5
TOLERANCE = 5e-3


def cross_entropy_loss_and_gradient(logits, targets):
    """
    logits:
        [sequence][vocab]

    targets:
        one target token per sequence position

    Returns:
        average_loss, gradient
    """

    if len(logits) != len(targets):
        raise ValueError("Length mismatch")

    gradient = []
    total_loss = 0.0

    for row, target in zip(logits, targets):
        maximum = max(row)

        exp_values = [
            math.exp(value - maximum)
            for value in row
        ]

        total = sum(exp_values)

        probabilities = [
            value / total
            for value in exp_values
        ]

        if target < 0 or target >= len(row):
            raise ValueError("Target outside vocabulary")

        total_loss -= math.log(
            max(probabilities[target], 1e-30)
        )

        grad = list(probabilities)
        grad[target] -= 1.0

        gradient.append(grad)

    count = len(targets)

    total_loss /= count

    gradient = [
        [
            value / count
            for value in row
        ]
        for row in gradient
    ]

    return total_loss, gradient


def evaluate(model, tokens, targets):
    """
    Forward-only evaluation.

    Training caches are explicitly cleared because this function
    does not consume them through backward().
    """

    model.clear_cache()

    logits = model.forward(tokens)

    loss, _ = cross_entropy_loss_and_gradient(
        logits,
        targets,
    )

    model.clear_cache()

    return loss


def analytic_gradient(
    model,
    tokens,
    targets,
    parameter_name,
    indices,
):
    PARAMETERS.zero_grad()
    model.clear_cache()

    logits = model.forward(tokens)

    loss, gradient = cross_entropy_loss_and_gradient(
        logits,
        targets,
    )

    model.backward(gradient)

    value = PARAMETERS.require(parameter_name)

    for index in indices:
        value = value[index]

    analytic = float(value)

    grad = PARAMETERS.require_gradient(parameter_name)

    for index in indices:
        grad = grad[index]

    analytic_gradient_value = float(grad)

    model.clear_cache()

    return (
        loss,
        analytic_gradient_value,
    )


def numerical_gradient(
    model,
    tokens,
    targets,
    parameter_name,
    indices,
    epsilon=EPSILON,
):
    parameter = PARAMETERS.require(parameter_name)

    value = parameter

    for index in indices[:-1]:
        value = value[index]

    final_index = indices[-1]

    original = value[final_index]

    value[final_index] = original + epsilon

    loss_plus = evaluate(
        model,
        tokens,
        targets,
    )

    value[final_index] = original - epsilon

    loss_minus = evaluate(
        model,
        tokens,
        targets,
    )

    value[final_index] = original

    model.clear_cache()

    return (
        loss_plus - loss_minus
    ) / (2.0 * epsilon)


def relative_error(analytic, numerical):
    denominator = max(
        abs(analytic),
        abs(numerical),
        1e-8,
    )

    return abs(
        analytic - numerical
    ) / denominator


def check_parameter(
    model,
    tokens,
    targets,
    parameter_name,
    indices,
):
    loss, analytic = analytic_gradient(
        model,
        tokens,
        targets,
        parameter_name,
        indices,
    )

    numerical = numerical_gradient(
        model,
        tokens,
        targets,
        parameter_name,
        indices,
    )

    error = relative_error(
        analytic,
        numerical,
    )

    passed = error < TOLERANCE

    print(
        f"{parameter_name}{indices}: "
        f"analytic={analytic:.10e} "
        f"numerical={numerical:.10e} "
        f"error={error:.10e} "
        f"{'PASS' if passed else 'FAIL'}"
    )

    return passed


def main():
    print("=" * 72)
    print("Neura Reference Backend Gradient Check")
    print("=" * 72)

    PARAMETERS.clear()

    model = Transformer(
        vocab_size=7,
        hidden_size=4,
        heads=2,
        num_layers=1,
        intermediate_size=8,
        context_length=8,
        tie_embeddings=False,
    )

    tokens = [1, 2, 3, 4]

    # Standard causal-LM targets.
    #
    # We only predict the next token for positions 0..N-2.
    #
    # Therefore the final position's logits receive zero gradient.
    targets = [2, 3, 4, 0]

    # We deliberately set the final target gradient to zero
    # by checking only the first three positions.
    #
    # The loss function below normally averages all supplied
    # targets, so use three logits explicitly.
    def causal_loss_gradient(logits):
        selected_logits = logits[:-1]
        selected_targets = tokens[1:]

        return cross_entropy_loss_and_gradient(
            selected_logits,
            selected_targets,
        )

    def evaluate_causal():
        model.clear_cache()

        logits = model.forward(tokens)

        loss, _ = causal_loss_gradient(logits)

        model.clear_cache()

        return loss

    def analytic_causal(parameter_name, indices):
        PARAMETERS.zero_grad()
        model.clear_cache()

        logits = model.forward(tokens)

        loss, gradient = causal_loss_gradient(logits)

        # Full sequence gradient with zero gradient on final token.
        full_gradient = [
            list(row)
            for row in gradient
        ]

        full_gradient.append(
            [0.0] * model.vocab_size
        )

        model.backward(full_gradient)

        grad = PARAMETERS.require_gradient(
            parameter_name
        )

        for index in indices:
            grad = grad[index]

        model.clear_cache()

        return loss, float(grad)

    def numerical_causal(parameter_name, indices):
        parameter = PARAMETERS.require(
            parameter_name
        )

        container = parameter

        for index in indices[:-1]:
            container = container[index]

        final_index = indices[-1]

        original = container[final_index]

        container[final_index] = (
            original + EPSILON
        )

        loss_plus = evaluate_causal()

        container[final_index] = (
            original - EPSILON
        )

        loss_minus = evaluate_causal()

        container[final_index] = original

        model.clear_cache()

        return (
            loss_plus - loss_minus
        ) / (2.0 * EPSILON)

    checks = [
        ("transformer.embedding.weight", [1, 0]),
        ("layers.0.attention.Wq", [0, 0]),
        ("layers.0.attention.Wk", [0, 0]),
        ("layers.0.attention.Wv", [0, 0]),
        ("layers.0.attention.Wo", [0, 0]),
        ("layers.0.norm1.weight", [0]),
        ("layers.0.norm2.weight", [0]),
        ("layers.0.feed_forward.W_gate", [0, 0]),
        ("layers.0.feed_forward.W_up", [0, 0]),
        ("layers.0.feed_forward.W_down", [0, 0]),
        ("transformer.final_norm.weight", [0]),
        ("transformer.lm_head.weight", [0, 0]),
    ]

    failures = []

    print()
    print("Running checks...")
    print()

    for parameter_name, indices in checks:
        loss, analytic = analytic_causal(
            parameter_name,
            indices,
        )

        numerical = numerical_causal(
            parameter_name,
            indices,
        )

        error = relative_error(
            analytic,
            numerical,
        )

        passed = error < TOLERANCE

        print(
            f"{parameter_name}{indices}: "
            f"analytic={analytic:.10e} "
            f"numerical={numerical:.10e} "
            f"error={error:.10e} "
            f"{'PASS' if passed else 'FAIL'}"
        )

        if not passed:
            failures.append(
                (
                    parameter_name,
                    indices,
                    analytic,
                    numerical,
                    error,
                )
            )

    print()
    print("=" * 72)

    if failures:
        print(
            f"FAILED: {len(failures)} gradient checks"
        )

        for failure in failures:
            print(
                " ",
                failure,
            )

        raise SystemExit(1)

    print(
        f"PASS: all {len(checks)} gradient checks"
    )


if __name__ == "__main__":
    main()
