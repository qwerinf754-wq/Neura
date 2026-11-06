"""
Neura AdamW Reference Test
"""

import math

from neura.model.parameters import PARAMETERS
from neura.training.optimizer import AdamW


def reset():
    PARAMETERS.weights.clear()
    PARAMETERS.grads.clear()


def reference_adamw(
    parameter,
    gradient,
    learning_rate,
    weight_decay,
    beta1,
    beta2,
    epsilon,
    m,
    v,
    step,
):
    m = beta1 * m + (1.0 - beta1) * gradient
    v = beta2 * v + (1.0 - beta2) * gradient * gradient

    bias_correction_1 = 1.0 - beta1 ** step
    bias_correction_2 = 1.0 - beta2 ** step

    m_hat = m / bias_correction_1
    v_hat = v / bias_correction_2

    update = m_hat / (
        math.sqrt(v_hat) + epsilon
    )

    # Reference for the CURRENT Neura implementation.
    parameter = parameter - learning_rate * update

    parameter = (
        parameter
        - learning_rate
        * weight_decay
        * parameter
    )

    return parameter, m, v


print("=" * 72)
print("Neura AdamW Reference Test")
print("=" * 72)


# ---------------------------------------------------------
# Test 1 + 2: exact scalar reference
# ---------------------------------------------------------

reset()

initial = 1.0
gradient = 0.5

lr = 0.1
weight_decay = 0.0
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8

PARAMETERS.add(
    "weight",
    initial,
)

PARAMETERS.set_gradient(
    "weight",
    gradient,
)

optimizer = AdamW(
    learning_rate=lr,
    weight_decay=weight_decay,
    beta1=beta1,
    beta2=beta2,
    epsilon=epsilon,
)

expected, m, v = reference_adamw(
    initial,
    gradient,
    lr,
    weight_decay,
    beta1,
    beta2,
    epsilon,
    0.0,
    0.0,
    1,
)

optimizer.step(PARAMETERS)

actual = PARAMETERS.get("weight")

print()
print("Test 1: exact first-step reference")
print("expected =", expected)
print("actual   =", actual)
print("error    =", abs(actual - expected))

assert abs(actual - expected) < 1e-12


# Second step

PARAMETERS.set_gradient(
    "weight",
    gradient,
)

expected, m, v = reference_adamw(
    expected,
    gradient,
    lr,
    weight_decay,
    beta1,
    beta2,
    epsilon,
    m,
    v,
    2,
)

optimizer.step(PARAMETERS)

actual = PARAMETERS.get("weight")

print()
print("Test 2: exact second-step reference")
print("expected =", expected)
print("actual   =", actual)
print("error    =", abs(actual - expected))

assert abs(actual - expected) < 1e-12
assert optimizer.step_count == 2


# ---------------------------------------------------------
# Test 3: zero gradient
# ---------------------------------------------------------

reset()

PARAMETERS.add(
    "weight",
    1.0,
)

PARAMETERS.set_gradient(
    "weight",
    0.0,
)

optimizer = AdamW(
    learning_rate=0.1,
    weight_decay=0.0,
)

optimizer.step(PARAMETERS)

actual = PARAMETERS.get("weight")

print()
print("Test 3: zero gradient")
print("actual =", actual)

assert abs(actual - 1.0) < 1e-12


# ---------------------------------------------------------
# Test 4: zero gradient + weight decay
# ---------------------------------------------------------

reset()

PARAMETERS.add(
    "weight",
    1.0,
)

PARAMETERS.set_gradient(
    "weight",
    0.0,
)

optimizer = AdamW(
    learning_rate=0.1,
    weight_decay=0.1,
)

optimizer.step(PARAMETERS)

actual = PARAMETERS.get("weight")

expected = 0.99

print()
print("Test 4: weight decay")
print("expected =", expected)
print("actual   =", actual)
print("error    =", abs(actual - expected))

assert abs(actual - expected) < 1e-12


# ---------------------------------------------------------
# Test 5: nested matrix
# ---------------------------------------------------------

reset()

PARAMETERS.add(
    "matrix",
    [
        [1.0, 2.0],
        [3.0, 4.0],
    ],
)

PARAMETERS.set_gradient(
    "matrix",
    [
        [0.1, 0.2],
        [0.3, 0.4],
    ],
)

optimizer = AdamW(
    learning_rate=0.01,
    weight_decay=0.0,
)

before = PARAMETERS.get("matrix")

optimizer.step(PARAMETERS)

after = PARAMETERS.get("matrix")

print()
print("Test 5: nested matrix")
print("before =", before)
print("after  =", after)

assert after != before
assert len(after) == 2
assert len(after[0]) == 2
assert len(after[1]) == 2


print()
print("=" * 72)
print("ADAMW TEST: PASS")
print("=" * 72)
