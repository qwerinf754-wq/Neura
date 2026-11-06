import copy
import random

from neura.model.attention import SelfAttention


random.seed(1234)

HIDDEN = 8
HEADS = 2
SEQ = 3
EPS = 1e-5


def loss(output, upstream):
    return sum(
        x * g
        for row, grow in zip(output, upstream)
        for x, g in zip(row, grow)
    )


def numerical_gradient(
    attention,
    queries,
    keys,
    values,
    upstream,
    target,
):
    data = {
        "q": copy.deepcopy(queries),
        "k": copy.deepcopy(keys),
        "v": copy.deepcopy(values),
    }

    result = [
        [0.0 for _ in row]
        for row in data[target]
    ]

    for i in range(len(data[target])):
        for j in range(len(data[target][i])):
            plus = copy.deepcopy(data)
            minus = copy.deepcopy(data)

            plus[target][i][j] += EPS
            minus[target][i][j] -= EPS

            out_plus = attention.forward(
                plus["q"],
                plus["k"],
                plus["v"],
            )

            out_minus = attention.forward(
                minus["q"],
                minus["k"],
                minus["v"],
            )

            result[i][j] = (
                loss(out_plus, upstream)
                - loss(out_minus, upstream)
            ) / (2.0 * EPS)

    return result


def max_error(a, b):
    return max(
        abs(x - y)
        for row_a, row_b in zip(a, b)
        for x, y in zip(row_a, row_b)
    )


print("=" * 60)
print("Neura Attention Gradient Check")
print("=" * 60)

attention = SelfAttention(
    HIDDEN,
    HEADS,
)

queries = [
    [
        random.uniform(-0.5, 0.5)
        for _ in range(HIDDEN)
    ]
    for _ in range(SEQ)
]

keys = [
    [
        random.uniform(-0.5, 0.5)
        for _ in range(HIDDEN)
    ]
    for _ in range(SEQ)
]

values = [
    [
        random.uniform(-0.5, 0.5)
        for _ in range(HIDDEN)
    ]
    for _ in range(SEQ)
]

upstream = [
    [
        random.uniform(-0.5, 0.5)
        for _ in range(HIDDEN)
    ]
    for _ in range(SEQ)
]


attention.forward(
    queries,
    keys,
    values,
)

analytic_q, analytic_k, analytic_v = (
    attention.backward(upstream)
)

numeric_q = numerical_gradient(
    attention,
    queries,
    keys,
    values,
    upstream,
    "q",
)

numeric_k = numerical_gradient(
    attention,
    queries,
    keys,
    values,
    upstream,
    "k",
)

numeric_v = numerical_gradient(
    attention,
    queries,
    keys,
    values,
    upstream,
    "v",
)


q_error = max_error(
    analytic_q,
    numeric_q,
)

k_error = max_error(
    analytic_k,
    numeric_k,
)

v_error = max_error(
    analytic_v,
    numeric_v,
)


print(f"Max Q error: {q_error}")
print(f"Max K error: {k_error}")
print(f"Max V error: {v_error}")


threshold = 1e-5

if (
    q_error < threshold
    and k_error < threshold
    and v_error < threshold
):
    print("Gradient Check: PASSED")
else:
    print("Gradient Check: FAILED")

    if q_error >= threshold:
        print("Q gradient requires investigation.")

    if k_error >= threshold:
        print("K gradient requires investigation.")

    if v_error >= threshold:
        print("V gradient requires investigation.")
