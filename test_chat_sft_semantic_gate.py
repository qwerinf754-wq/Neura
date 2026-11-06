"""
Neura Chat SFT Semantic Quality Gate v3

Tests whether SFT actually learned question -> answer mappings.

This is intentionally stricter than loss-only evaluation.
"""

from __future__ import annotations

import argparse
import math
import random

import torch

from neura.model.torch_model import NeuraTorchModel
from neura.tokenizer import TOKENIZER
from neura.training.sft_checkpoint import SFTCheckpoint


# =========================================================
# CONFIG
# =========================================================

DEFAULT_CHECKPOINT = (
    "checkpoints/neura_chat_sft_best.pt"
)

DEFAULT_MAX_NEW_TOKENS = 24
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_K = 0
DEFAULT_TOP_P = 1.0

MIN_TARGET_RECALL = 0.30
MIN_TARGET_OVERLAP = 0.20
MAX_BIGRAM_REPETITION = 0.50
MAX_TRIGRAM_REPETITION = 0.45
MIN_RESPONSE_TOKENS = 1


# =========================================================
# ARGUMENTS
# =========================================================

parser = argparse.ArgumentParser(
    description="Neura semantic SFT quality gate"
)

parser.add_argument(
    "--checkpoint",
    default=DEFAULT_CHECKPOINT,
)

parser.add_argument(
    "--max-new-tokens",
    type=int,
    default=DEFAULT_MAX_NEW_TOKENS,
)

parser.add_argument(
    "--temperature",
    type=float,
    default=DEFAULT_TEMPERATURE,
)

parser.add_argument(
    "--top-k",
    type=int,
    default=DEFAULT_TOP_K,
)

parser.add_argument(
    "--top-p",
    type=float,
    default=DEFAULT_TOP_P,
)

args = parser.parse_args()


# =========================================================
# DEVICE
# =========================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# =========================================================
# CHECKPOINT
# =========================================================

checkpoint = SFTCheckpoint()

state = checkpoint.load_state(
    args.checkpoint
)

config = state["model_config"]

vocab_size = int(
    config["vocab_size"]
)


# =========================================================
# VOCABULARY RESTORE
# =========================================================

checkpoint._restore_vocabulary(
    state["vocabulary"]
)

if not TOKENIZER.frozen:
    raise RuntimeError(
        "tokenizer must be frozen after checkpoint restore"
    )

if TOKENIZER.vocab.size() != vocab_size:
    raise RuntimeError(
        "tokenizer/model vocabulary mismatch"
    )


# =========================================================
# MODEL
# =========================================================

model = NeuraTorchModel(
    vocab_size=vocab_size,
    hidden_size=config["hidden_size"],
    layers=config["layers"],
    heads=config["heads"],
    context_length=config["context_length"],
    dropout=config.get("dropout", 0.0),
).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
)

restored = checkpoint.restore(
    args.checkpoint,
    model,
    optimizer,
    map_location=device,
)

model.eval()


# =========================================================
# SPECIAL TOKEN
# =========================================================

END_ID = None

for token, idx in (
    TOKENIZER.vocab.token_to_id.items()
):
    if token == "<|end|>":
        END_ID = int(idx)
        break


# =========================================================
# DATA
# =========================================================

tests = [
    {
        "category": "greeting",
        "question": "Hello",
        "target": "Hello. I'm Neura.",
    },
    {
        "category": "identity",
        "question": "What are you?",
        "target": "I am Neura.",
    },
    {
        "category": "capabilities",
        "question": "What can you do?",
        "target": "I can learn and respond.",
    },
]


# =========================================================
# PROMPT
# =========================================================

def render_messages(messages):

    output = []

    for message in messages:

        output.append(
            f"<|{message['role']}|>\n"
            f"{message['content']}\n"
            "<|end|>\n"
        )

    output.append(
        "<|assistant|>\n"
    )

    return "".join(output)


# =========================================================
# TOKENIZER SAFETY
# =========================================================

def encode_safe(text):

    ids = TOKENIZER.encode(text)

    invalid = [
        idx
        for idx in ids
        if idx < 0
        or idx >= vocab_size
    ]

    if invalid:
        raise RuntimeError(
            f"invalid tokenizer IDs: {invalid}"
        )

    return ids


# =========================================================
# GENERATION
# =========================================================

@torch.no_grad()
def generate(messages):

    prompt = render_messages(
        messages
    )

    ids = encode_safe(prompt)

    input_ids = torch.tensor(
        [ids],
        dtype=torch.long,
        device=device,
    )

    generated = []

    for _ in range(
        args.max_new_tokens
    ):

        if (
            input_ids.shape[1]
            >= config["context_length"]
        ):
            break

        logits = model(
            input_ids
        )

        next_logits = (
            logits[0, -1, :]
            .clone()
        )

        if args.temperature <= 0:

            next_id = int(
                torch.argmax(
                    next_logits
                ).item()
            )

        else:

            next_logits = (
                next_logits
                / args.temperature
            )

            if args.top_k > 0:

                k = min(
                    args.top_k,
                    next_logits.shape[-1],
                )

                values, indices = (
                    torch.topk(
                        next_logits,
                        k,
                    )
                )

                filtered = torch.full_like(
                    next_logits,
                    float("-inf"),
                )

                filtered.scatter_(
                    0,
                    indices,
                    values,
                )

                next_logits = filtered

            probabilities = torch.softmax(
                next_logits,
                dim=-1,
            )

            if (
                args.top_p > 0
                and args.top_p < 1
            ):

                values, indices = torch.sort(
                    probabilities,
                    descending=True,
                )

                cumulative = torch.cumsum(
                    values,
                    dim=-1,
                )

                remove = (
                    cumulative > args.top_p
                )

                remove[1:] = (
                    remove[:-1].clone()
                )

                remove[0] = False

                values[remove] = 0.0

                filtered = torch.zeros_like(
                    probabilities
                )

                filtered.scatter_(
                    0,
                    indices,
                    values,
                )

                probabilities = filtered

                total = probabilities.sum()

                if total <= 0:
                    next_id = int(
                        torch.argmax(
                            next_logits
                        ).item()
                    )
                else:
                    probabilities = (
                        probabilities / total
                    )

                    next_id = int(
                        torch.multinomial(
                            probabilities,
                            1,
                        ).item()
                    )

            else:

                next_id = int(
                    torch.multinomial(
                        probabilities,
                        1,
                    ).item()
                )

        if not (
            0 <= next_id < vocab_size
        ):
            raise RuntimeError(
                f"invalid generated token: {next_id}"
            )

        generated.append(
            next_id
        )

        if (
            END_ID is not None
            and next_id == END_ID
        ):
            break

        input_ids = torch.cat(
            [
                input_ids,
                torch.tensor(
                    [[next_id]],
                    dtype=torch.long,
                    device=device,
                ),
            ],
            dim=1,
        )

    return (
        TOKENIZER.decode(generated),
        generated,
    )


# =========================================================
# METRICS
# =========================================================

def ngrams(ids, n):

    if len(ids) < n:
        return []

    return [
        tuple(ids[i:i + n])
        for i in range(
            len(ids) - n + 1
        )
    ]


def repetition_ratio(
    ids,
    n,
):

    grams = ngrams(
        ids,
        n,
    )

    if not grams:
        return 0.0

    return (
        1.0
        - len(set(grams))
        / len(grams)
    )


def target_recall(
    generated_ids,
    target_ids,
):

    if not target_ids:
        return 0.0

    generated_set = set(
        generated_ids
    )

    matched = sum(
        token_id in generated_set
        for token_id in target_ids
    )

    return matched / len(
        target_ids
    )


def target_overlap(
    generated_ids,
    target_ids,
):

    if not target_ids:
        return 0.0

    generated_set = set(
        generated_ids
    )

    target_set = set(
        target_ids
    )

    if not target_set:
        return 0.0

    return len(
        generated_set
        & target_set
    ) / len(
        target_set
    )


def target_bigram_recall(
    generated_ids,
    target_ids,
):

    target_grams = set(
        ngrams(
            target_ids,
            2,
        )
    )

    if not target_grams:
        return 0.0

    generated_grams = set(
        ngrams(
            generated_ids,
            2,
        )
    )

    return len(
        target_grams
        & generated_grams
    ) / len(
        target_grams
    )


def exact_question_leakage(
    response,
    question,
):

    return (
        question.strip().lower()
        in response.strip().lower()
    )


def role_leakage(
    response,
):

    forbidden = (
        "<|system|>",
        "<|user|>",
        "<|assistant|>",
        "<|end|>",
    )

    return any(
        marker in response
        for marker in forbidden
    )


# =========================================================
# HEADER
# =========================================================

print("=" * 60)
print("NEURA CHAT SFT SEMANTIC QUALITY GATE v3")
print("=" * 60)

print(
    "checkpoint:",
    args.checkpoint,
)

print(
    "device:",
    device,
)

print(
    "checkpoint step:",
    restored["step"],
)

print(
    "checkpoint loss:",
    restored["loss"],
)

print(
    "vocab:",
    vocab_size,
)

print(
    "tokenizer frozen:",
    TOKENIZER.frozen,
)

print(
    "temperature:",
    args.temperature,
)

print(
    "max_new_tokens:",
    args.max_new_tokens,
)


# =========================================================
# RUN
# =========================================================

passed = 0

for index, test in enumerate(
    tests,
    start=1,
):

    messages = [
        {
            "role": "system",
            "content": "You are Neura.",
        },
        {
            "role": "user",
            "content": test["question"],
        },
    ]

    response, generated_ids = generate(
        messages
    )

    target_ids = encode_safe(
        test["target"]
    )

    recall = target_recall(
        generated_ids,
        target_ids,
    )

    overlap = target_overlap(
        generated_ids,
        target_ids,
    )

    bigram_recall = (
        target_bigram_recall(
            generated_ids,
            target_ids,
        )
    )

    bigram_repeat = (
        repetition_ratio(
            generated_ids,
            2,
        )
    )

    trigram_repeat = (
        repetition_ratio(
            generated_ids,
            3,
        )
    )

    unique_ratio = (
        len(set(generated_ids))
        / len(generated_ids)
        if generated_ids
        else 0.0
    )

    prompt_leak = (
        exact_question_leakage(
            response,
            test["question"],
        )
    )

    roles = role_leakage(
        response
    )

    eos_reached = (
        END_ID is not None
        and END_ID in generated_ids
    )

    valid_ids = all(
        0 <= token_id < vocab_size
        for token_id in generated_ids
    )

    test_pass = (
        len(generated_ids)
        >= MIN_RESPONSE_TOKENS
        and valid_ids
        and not prompt_leak
        and not roles
        and recall >= MIN_TARGET_RECALL
        and overlap >= MIN_TARGET_OVERLAP
        and bigram_repeat
            <= MAX_BIGRAM_REPETITION
        and trigram_repeat
            <= MAX_TRIGRAM_REPETITION
    )

    if test_pass:
        passed += 1

    print()
    print("-" * 60)

    print(
        f"TEST {index} "
        f"[{test['category']}]"
    )

    print(
        "question:",
        test["question"],
    )

    print(
        "target:",
        repr(test["target"]),
    )

    print(
        "response:",
        repr(response),
    )

    print(
        "generated tokens:",
        len(generated_ids),
    )

    print(
        "target tokens:",
        len(target_ids),
    )

    print(
        "target recall:",
        f"{recall:.3f}",
    )

    print(
        "target overlap:",
        f"{overlap:.3f}",
    )

    print(
        "target bigram recall:",
        f"{bigram_recall:.3f}",
    )

    print(
        "unique ratio:",
        f"{unique_ratio:.3f}",
    )

    print(
        "bigram repetition:",
        f"{bigram_repeat:.3f}",
    )

    print(
        "trigram repetition:",
        f"{trigram_repeat:.3f}",
    )

    print(
        "prompt leakage:",
        "FAIL"
        if prompt_leak
        else "PASS",
    )

    print(
        "role leakage:",
        "FAIL"
        if roles
        else "PASS",
    )

    print(
        "EOS:",
        "REACHED"
        if eos_reached
        else "NOT_REACHED",
    )

    print(
        "token IDs:",
        "PASS"
        if valid_ids
        else "FAIL",
    )

    print(
        "semantic quality:",
        "PASS"
        if test_pass
        else "FAIL",
    )


# =========================================================
# SUMMARY
# =========================================================

print()
print("=" * 60)
print("SEMANTIC QUALITY SUMMARY")
print("=" * 60)

print(
    "tests passed:",
    f"{passed}/{len(tests)}",
)

print(
    "minimum target recall:",
    MIN_TARGET_RECALL,
)

print(
    "minimum target overlap:",
    MIN_TARGET_OVERLAP,
)

print(
    "generation mode:",
    "greedy"
    if args.temperature <= 0
    else "sampling",
)

print()
print("=" * 60)

if passed == len(tests):
    print(
        "RESULT: SFT SEMANTIC QUALITY GATE PASS"
    )
else:
    print(
        "RESULT: SFT SEMANTIC QUALITY GATE WARNING"
    )

print("=" * 60)
