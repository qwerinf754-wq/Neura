"""
Neura Chat SFT Quality Gate v2

Strict structural/semantic evaluation for the current SFT checkpoint.
"""

from __future__ import annotations

import argparse
import re
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
DEFAULT_TEMPERATURE = 0.8
DEFAULT_TOP_K = 5
DEFAULT_TOP_P = 0.9

MIN_RESPONSE_TOKENS = 1
MAX_BIGRAM_REPETITION = 0.50
MAX_TRIGRAM_REPETITION = 0.45
MIN_UNIQUE_RATIO = 0.15


# =========================================================
# ARGUMENTS
# =========================================================

parser = argparse.ArgumentParser(
    description="Neura strict SFT quality gate v2"
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
# RESTORE VOCABULARY
# =========================================================

checkpoint._restore_vocabulary(
    state["vocabulary"]
)

assert TOKENIZER.frozen

assert (
    TOKENIZER.vocab.size()
    == vocab_size
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
# SPECIAL TOKENS
# =========================================================

END_ID = None

for token, idx in (
    TOKENIZER.vocab.token_to_id.items()
):
    if token == "<|end|>":
        END_ID = int(idx)
        break


# =========================================================
# PROMPT RENDERING
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
# SAMPLING
# =========================================================

def sample_next_token(
    logits,
    *,
    temperature,
    top_k,
    top_p,
):

    if temperature <= 0:

        return int(
            torch.argmax(logits).item()
        )

    logits = logits / temperature

    # -----------------------------------------------------
    # TOP-K
    # -----------------------------------------------------

    if top_k > 0:

        k = min(
            top_k,
            logits.shape[-1],
        )

        values, indices = torch.topk(
            logits,
            k,
        )

        filtered = torch.full_like(
            logits,
            float("-inf"),
        )

        filtered.scatter_(
            0,
            indices,
            values,
        )

        logits = filtered

    # -----------------------------------------------------
    # TOP-P
    # -----------------------------------------------------

    if (
        top_p > 0
        and top_p < 1
    ):

        sorted_logits, sorted_indices = (
            torch.sort(
                logits,
                descending=True,
            )
        )

        probabilities = torch.softmax(
            sorted_logits,
            dim=-1,
        )

        cumulative = torch.cumsum(
            probabilities,
            dim=-1,
        )

        remove = (
            cumulative > top_p
        )

        if remove.numel() > 1:

            remove[1:] = (
                remove[:-1].clone()
            )

        remove[0] = False

        sorted_logits[
            remove
        ] = float("-inf")

        filtered = torch.full_like(
            logits,
            float("-inf"),
        )

        filtered.scatter_(
            0,
            sorted_indices,
            sorted_logits,
        )

        logits = filtered

    probabilities = torch.softmax(
        logits,
        dim=-1,
    )

    if not torch.isfinite(
        probabilities
    ).all():

        return int(
            torch.argmax(logits).item()
        )

    return int(
        torch.multinomial(
            probabilities,
            1,
        ).item()
    )


# =========================================================
# GENERATION
# =========================================================

@torch.no_grad()
def generate(messages):

    prompt = render_messages(
        messages
    )

    ids = encode_safe(
        prompt
    )

    input_ids = torch.tensor(
        [ids],
        dtype=torch.long,
        device=device,
    )

    generated_ids = []

    eos_reached = False

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

        next_logits = logits[
            0,
            -1,
            :
        ].clone()

        # -------------------------------------------------
        # REPETITION PENALTY
        # -------------------------------------------------

        if generated_ids:

            recent = set(
                generated_ids[-4:]
            )

            for token_id in recent:

                next_logits[
                    token_id
                ] /= 1.35

        next_id = sample_next_token(
            next_logits,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )

        if (
            next_id < 0
            or next_id >= vocab_size
        ):
            raise RuntimeError(
                f"model produced invalid ID: {next_id}"
            )

        generated_ids.append(
            next_id
        )

        if (
            END_ID is not None
            and next_id == END_ID
        ):
            eos_reached = True
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

    text = TOKENIZER.decode(
        generated_ids
    )

    return (
        text,
        generated_ids,
        eos_reached,
    )


# =========================================================
# METRICS
# =========================================================

def has_role_leakage(text):

    forbidden = (
        "<|system|>",
        "<|user|>",
        "<|assistant|>",
        "<|end|>",
    )

    return any(
        marker in text
        for marker in forbidden
    )


def prompt_leakage_details(
    text,
    question,
):
    """
    Return exact prompt-leakage reasons.

    This intentionally does not treat ordinary words such as
    "Neura" or "I" as prompt leakage.
    """

    normalized_text = text.lower()
    normalized_question = question.lower()

    reasons = []

    forbidden_markers = (
        "<|system|>",
        "<|user|>",
        "<|assistant|>",
        "<|end|>",
    )

    for marker in forbidden_markers:
        if marker in normalized_text:
            reasons.append(
                f"marker:{marker}"
            )

    system_prompt = "you are neura."

    if system_prompt in normalized_text:
        reasons.append(
            "system_prompt"
        )

    if (
        normalized_question
        and normalized_question in normalized_text
    ):
        reasons.append(
            "exact_user_question"
        )

    return reasons


def has_prompt_leakage(
    text,
    question,
):
    return bool(
        prompt_leakage_details(
            text,
            question,
        )
    )


def adjacent_repetition(ids):

    if len(ids) <= 1:
        return 0.0

    repeats = 0

    for i in range(
        1,
        len(ids),
    ):

        if ids[i] == ids[i - 1]:
            repeats += 1

    return repeats / (
        len(ids) - 1
    )


def ngram_repetition(
    ids,
    n,
):

    if len(ids) < n * 2:
        return 0.0

    ngrams = [
        tuple(
            ids[i:i + n]
        )
        for i in range(
            len(ids) - n + 1
        )
    ]

    if not ngrams:
        return 0.0

    return 1.0 - (
        len(set(ngrams))
        / len(ngrams)
    )


def unique_ratio(ids):

    if not ids:
        return 0.0

    return (
        len(set(ids))
        / len(ids)
    )


def keyword_relevance(
    question,
    response,
):

    q = question.lower()
    r = response.lower()

    keyword_sets = {

        "hello": (
            "hello",
            "hi",
            "neura",
            "respond",
        ),

        "what are you?": (
            "neura",
            "ai",
            "model",
        ),

        "what can you do?": (
            "learn",
            "respond",
            "help",
            "answer",
            "code",
        ),

        "tell me about neura.": (
            "neura",
            "ai",
            "model",
        ),

        "say something short.": (
            "hello",
            "neura",
            "ai",
            "learn",
            "respond",
        ),
    }

    candidates = keyword_sets.get(
        q,
        (),
    )

    if not candidates:
        return True

    return any(
        keyword in r
        for keyword in candidates
    )


def language_check(text):

    if not text.strip():
        return False

    arabic_chars = len(
        re.findall(
            r"[\u0600-\u06FF]",
            text,
        )
    )

    latin_chars = len(
        re.findall(
            r"[A-Za-z]",
            text,
        )
    )

    meaningful_chars = (
        arabic_chars
        + latin_chars
    )

    if meaningful_chars == 0:
        return False

    # Current Neura SFT dataset is English.
    # Arabic-only accidental outputs are therefore suspicious.
    if (
        arabic_chars > 0
        and latin_chars == 0
    ):
        return False

    return True


# =========================================================
# DETERMINISM
# =========================================================

def deterministic_generation_test(
    messages,
):

    torch.manual_seed(12345)

    first, _, _ = generate(
        messages
    )

    torch.manual_seed(12345)

    second, _, _ = generate(
        messages
    )

    return first == second


# =========================================================
# HEADER
# =========================================================

print("=" * 60)
print("NEURA CHAT SFT QUALITY GATE v2")
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
    "top_k:",
    args.top_k,
)

print(
    "top_p:",
    args.top_p,
)


# =========================================================
# TEST SET
# =========================================================

tests = [
    (
        "Hello",
        "greeting",
    ),
    (
        "What are you?",
        "identity",
    ),
    (
        "What can you do?",
        "capabilities",
    ),
    (
        "Tell me about Neura.",
        "project",
    ),
    (
        "Say something short.",
        "instruction",
    ),
]


# =========================================================
# RUN TESTS
# =========================================================

responses = []

passed = 0

total = len(tests)


for index, (
    question,
    category,
) in enumerate(
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
            "content": question,
        },
    ]

    response, ids, eos = generate(
        messages
    )

    adjacent = adjacent_repetition(
        ids
    )

    bigram = ngram_repetition(
        ids,
        2,
    )

    trigram = ngram_repetition(
        ids,
        3,
    )

    diversity = unique_ratio(
        ids
    )

    leakage = has_role_leakage(
        response
    )

    leakage_reasons = (
        prompt_leakage_details(
            response,
            question,
        )
    )

    prompt_leakage = bool(
        leakage_reasons
    )

    relevance = keyword_relevance(
        question,
        response,
    )

    language = language_check(
        response
    )

    valid_ids = all(
        0 <= token_id < vocab_size
        for token_id in ids
    )

    length_ok = (
        len(ids)
        >= MIN_RESPONSE_TOKENS
    )

    repetition_ok = (
        bigram
        <= MAX_BIGRAM_REPETITION
        and trigram
        <= MAX_TRIGRAM_REPETITION
    )

    diversity_ok = (
        diversity
        >= MIN_UNIQUE_RATIO
    )

    test_pass = all(
        (
            valid_ids,
            length_ok,
            repetition_ok,
            diversity_ok,
            relevance,
            language,
            not leakage,
            not prompt_leakage,
        )
    )

    if test_pass:
        passed += 1

    responses.append(
        response
    )

    print()
    print("-" * 60)

    print(
        f"TEST {index} [{category}]"
    )

    print(
        "user:",
        question,
    )

    print(
        "response:",
        repr(response),
    )

    print(
        "tokens:",
        len(ids),
    )

    print(
        "unique ratio:",
        f"{diversity:.3f}",
    )

    print(
        "adjacent repetition:",
        f"{adjacent:.3f}",
    )

    print(
        "bigram repetition:",
        f"{bigram:.3f}",
    )

    print(
        "trigram repetition:",
        f"{trigram:.3f}",
    )

    print(
        "keyword relevance:",
        "PASS"
        if relevance
        else "FAIL",
    )

    print(
        "language:",
        "PASS"
        if language
        else "FAIL",
    )

    print(
        "role leakage:",
        "FAIL"
        if leakage
        else "PASS",
    )

    print(
        "prompt leakage:",
        "FAIL"
        if prompt_leakage
        else "PASS",
    )

    if leakage_reasons:
        print(
            "leakage reasons:",
            ", ".join(
                leakage_reasons
            ),
        )

    print(
        "EOS:",
        "REACHED"
        if eos
        else "NOT_REACHED",
    )

    print(
        "token IDs:",
        "PASS"
        if valid_ids
        else "FAIL",
    )

    print(
        "quality:",
        "PASS"
        if test_pass
        else "FAIL",
    )


# =========================================================
# RESPONSE DIVERSITY
# =========================================================

distinct_responses = len(
    set(responses)
)

response_diversity = (
    distinct_responses / total
)


# =========================================================
# DETERMINISTIC TEST
# =========================================================

deterministic_messages = [
    {
        "role": "system",
        "content": "You are Neura.",
    },
    {
        "role": "user",
        "content": "Hello",
    },
]

deterministic_pass = (
    deterministic_generation_test(
        deterministic_messages
    )
)


# =========================================================
# SUMMARY
# =========================================================

print()
print("=" * 60)
print("QUALITY GATE SUMMARY")
print("=" * 60)

print(
    "tests passed:",
    f"{passed}/{total}",
)

print(
    "distinct responses:",
    distinct_responses,
)

print(
    "response diversity:",
    f"{response_diversity:.3f}",
)

print(
    "deterministic generation:",
    "PASS"
    if deterministic_pass
    else "FAIL",
)

if response_diversity < 0.4:

    print(
        "generation behavior:",
        "HIGH REPETITION / COLLAPSE",
    )

elif response_diversity < 0.8:

    print(
        "generation behavior:",
        "LIMITED DIVERSITY",
    )

else:

    print(
        "generation behavior:",
        "DIVERSE",
    )


# =========================================================
# FINAL GATE
# =========================================================

print()
print("=" * 60)

if (
    passed == total
    and deterministic_pass
):

    print(
        "RESULT: SFT QUALITY GATE PASS"
    )

else:

    print(
        "RESULT: SFT QUALITY GATE WARNING"
    )

print("=" * 60)
