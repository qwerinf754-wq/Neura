from neura.tokenizer import TOKENIZER

IGNORE_INDEX = -100

examples = [
    {
        "system": "You are Neura.",
        "user": "Hello",
        "assistant": "Hello. I'm Neura.",
    },
    {
        "system": "You are Neura.",
        "user": "What are you?",
        "assistant": "I am Neura.",
    },
    {
        "system": "You are Neura.",
        "user": "What can you do?",
        "assistant": "I can learn and respond.",
    },
]

def render(example):
    return (
        f"<|system|>\n"
        f"{example['system']}\n"
        f"<|end|>\n"
        f"<|user|>\n"
        f"{example['user']}\n"
        f"<|end|>\n"
        f"<|assistant|>\n"
        f"{example['assistant']}\n"
        f"<|end|>\n"
    )

def find_subsequence(sequence, subsequence):
    n = len(subsequence)

    for i in range(len(sequence) - n + 1):
        if sequence[i:i+n] == subsequence:
            return i

    return None

print("=" * 60)
print("NEURA SFT LABEL / INPUT ALIGNMENT AUDIT")
print("=" * 60)

print("vocab size:", TOKENIZER.vocab.size())
print("tokenizer frozen:", TOKENIZER.frozen)

for index, example in enumerate(examples):

    prompt = render(example)
    ids = TOKENIZER.encode(prompt)

    assistant_marker = TOKENIZER.encode(
        "<|assistant|>\n"
    )

    end_marker = TOKENIZER.encode(
        "<|end|>"
    )

    assistant_text_ids = TOKENIZER.encode(
        example["assistant"]
    )

    assistant_pos = find_subsequence(
        ids,
        assistant_marker,
    )

    assistant_text_pos = find_subsequence(
        ids,
        assistant_text_ids,
    )

    end_after_assistant = None

    if assistant_text_pos is not None:
        search_start = (
            assistant_text_pos
            + len(assistant_text_ids)
        )

        for pos in range(
            search_start,
            len(ids) - len(end_marker) + 1,
        ):
            if (
                ids[pos:pos + len(end_marker)]
                == end_marker
            ):
                end_after_assistant = pos
                break

    print()
    print("-" * 60)
    print("EXAMPLE", index)
    print("system:", example["system"])
    print("user:", example["user"])
    print("assistant:", example["assistant"])

    print()
    print("full ids:")
    print(ids)

    print()
    print("assistant marker ids:")
    print(assistant_marker)

    print("assistant marker position:")
    print(assistant_pos)

    print()
    print("assistant content ids:")
    print(assistant_text_ids)

    print("assistant content position:")
    print(assistant_text_pos)

    print()
    print("assistant end marker position:")
    print(end_after_assistant)

    if assistant_text_pos is not None:
        start = assistant_text_pos
        stop = (
            end_after_assistant
            + len(end_marker)
            if end_after_assistant is not None
            else len(ids)
        )

        print()
        print("ASSISTANT REGION:")
        print(ids[start:stop])

        print()
        print("ASSISTANT TOKENS:")

        for pos in range(start, stop):
            token_id = ids[pos]

            token = None

            for text, idx in (
                TOKENIZER.vocab.token_to_id.items()
            ):
                if idx == token_id:
                    token = text
                    break

            print(
                f"position={pos:03d} "
                f"id={token_id:03d} "
                f"token={token!r}"
            )

    # -----------------------------------------------------
    # AUTOREGRESSIVE ALIGNMENT
    # -----------------------------------------------------

    print()
    print("AUTOREGRESSIVE CHECK:")

    if len(ids) >= 2:

        for pos in range(
            max(0, len(ids) - 12),
            len(ids) - 1,
        ):
            input_token = ids[pos]
            target_token = ids[pos + 1]

            input_text = None
            target_text = None

            for text, idx in (
                TOKENIZER.vocab.token_to_id.items()
            ):
                if idx == input_token:
                    input_text = text

                if idx == target_token:
                    target_text = text

            print(
                f"input[{pos}]={input_token}:{input_text!r}"
                f" -> "
                f"target[{pos+1}]={target_token}:{target_text!r}"
            )

print()
print("=" * 60)
print("RESULT: ALIGNMENT AUDIT COMPLETE")
print("=" * 60)
