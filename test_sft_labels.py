from neura.tokenizer import TOKENIZER
from neura.training.chat_sft_smoke_train import (
    build_batch,
    IGNORE_INDEX,
)

print("=" * 60)
print("NEURA SFT LABEL / INPUT ALIGNMENT AUDIT v2")
print("=" * 60)

batch = build_batch()

input_ids = batch["input_ids"]
labels = batch["labels"]

print("input shape:", tuple(input_ids.shape))
print("labels shape:", tuple(labels.shape))

print(
    "trainable labels:",
    int((labels != IGNORE_INDEX).sum().item()),
)

print(
    "ignored labels:",
    int((labels == IGNORE_INDEX).sum().item()),
)


def token_for_id(token_id):
    for token, idx in (
        TOKENIZER.vocab.token_to_id.items()
    ):
        if idx == token_id:
            return token
    return f"<UNKNOWN:{token_id}>"


for example_index in range(
    input_ids.shape[0]
):

    print()
    print("-" * 60)
    print(
        f"EXAMPLE {example_index}"
    )

    ids = input_ids[
        example_index
    ].tolist()

    target = labels[
        example_index
    ].tolist()

    for pos in range(
        len(ids)
    ):

        input_id = ids[pos]
        label_id = target[pos]

        input_token = token_for_id(
            input_id
        )

        if label_id == IGNORE_INDEX:
            label_text = "IGNORE"
            status = "IGNORE"
        else:
            label_text = (
                f"{label_id}:{token_for_id(label_id)!r}"
            )
            status = "TRAIN"

        print(
            f"pos={pos:03d} "
            f"input={input_id}:{input_token!r} "
            f"target={label_text} "
            f"{status}"
        )

    print()
    print("TRAINABLE REGION:")

    for pos, label_id in enumerate(
        target
    ):

        if label_id != IGNORE_INDEX:

            print(
                f"pos={pos:03d} "
                f"target={label_id}:"
                f"{token_for_id(label_id)!r}"
            )


print()
print("=" * 60)
print("RESULT: LABEL AUDIT COMPLETE")
print("=" * 60)
