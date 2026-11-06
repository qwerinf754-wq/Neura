from neura.training.sft_dataset import (
    SFTExample,
    SFTCollator,
)


print("=" * 60)
print("NEURA SFT DATASET / MULTI-BATCH CONTRACT")
print("=" * 60)


examples = [
    SFTExample(
        input_ids=[4, 5, 6, 7],
        labels=[-100, -100, 7, 8],
        loss_mask=[0, 0, 1, 1],
    ),
    SFTExample(
        input_ids=[4, 5, 9],
        labels=[-100, 5, 10],
        loss_mask=[0, 1, 1],
    ),
]


collator = SFTCollator()

batch = collator.collate(
    examples
)

assert len(batch["input_ids"]) == 2
assert len(batch["labels"]) == 2
assert len(batch["loss_mask"]) == 2

print("Batch creation: PASS")


assert all(
    len(row) == 4
    for row in batch["input_ids"]
)

assert all(
    len(row) == 4
    for row in batch["labels"]
)

assert all(
    len(row) == 4
    for row in batch["loss_mask"]
)

print("Dynamic padding: PASS")


assert batch["input_ids"][1][-1] == 0
assert batch["labels"][1][-1] == -100
assert batch["loss_mask"][1][-1] == 0

print("Padding / ignore-index contract: PASS")


for labels, mask in zip(
    batch["labels"],
    batch["loss_mask"],
):
    for label, enabled in zip(
        labels,
        mask,
    ):
        if enabled:
            assert label != -100
        else:
            assert label == -100

print("Assistant-only loss masking: PASS")


print("=" * 60)
print(
    "RESULT: SFT DATASET / MULTI-BATCH CONTRACT PASSED"
)
print("=" * 60)
