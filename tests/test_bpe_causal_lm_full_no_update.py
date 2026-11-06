import copy
import math

from neura.tokenizer.neura_bpe_tokenizer import NeuraBPETokenizer
from neura.training.chat_sft_bpe_dataset import ChatSFTBuilderBPE
from neura.training.loss import CrossEntropyLoss
from neura.model.causal_lm import CausalLM
from neura.model.parameters import PARAMETERS


TOKENIZER_PATH = (
    "data/processed/neura_arabic_bpe_32k/tokenizer.json"
)


def snapshot():
    return copy.deepcopy(PARAMETERS.state_dict())


def compare_exact(before, after):
    if before.keys() != after.keys():
        return False, "Parameter registry keys changed"

    for name in before:
        if before[name] != after[name]:
            return False, f"Parameter changed: {name}"

    return True, "All parameters identical"


def main():

    print("=" * 72)
    print("NEURA FINAL BPE -> SFT -> CAUSAL LM CONTRACT")
    print("FORWARD -> LOSS -> BACKWARD -> NO WEIGHT UPDATE")
    print("=" * 72)

    # ============================================================
    # 1. TOKENIZER
    # ============================================================
    print()
    print("[1/7] BPE tokenizer")

    tokenizer = NeuraBPETokenizer(TOKENIZER_PATH)

    if tokenizer.vocab_size != 2048:
        raise RuntimeError(
            f"Unexpected test vocabulary: "
            f"{tokenizer.vocab_size}"
        )

    print("[PASS] Tokenizer")
    print("Vocab:", tokenizer.vocab_size)

    # ============================================================
    # 2. CHAT SFT
    # ============================================================
    print()
    print("[2/7] Chat SFT")

    builder = ChatSFTBuilderBPE(
        tokenizer,
        max_length=128,
        add_bos=True,
        add_eos=False,
    )

    sample = builder.build([
        {
            "role": "system",
            "content": "You are Neura.",
        },
        {
            "role": "user",
            "content": "What is Neura?",
        },
        {
            "role": "assistant",
            "content": "Neura is an AI model.",
        },
    ])

    input_ids = sample.input_ids
    labels = sample.labels
    loss_mask = sample.loss_mask

    if not (
        len(input_ids)
        == len(labels)
        == len(loss_mask)
    ):
        raise RuntimeError(
            "SFT tensor lengths are inconsistent"
        )

    assistant_tokens = sum(
        1 for x in loss_mask if x == 1
    )

    ignored_tokens = sum(
        1 for x in loss_mask if x == 0
    )

    if assistant_tokens == 0:
        raise RuntimeError(
            "No assistant tokens"
        )

    for i, (label, mask) in enumerate(
        zip(labels, loss_mask)
    ):
        expected = 1 if label != -100 else 0

        if mask != expected:
            raise RuntimeError(
                f"Mask violation at {i}: "
                f"label={label}, mask={mask}"
            )

    print("[PASS] Chat SFT")
    print("Sequence:", len(input_ids))
    print("Assistant tokens:", assistant_tokens)
    print("Ignored tokens:", ignored_tokens)

    # ============================================================
    # 3. CAUSAL ALIGNMENT
    # ============================================================
    print()
    print("[3/7] Causal alignment")

    model_input = input_ids[:-1]
    model_target = labels[1:]

    if len(model_input) != len(model_target):
        raise RuntimeError(
            "Causal input/target mismatch"
        )

    print("[PASS] Causal alignment")
    print("Input:", len(model_input))
    print("Target:", len(model_target))

    # ============================================================
    # 4. MODEL
    # ============================================================
    print()
    print("[4/7] CausalLM")

    model = CausalLM(
        vocab_size=tokenizer.vocab_size,
        hidden_size=128,
        heads=4,
        num_layers=2,
        intermediate_size=512,
        context_length=128,
    )

    parameter_count = model.parameter_count()

    if parameter_count != 1049216:
        raise RuntimeError(
            f"Unexpected parameter count: "
            f"{parameter_count}"
        )

    before = snapshot()

    if len(before) != 21:
        raise RuntimeError(
            f"Expected 21 parameters, got {len(before)}"
        )

    print("[PASS] CausalLM")
    print("Parameters:", parameter_count)
    print("Registry entries:", len(before))

    # ============================================================
    # 5. FORWARD
    # ============================================================
    print()
    print("[5/7] Forward")

    logits = model.forward(model_input)

    if len(logits) != len(model_input):
        raise RuntimeError(
            "Forward output length mismatch"
        )

    if not logits:
        raise RuntimeError(
            "Forward returned empty logits"
        )

    for row in logits:
        for value in row:
            if not math.isfinite(value):
                raise RuntimeError(
                    "Non-finite logit detected"
                )

    print("[PASS] Forward")
    print("Logit rows:", len(logits))

    # ============================================================
    # 6. LOSS
    # ============================================================
    print()
    print("[6/7] Assistant-only loss")

    loss_fn = CrossEntropyLoss()

    loss = loss_fn.forward(
        logits,
        model_target,
    )

    if not math.isfinite(loss):
        raise RuntimeError(
            f"Non-finite loss: {loss}"
        )

    gradients = loss_fn.gradients()

    if gradients is None:
        raise RuntimeError(
            "Loss gradients are None"
        )

    print("[PASS] Loss")
    print("Loss:", loss)
    print("Assistant loss tokens:", assistant_tokens)

    # ============================================================
    # 7. BACKWARD + IMMUTABILITY
    # ============================================================
    print()
    print("[7/7] Backward + parameter immutability")

    backward_result = model.backward()

    if backward_result is None:
        raise RuntimeError(
            "CausalLM.backward() returned None"
        )

    after = snapshot()

    unchanged, reason = compare_exact(
        before,
        after,
    )

    if not unchanged:
        raise RuntimeError(reason)

    print("[PASS] Backward")
    print(
        "Backward return type:",
        type(backward_result).__name__,
    )

    print("[PASS] Parameter immutability")
    print("Changed parameters: 0")

    # ============================================================
    # FINAL
    # ============================================================
    print()
    print("=" * 72)
    print("===== FINAL CONTRACT: PASS =====")
    print("=" * 72)
    print("BPE:                     PASS")
    print("CHAT SFT:                PASS")
    print("ASSISTANT-ONLY MASK:     PASS")
    print("CAUSAL ALIGNMENT:        PASS")
    print("CAUSAL LM FORWARD:       PASS")
    print("LOSS:                    PASS")
    print("LOSS GRADIENTS:          PASS")
    print("CAUSAL LM BACKWARD:      PASS")
    print("PARAMETERS UNCHANGED:    PASS")
    print()
    print("NO OPTIMIZER")
    print("NO CHECKPOINT SAVE")
    print("NO WEIGHT UPDATE")
    print("NO TRAINING")
    print("=" * 72)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 72)
        print("===== FINAL CONTRACT: FAIL =====")
        print(type(exc).__name__, str(exc))
        print("=" * 72)
        raise
