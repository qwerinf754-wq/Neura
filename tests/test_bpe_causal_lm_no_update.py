import copy
import math

from neura.tokenizer.neura_bpe_tokenizer import NeuraBPETokenizer
from neura.training.chat_sft_bpe_dataset import ChatSFTBuilderBPE
from neura.training.loss import CrossEntropyLoss
from neura.model.causal_lm import CausalLM
from neura.model.parameters import PARAMETERS


TOKENIZER_PATH = "data/processed/neura_arabic_bpe_32k/tokenizer.json"


def snapshot_parameters():
    return copy.deepcopy(PARAMETERS.state_dict())


def compare_parameters(before, after):
    if before.keys() != after.keys():
        missing = sorted(set(before) - set(after))
        added = sorted(set(after) - set(before))
        return False, (
            f"registry keys changed: "
            f"missing={missing}, added={added}"
        )

    for name in before:
        a = before[name]
        b = after[name]

        if type(a) is not type(b):
            return False, f"{name}: type changed"

        if isinstance(a, list):
            if len(a) != len(b):
                return False, f"{name}: outer length changed"

            for i, (ra, rb) in enumerate(zip(a, b)):
                if isinstance(ra, list):
                    if len(ra) != len(rb):
                        return False, f"{name}[{i}]: row length changed"

                    for j, (xa, xb) in enumerate(zip(ra, rb)):
                        if xa != xb:
                            return False, (
                                f"{name}[{i}][{j}] changed: "
                                f"{xa} -> {xb}"
                            )
                else:
                    if ra != rb:
                        return False, (
                            f"{name}[{i}] changed: {ra} -> {rb}"
                        )
        else:
            if a != b:
                return False, f"{name} changed"

    return True, "all parameter values identical"


def main():
    print("=" * 72)
    print("BPE -> CHAT SFT -> CAUSAL LM -> LOSS -> BACKWARD")
    print("NO WEIGHT UPDATE CONTRACT")
    print("=" * 72)

    # ============================================================
    # 1. BPE
    # ============================================================
    print()
    print("[1/8] Loading BPE tokenizer")

    tokenizer = NeuraBPETokenizer(TOKENIZER_PATH)

    print("[PASS] BPE tokenizer")
    print("Vocab:", tokenizer.vocab_size)

    # ============================================================
    # 2. Chat SFT
    # ============================================================
    print()
    print("[2/8] Building Chat SFT example")

    builder = ChatSFTBuilderBPE(
        tokenizer,
        max_length=128,
        add_bos=True,
        add_eos=False,
    )

    messages = [
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
    ]

    sample = builder.build(messages)

    # ChatSFTExample is a dataclass.
    input_ids = sample.input_ids
    labels = sample.labels
    loss_mask = sample.loss_mask

    print("[PASS] Chat SFT example")
    print("Sequence:", len(input_ids))

    assistant_tokens = sum(
        1 for x in loss_mask if x == 1
    )

    ignored_tokens = sum(
        1 for x in loss_mask if x == 0
    )

    print("Assistant tokens:", assistant_tokens)
    print("Ignored tokens:", ignored_tokens)

    if assistant_tokens <= 0:
        raise RuntimeError(
            "No assistant tokens available for loss"
        )

    if len(input_ids) != len(labels):
        raise RuntimeError(
            "input_ids / labels length mismatch"
        )

    if len(input_ids) != len(loss_mask):
        raise RuntimeError(
            "input_ids / loss_mask length mismatch"
        )

    # Verify the SFT mask contract.
    for i, (label, mask) in enumerate(
        zip(labels, loss_mask)
    ):
        expected = 1 if label != -100 else 0

        if mask != expected:
            raise RuntimeError(
                f"SFT mask mismatch at index {i}: "
                f"label={label}, mask={mask}"
            )

    print("[PASS] Assistant-only loss mask")

    # ============================================================
    # 3. Causal alignment
    # ============================================================
    print()
    print("[3/8] Preparing causal targets")

    model_input = input_ids[:-1]
    model_target = labels[1:]

    if len(model_input) != len(model_target):
        raise RuntimeError(
            "Causal input/target length mismatch"
        )

    print("[PASS] Causal target alignment")
    print("Input tokens:", len(model_input))
    print("Target tokens:", len(model_target))

    # ============================================================
    # 4. Model
    # ============================================================
    print()
    print("[4/8] Constructing CausalLM")

    model = CausalLM(
        vocab_size=tokenizer.vocab_size,
        hidden_size=128,
        heads=4,
        num_layers=2,
        intermediate_size=512,
        context_length=128,
    )

    parameter_count = model.parameter_count()

    print("[PASS] CausalLM constructed")
    print("Parameters:", parameter_count)

    if parameter_count != 1049216:
        raise RuntimeError(
            f"Unexpected parameter count: "
            f"{parameter_count}"
        )

    # ============================================================
    # 5. Snapshot BEFORE
    # ============================================================
    print()
    print("[5/8] Taking parameter snapshot BEFORE")

    before = snapshot_parameters()

    if len(before) != 21:
        raise RuntimeError(
            f"Expected 21 registry entries, "
            f"got {len(before)}"
        )

    print("[PASS] Snapshot BEFORE")
    print("Registry entries:", len(before))

    # ============================================================
    # 6. Forward
    # ============================================================
    print()
    print("[6/8] Forward pass")

    logits = model.forward(model_input)

    if not isinstance(logits, list):
        raise RuntimeError(
            f"Expected logits list, "
            f"got {type(logits).__name__}"
        )

    if len(logits) != len(model_input):
        raise RuntimeError(
            f"Logit rows {len(logits)} != "
            f"input length {len(model_input)}"
        )

    print("[PASS] Forward pass")
    print("Logit rows:", len(logits))

    # ============================================================
    # 7. Loss + gradients + backward
    # ============================================================
    print()
    print("[7/8] Loss + gradients + backward")

    loss_fn = CrossEntropyLoss()

    # ChatSFTExample.labels already contains -100
    # for system/user tokens.
    loss = loss_fn.forward(
        logits,
        model_target,
    )

    if not math.isfinite(loss):
        raise RuntimeError(
            f"Loss is not finite: {loss}"
        )

    gradients = loss_fn.gradients()

    if gradients is None:
        raise RuntimeError(
            "CrossEntropyLoss.gradients() returned None"
        )

    print("[PASS] Loss computation")
    print("Loss tokens:", assistant_tokens)
    print("Loss:", loss)

    # CausalLM.backward() takes no explicit argument.
    # It consumes the gradient state/cache prepared by
    # the forward/loss pipeline.
    model.backward()

    print("[PASS] Backward pass")

    # ============================================================
    # 8. Snapshot AFTER + exact comparison
    # ============================================================
    print()
    print("[8/8] Verifying parameter immutability")

    after = snapshot_parameters()

    unchanged, reason = compare_parameters(
        before,
        after,
    )

    if not unchanged:
        print()
        print("=" * 72)
        print("===== WEIGHT UPDATE DETECTED =====")
        print(reason)
        print("=" * 72)
        raise RuntimeError(reason)

    print("[PASS] Parameter registry unchanged")
    print("All 21 parameter tensors are exactly identical")

    print()
    print("=" * 72)
    print("===== SUCCESS =====")
    print("BPE: PASS")
    print("CHAT SFT: PASS")
    print("ASSISTANT-ONLY MASK: PASS")
    print("CAUSAL LM FORWARD: PASS")
    print("LOSS: PASS")
    print("BACKWARD: PASS")
    print("NO WEIGHT UPDATE: PASS")
    print("===== NO TRAINING WAS PERFORMED =====")
    print("=" * 72)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 72)
        print("===== TEST FAILED =====")
        print(type(exc).__name__, str(exc))
        print("=" * 72)
        raise
