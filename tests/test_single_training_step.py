import copy
import math

from neura.tokenizer.neura_bpe_tokenizer import NeuraBPETokenizer
from neura.training.chat_sft_bpe_dataset import ChatSFTBuilderBPE
from neura.training.loss import CrossEntropyLoss
from neura.model.causal_lm import CausalLM
from neura.model.parameters import PARAMETERS
from neura.training.optimizer import AdamW


TOKENIZER_PATH = (
    "data/processed/neura_arabic_bpe_32k/tokenizer.json"
)

VOCAB_SIZE = 2048
HIDDEN_SIZE = 128
HEADS = 4
NUM_LAYERS = 2
INTERMEDIATE_SIZE = 512
CONTEXT_LENGTH = 128

EXPECTED_PARAMETER_COUNT = 1_049_216
EXPECTED_REGISTRY_ENTRIES = 21

LR = 1e-4
WEIGHT_DECAY = 0.01


def snapshot_parameters():
    return copy.deepcopy(PARAMETERS.state_dict())


def snapshot_gradients():
    return copy.deepcopy(PARAMETERS.gradient_dict())


def is_finite_value(value):
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))

    if isinstance(value, list):
        return all(is_finite_value(x) for x in value)

    return True


def flatten_values(value):
    if isinstance(value, list):
        result = []
        for item in value:
            result.extend(flatten_values(item))
        return result

    return [value]


def parameter_difference(before, after):
    changed = []
    max_abs_delta = 0.0

    if before.keys() != after.keys():
        raise RuntimeError("Parameter registry keys changed")

    for name in before:
        old_values = flatten_values(before[name])
        new_values = flatten_values(after[name])

        if len(old_values) != len(new_values):
            raise RuntimeError(
                f"Parameter shape changed: {name}"
            )

        local_changed = False

        for old, new in zip(old_values, new_values):
            delta = abs(float(new) - float(old))

            if delta != 0.0:
                local_changed = True

            if delta > max_abs_delta:
                max_abs_delta = delta

        if local_changed:
            changed.append(name)

    return changed, max_abs_delta


def assert_parameters_equal(before, after, message):
    if before.keys() != after.keys():
        raise RuntimeError(
            f"{message}: parameter registry keys changed"
        )

    for name in before:
        old_values = flatten_values(before[name])
        new_values = flatten_values(after[name])

        if len(old_values) != len(new_values):
            raise RuntimeError(
                f"{message}: shape changed for {name}"
            )

        for old, new in zip(old_values, new_values):
            if old != new:
                raise RuntimeError(
                    f"{message}: parameter changed: {name}"
                )


def count_gradient_values(gradients):
    total = 0
    finite = 0
    nonzero = 0

    for name, gradient in gradients.items():
        values = flatten_values(gradient)

        for value in values:
            total += 1

            if not math.isfinite(float(value)):
                raise RuntimeError(
                    f"Non-finite gradient detected: {name}"
                )

            finite += 1

            if float(value) != 0.0:
                nonzero += 1

    return total, finite, nonzero


def main():
    print("=" * 72)
    print("NEURA SINGLE TRAINING STEP CONTRACT")
    print("FORWARD -> LOSS -> BACKWARD -> GRADIENTS")
    print("NO UPDATE BEFORE OPTIMIZER.STEP -> UPDATE AFTER STEP")
    print("=" * 72)

    # ---------------------------------------------------------------
    # 1. TOKENIZER
    # ---------------------------------------------------------------

    print()
    print("[1/10] BPE tokenizer")

    tokenizer = NeuraBPETokenizer(TOKENIZER_PATH)

    if tokenizer.vocab_size != VOCAB_SIZE:
        raise RuntimeError(
            f"Unexpected vocabulary size: {tokenizer.vocab_size}"
        )

    print("[PASS] Tokenizer")
    print("Vocab:", tokenizer.vocab_size)

    # ---------------------------------------------------------------
    # 2. CHAT SFT
    # ---------------------------------------------------------------

    print()
    print("[2/10] Chat SFT")

    builder = ChatSFTBuilderBPE(
        tokenizer,
        max_length=CONTEXT_LENGTH,
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
            "SFT sequence lengths are inconsistent"
        )

    assistant_tokens = sum(
        1 for x in loss_mask if x == 1
    )

    if assistant_tokens == 0:
        raise RuntimeError(
            "No assistant loss tokens"
        )

    for index, (label, mask) in enumerate(
        zip(labels, loss_mask)
    ):
        expected = 1 if label != -100 else 0

        if mask != expected:
            raise RuntimeError(
                f"Mask violation at {index}: "
                f"label={label}, mask={mask}"
            )

    print("[PASS] Chat SFT")
    print("Sequence:", len(input_ids))
    print("Assistant tokens:", assistant_tokens)

    # ---------------------------------------------------------------
    # 3. CAUSAL ALIGNMENT
    # ---------------------------------------------------------------

    print()
    print("[3/10] Causal alignment")

    model_input = input_ids[:-1]
    model_target = labels[1:]

    if len(model_input) != len(model_target):
        raise RuntimeError(
            "Causal input/target mismatch"
        )

    print("[PASS] Causal alignment")
    print("Input:", len(model_input))
    print("Target:", len(model_target))

    # ---------------------------------------------------------------
    # 4. MODEL
    # ---------------------------------------------------------------

    print()
    print("[4/10] CausalLM")

    model = CausalLM(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        heads=HEADS,
        num_layers=NUM_LAYERS,
        intermediate_size=INTERMEDIATE_SIZE,
        context_length=CONTEXT_LENGTH,
    )

    parameter_count = model.parameter_count()

    if parameter_count != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(
            f"Unexpected parameter count: "
            f"{parameter_count}"
        )

    if len(PARAMETERS.names()) != EXPECTED_REGISTRY_ENTRIES:
        raise RuntimeError(
            f"Unexpected registry size: "
            f"{len(PARAMETERS.names())}"
        )

    print("[PASS] CausalLM")
    print("Parameters:", parameter_count)
    print("Registry entries:", len(PARAMETERS.names()))

    # ---------------------------------------------------------------
    # 5. INITIAL FINITE CHECK
    # ---------------------------------------------------------------

    print()
    print("[5/10] Initial parameter finite check")

    initial = snapshot_parameters()

    for name, value in initial.items():
        if not is_finite_value(value):
            raise RuntimeError(
                f"Non-finite initial parameter: {name}"
            )

    print("[PASS] Initial parameters finite")

    # ---------------------------------------------------------------
    # 6. FORWARD + LOSS
    # ---------------------------------------------------------------

    print()
    print("[6/10] Forward + loss")

    logits = model.forward(model_input)

    if len(logits) != len(model_input):
        raise RuntimeError(
            "Logit length mismatch"
        )

    for row in logits:
        for value in row:
            if not math.isfinite(float(value)):
                raise RuntimeError(
                    "Non-finite logit detected"
                )

    loss_fn = CrossEntropyLoss()

    loss = loss_fn.forward(
        logits,
        model_target,
    )

    if not math.isfinite(float(loss)):
        raise RuntimeError(
            f"Non-finite loss: {loss}"
        )

    gradients_from_loss = loss_fn.gradients()

    if gradients_from_loss is None:
        raise RuntimeError(
            "Loss gradients are None"
        )

    print("[PASS] Forward")
    print("[PASS] Loss")
    print("Loss:", loss)

    # ---------------------------------------------------------------
    # 7. BACKWARD + GRADIENT CHECK
    # ---------------------------------------------------------------

    print()
    print("[7/10] Backward + gradients")

    backward_result = model.backward()

    if backward_result is None:
        raise RuntimeError(
            "CausalLM.backward() returned None"
        )

    gradients = snapshot_gradients()

    if not gradients:
        raise RuntimeError(
            "Parameter gradient registry is empty"
        )

    total_gradients, finite_gradients, nonzero_gradients = (
        count_gradient_values(gradients)
    )

    if total_gradients == 0:
        raise RuntimeError(
            "No gradient values found"
        )

    if finite_gradients != total_gradients:
        raise RuntimeError(
            "Some gradients are non-finite"
        )

    if nonzero_gradients == 0:
        raise RuntimeError(
            "All parameter gradients are zero"
        )

    print("[PASS] Backward")
    print("[PASS] Gradients exist")
    print("Gradient values:", total_gradients)
    print("Finite gradients:", finite_gradients)
    print("Non-zero gradients:", nonzero_gradients)

    # ---------------------------------------------------------------
    # 8. CRITICAL: NO UPDATE BEFORE OPTIMIZER STEP
    # ---------------------------------------------------------------

    print()
    print("[8/10] Verify NO parameter update before optimizer.step()")

    before_optimizer = snapshot_parameters()

    assert_parameters_equal(
        initial,
        before_optimizer,
        "Unexpected update before optimizer.step()",
    )

    print("[PASS] No parameter update before optimizer.step()")
    print("Changed parameters before step: 0")

    # ---------------------------------------------------------------
    # 9. OPTIMIZER STEP
    # ---------------------------------------------------------------

    print()
    print("[9/10] AdamW optimizer.step()")

    optimizer = AdamW(
        learning_rate=LR,
        weight_decay=WEIGHT_DECAY,
    )

    optimizer.step(PARAMETERS)

    after_optimizer = snapshot_parameters()

    # ---------------------------------------------------------------
    # 10. VERIFY UPDATE
    # ---------------------------------------------------------------

    print()
    print("[10/10] Verify optimizer update")

    changed, max_abs_delta = parameter_difference(
        before_optimizer,
        after_optimizer,
    )

    if not changed:
        raise RuntimeError(
            "Optimizer step completed but no parameters changed"
        )

    for name, value in after_optimizer.items():
        if not is_finite_value(value):
            raise RuntimeError(
                f"Non-finite parameter after optimizer.step(): "
                f"{name}"
            )

    if not math.isfinite(max_abs_delta):
        raise RuntimeError(
            "Non-finite parameter delta"
        )

    print("[PASS] Parameters changed after optimizer.step()")
    print("Changed parameters:", len(changed))
    print("Max absolute parameter delta:", max_abs_delta)

    print()
    print("=" * 72)
    print("===== SINGLE TRAINING STEP: PASS =====")
    print("=" * 72)

    print("BPE:                         PASS")
    print("CHAT SFT:                    PASS")
    print("CAUSAL ALIGNMENT:            PASS")
    print("FORWARD:                     PASS")
    print("LOSS:                        PASS")
    print("BACKWARD:                    PASS")
    print("GRADIENTS EXIST:             PASS")
    print("GRADIENTS FINITE:            PASS")
    print("NO UPDATE BEFORE STEP:       PASS")
    print("OPTIMIZER STEP:              PASS")
    print("WEIGHTS CHANGED AFTER STEP:  PASS")
    print("PARAMETERS FINITE:           PASS")

    print()
    print("ONE OPTIMIZER STEP ONLY")
    print("NO CHECKPOINT SAVE")
    print("NO LONG TRAINING")
    print("=" * 72)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 72)
        print("===== SINGLE TRAINING STEP: FAIL =====")
        print(type(exc).__name__, str(exc))
        print("=" * 72)
        raise
