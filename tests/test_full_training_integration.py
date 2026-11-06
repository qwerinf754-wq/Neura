from __future__ import annotations

import math
import shutil
from pathlib import Path

from neura.model.parameters import PARAMETERS
from neura.model.causal_lm import CausalLM
from neura.tokenizer.vocabulary import Vocabulary
from neura.tokenizer.tokenizer import Tokenizer
from neura.training.dataset import CausalLMDataset
from neura.training.dataloader import CausalLMDataLoader
from neura.training.loss import CrossEntropyLoss
from neura.training.gradient_clipping import (
    global_gradient_norm,
    clip_gradients,
)
from neura.training.optimizer import AdamW
from neura.training.scheduler import WarmupCosineScheduler
from neura.training.checkpoint import CheckpointManager


ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = ROOT / "checkpoints" / "test_full_integration"


def reset_parameters():
    PARAMETERS.weights.clear()
    PARAMETERS.grads.clear()
    PARAMETERS.count = 0


def build_tokenizer():
    vocab = Vocabulary()

    # Keep the test vocabulary intentionally tiny.
    tokens = [
        "<PAD>",
        "<UNK>",
        "<BOS>",
        "<EOS>",
        "Neura",
        "AI",
        "is",
        "smart",
    ]

    for token in tokens:
        vocab.add(token)

    tokenizer = Tokenizer(vocab)
    tokenizer.freeze()

    return tokenizer


def build_model(vocab_size):
    return CausalLM(
        vocab_size=vocab_size,
        hidden_size=8,
        heads=2,
        num_layers=1,
        intermediate_size=16,
        context_length=8,
    )


def main():
    print("Neura Full Training Integration Test")
    print("====================================")

    # ---------------------------------------------------------
    # 1. Text -> Tokenizer -> IDs
    # ---------------------------------------------------------

    reset_parameters()

    tokenizer = build_tokenizer()

    text = "Neura AI is smart"
    token_ids = tokenizer.encode(text)

    assert token_ids
    assert all(isinstance(x, int) for x in token_ids)
    assert tokenizer.decode(token_ids) == text

    print("[PASS] Text -> Tokenizer -> Token IDs")
    print("       Text :", text)
    print("       IDs  :", token_ids)

    # ---------------------------------------------------------
    # 2. Dataset
    # ---------------------------------------------------------

    # Repeat enough tokens to produce several training samples.
    corpus_ids = token_ids * 4

    dataset = CausalLMDataset(
        corpus_ids,
        sequence_length=4,
        stride=4,
    )

    assert len(dataset) > 0

    for sample in dataset.samples():
        assert len(sample["input_ids"]) == 4
        assert len(sample["target_ids"]) == 4

    print("[PASS] Token IDs -> CausalLMDataset")
    print("       Samples:", len(dataset))

    # ---------------------------------------------------------
    # 3. DataLoader
    # ---------------------------------------------------------

    loader = CausalLMDataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
    )

    batches = list(loader)

    assert len(batches) > 0
    assert batches[0]["input_ids"]
    assert batches[0]["target_ids"]

    print("[PASS] Dataset -> CausalLMDataLoader")
    print("       Batches:", len(batches))

    # ---------------------------------------------------------
    # 4. Model
    # ---------------------------------------------------------

    model = build_model(tokenizer.vocab_size)

    assert model.vocab_size == tokenizer.vocab_size
    assert model.parameter_count() > 0

    print("[PASS] CausalLM construction")
    print("       Vocab      :", model.vocab_size)
    print("       Parameters :", model.parameter_count())

    # ---------------------------------------------------------
    # 5. Forward
    # ---------------------------------------------------------

    first_batch = batches[0]
    input_ids = first_batch["input_ids"][0]

    logits = model.forward(
        input_ids,
        use_cache=False,
    )

    assert len(logits) == len(input_ids)
    assert len(logits[0]) == tokenizer.vocab_size

    for row in logits:
        for value in row:
            assert math.isfinite(value)

    print("[PASS] CausalLM.forward()")
    print(
        "       Logits:",
        len(logits),
        "x",
        len(logits[0]),
    )

    # ---------------------------------------------------------
    # 6. Cross Entropy Loss
    # ---------------------------------------------------------

    # The generic CE loss is tested independently against the
    # model's next-token targets.
    ce = CrossEntropyLoss()

    targets = first_batch["target_ids"][0]

    ce_loss = ce.forward(
        logits,
        targets,
    )

    assert math.isfinite(ce_loss)
    assert ce_loss > 0.0

    ce_gradients = ce.gradients()

    assert len(ce_gradients) == len(logits)
    assert len(ce_gradients[0]) == tokenizer.vocab_size

    print("[PASS] CrossEntropyLoss")
    print("       Loss:", ce_loss)

    # The forward above was stored in the CausalLM training cache.
    # Clear it before performing the model's own causal backward test.
    model.clear_training_cache()

    # ---------------------------------------------------------
    # 7. CausalLM loss + backward
    # ---------------------------------------------------------

    loss_value = model.loss(input_ids)

    assert math.isfinite(loss_value)
    assert loss_value > 0.0

    model.backward()

    assert PARAMETERS.grads
    assert any(
        gradient is not None
        for gradient in PARAMETERS.grads.values()
    )

    print("[PASS] CausalLM loss + backward")

    # ---------------------------------------------------------
    # 8. Gradient norm + clipping
    # ---------------------------------------------------------

    before_norm = global_gradient_norm(
        PARAMETERS.grads
    )

    assert math.isfinite(before_norm)
    assert before_norm >= 0.0

    original_norm, clipped = clip_gradients(
        PARAMETERS.grads,
        max_norm=1.0,
    )

    after_norm = global_gradient_norm(
        PARAMETERS.grads
    )

    assert math.isfinite(original_norm)
    assert math.isfinite(after_norm)
    assert after_norm <= 1.0 + 1e-10

    if original_norm > 1.0:
        assert clipped is True

    print("[PASS] Gradient clipping")
    print("       Before:", before_norm)
    print("       After :", after_norm)

    # ---------------------------------------------------------
    # 9. Optimizer
    # ---------------------------------------------------------

    optimizer = AdamW(
        learning_rate=0.001,
        weight_decay=0.01,
    )

    optimizer_step_before = optimizer.step_count

    # Save one parameter before the optimizer update.
    parameter_names = PARAMETERS.names()
    assert parameter_names

    tracked_name = parameter_names[0]
    tracked_before = PARAMETERS.get(tracked_name)

    optimizer.step(PARAMETERS)

    assert optimizer.step_count == optimizer_step_before + 1

    tracked_after = PARAMETERS.get(tracked_name)

    assert tracked_before != tracked_after

    print("[PASS] AdamW optimizer")
    print("       Step:", optimizer.step_count)

    # ---------------------------------------------------------
    # 10. Scheduler
    # ---------------------------------------------------------

    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        warmup_steps=1,
        total_steps=4,
        min_learning_rate=0.0001,
    )

    initial_lr = scheduler.get_lr()

    scheduler.step()

    assert scheduler.step_count == 1
    assert math.isfinite(scheduler.get_lr())

    print("[PASS] WarmupCosineScheduler")
    print("       Initial LR:", initial_lr)
    print("       Current LR:", scheduler.get_lr())

    # ---------------------------------------------------------
    # 11. Checkpoint
    # ---------------------------------------------------------

    if CHECKPOINT_DIR.exists():
        shutil.rmtree(CHECKPOINT_DIR)

    manager = CheckpointManager(
        str(CHECKPOINT_DIR)
    )

    saved_global_step = 1

    manager.save(
        "full_integration",
        PARAMETERS,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=1,
        global_step=saved_global_step,
        metadata={
            "test": "full_training_integration",
            "text": text,
        },
    )

    print("[PASS] Checkpoint save")

    # ---------------------------------------------------------
    # 12. Restore into a fresh model/runtime
    # ---------------------------------------------------------

    reset_parameters()

    model2 = build_model(tokenizer.vocab_size)

    optimizer2 = AdamW(
        learning_rate=0.001,
        weight_decay=0.01,
    )

    scheduler2 = WarmupCosineScheduler(
        optimizer=optimizer2,
        warmup_steps=1,
        total_steps=4,
        min_learning_rate=0.0001,
    )

    restored = manager.load(
        "full_integration",
        PARAMETERS,
        optimizer=optimizer2,
        scheduler=scheduler2,
    )

    assert restored["epoch"] == 1
    assert restored["global_step"] == saved_global_step
    assert restored["metadata"]["test"] == (
        "full_training_integration"
    )

    assert optimizer2.step_count == optimizer.step_count
    assert scheduler2.step_count == scheduler.step_count

    print("[PASS] Checkpoint restore")
    print("       Global step:", restored["global_step"])
    print("       Optimizer step:", optimizer2.step_count)
    print("       Scheduler step:", scheduler2.step_count)

    # ---------------------------------------------------------
    # 13. Continue training after restore
    # ---------------------------------------------------------

    input_ids_2 = batches[1 % len(batches)]["input_ids"][0]

    continued_loss = model2.loss(input_ids_2)

    assert math.isfinite(continued_loss)
    assert continued_loss > 0.0

    model2.backward()

    assert PARAMETERS.grads

    clip_gradients(
        PARAMETERS.grads,
        max_norm=1.0,
    )

    optimizer2.step(PARAMETERS)
    scheduler2.step()

    assert optimizer2.step_count > optimizer.step_count
    assert scheduler2.step_count > scheduler.step_count

    print("[PASS] Continue training after restore")
    print("       Loss:", continued_loss)
    print("       Optimizer step:", optimizer2.step_count)
    print("       Scheduler step:", scheduler2.step_count)

    # ---------------------------------------------------------
    # Final
    # ---------------------------------------------------------

    print()
    print("====================================")
    print("FULL TRAINING INTEGRATION: PASS")
    print("====================================")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
