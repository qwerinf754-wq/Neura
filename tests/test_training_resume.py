from neura.model.parameters import PARAMETERS
from neura.model.causal_lm import CausalLM
from neura.training.dataset import CausalLMDataset
from neura.training.dataloader import CausalLMDataLoader
from neura.training.optimizer import AdamW
from neura.training.checkpoint import CheckpointManager
from neura.training.scheduler import WarmupCosineScheduler
from neura.training.trainer import Trainer


def build():
    PARAMETERS.clear()

    model = CausalLM(
        vocab_size=16,
        hidden_size=8,
        heads=2,
        num_layers=1,
        context_length=8,
    )

    tokens = list(range(16))
    dataset = CausalLMDataset(tokens, sequence_length=4)
    loader = CausalLMDataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
    )

    optimizer = AdamW(
        learning_rate=0.001,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        dataloader=loader,
        optimizer=optimizer,
        epochs=1,
        max_grad_norm=1.0,
        warmup_steps=1,
    )

    return model, loader, optimizer, trainer


print("Neura Training Resume Test")
print("==========================")

# ---------------------------------------------------------
# First training session
# ---------------------------------------------------------

model, loader, optimizer, trainer = build()

history = trainer.train()

manager = CheckpointManager("checkpoints/test_resume")

manager.save(
    "resume_test",
    PARAMETERS,
    optimizer=optimizer,
    scheduler=trainer.scheduler,
    epoch=1,
    global_step=trainer.global_step,
    metadata={"test": "resume"},
)

saved_step = trainer.global_step
saved_optimizer_step = optimizer.step_count
saved_scheduler_step = trainer.scheduler.step_count

print(f"Saved global step     : {saved_step}")
print(f"Saved optimizer step  : {saved_optimizer_step}")
print(f"Saved scheduler step  : {saved_scheduler_step}")

# ---------------------------------------------------------
# Destroy current state
# ---------------------------------------------------------

PARAMETERS.clear()

# ---------------------------------------------------------
# Rebuild model/training state
# ---------------------------------------------------------

model2, loader2, optimizer2, trainer2 = build()

# Trainer creates its own scheduler.
# Load the checkpoint into the rebuilt state.

restored = manager.load(
    "resume_test",
    PARAMETERS,
    optimizer=optimizer2,
    scheduler=trainer2.scheduler,
)

trainer2.global_step = restored["global_step"]

print(f"Restored global step  : {trainer2.global_step}")
print(f"Restored optimizer step: {optimizer2.step_count}")
print(f"Restored scheduler step: {trainer2.scheduler.step_count}")

# ---------------------------------------------------------
# Assertions
# ---------------------------------------------------------

assert trainer2.global_step == saved_step
assert optimizer2.step_count == saved_optimizer_step
assert trainer2.scheduler.step_count == saved_scheduler_step

assert restored["epoch"] == 1
assert restored["metadata"]["test"] == "resume"

print()
print("CHECKPOINT RESTORE: PASS")

# ---------------------------------------------------------
# Continue training
# ---------------------------------------------------------

trainer2.epochs = 2
trainer2.total_steps = (
    trainer2.epochs * len(loader2)
)

trainer2.scheduler.total_steps = trainer2.total_steps

history2 = trainer2.train()

assert trainer2.global_step > saved_step
assert optimizer2.step_count > saved_optimizer_step
assert trainer2.scheduler.step_count > saved_scheduler_step

print()
print(f"Final global step     : {trainer2.global_step}")
print(f"Final optimizer step  : {optimizer2.step_count}")
print(f"Final scheduler step  : {trainer2.scheduler.step_count}")

print()
print("RESUME TRAINING TEST: PASS")
