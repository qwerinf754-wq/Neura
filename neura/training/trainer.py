"""
Neura Trainer
Reference training loop for CausalLM.

Pipeline:
Dataset -> DataLoader -> CausalLM -> Loss -> Backward
-> Gradient Clipping -> AdamW -> LR Scheduler
"""

from ..model.parameters import PARAMETERS
from .gradient_clipping import clip_gradients, global_gradient_norm
from .scheduler import WarmupCosineScheduler


class Trainer:

    def __init__(
        self,
        model,
        dataloader,
        optimizer,
        epochs=1,
        max_grad_norm=1.0,
        warmup_steps=0,
        min_learning_rate=0.0,
    ):
        if epochs <= 0:
            raise ValueError("epochs must be positive")

        if max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")

        if warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")

        if min_learning_rate < 0:
            raise ValueError(
                "min_learning_rate must be >= 0"
            )

        if min_learning_rate > optimizer.learning_rate:
            raise ValueError(
                "min_learning_rate cannot exceed "
                "initial learning rate"
            )

        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer

        self.epochs = epochs
        self.max_grad_norm = max_grad_norm

        self.global_step = 0

        # Total optimizer steps.
        self.total_steps = (
            self.epochs * len(self.dataloader)
        )

        self.scheduler = WarmupCosineScheduler(
            optimizer=self.optimizer,
            warmup_steps=warmup_steps,
            total_steps=self.total_steps,
            min_learning_rate=min_learning_rate,
        )

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------

    def _scale_value(self, value, scale):

        if isinstance(value, (int, float)):
            return value * scale

        if isinstance(value, list):
            return [
                self._scale_value(x, scale)
                for x in value
            ]

        if isinstance(value, tuple):
            return tuple(
                self._scale_value(x, scale)
                for x in value
            )

        raise TypeError(
            f"Unsupported gradient type: "
            f"{type(value).__name__}"
        )

    def _average_gradients(self, count):

        if count <= 0:
            raise ValueError(
                "gradient count must be positive"
            )

        scale = 1.0 / count

        for name in PARAMETERS.names():

            gradient = PARAMETERS.gradient(name)

            if gradient is None:
                continue

            PARAMETERS.set_gradient(
                name,
                self._scale_value(
                    gradient,
                    scale,
                ),
            )

    # ---------------------------------------------------------
    # Training Step
    # ---------------------------------------------------------

    def train_step(self, batch):

        input_batch = batch["input_ids"]

        if not input_batch:
            raise ValueError("Empty batch")

        PARAMETERS.zero_grad()

        total_loss = 0.0
        sample_count = 0

        # Accumulate gradients over samples.
        for input_ids in input_batch:

            loss = self.model.loss(input_ids)

            self.model.backward()

            total_loss += loss
            sample_count += 1

        # Average batch gradients.
        self._average_gradients(sample_count)

        gradients = {
            name: PARAMETERS.gradient(name)
            for name in PARAMETERS.names()
            if PARAMETERS.gradient(name) is not None
        }

        # Norm before clipping.
        grad_norm = global_gradient_norm(
            gradients
        )

        # Clip in-place.
        clipped = clip_gradients(
            gradients,
            self.max_grad_norm,
        )

        # Write clipped gradients back.
        for name, gradient in gradients.items():

            PARAMETERS.set_gradient(
                name,
                gradient,
            )

        # Optimizer update.
        self.optimizer.step(PARAMETERS)

        # Scheduler update AFTER optimizer step.
        learning_rate = self.scheduler.step()

        self.global_step += 1

        return {
            "loss": total_loss / sample_count,
            "grad_norm": grad_norm,
            "clipped": clipped,
            "lr": learning_rate,
            "step": self.global_step,
        }

    # ---------------------------------------------------------
    # Full Training
    # ---------------------------------------------------------

    def train(self):

        history = []

        for epoch in range(1, self.epochs + 1):

            epoch_loss = 0.0
            batches = 0

            for batch in self.dataloader:

                metrics = self.train_step(batch)

                epoch_loss += metrics["loss"]
                batches += 1

                print(
                    f"epoch={epoch} "
                    f"step={metrics['step']} "
                    f"loss={metrics['loss']:.6f} "
                    f"lr={metrics['lr']:.8f} "
                    f"grad_norm={metrics['grad_norm']:.6f} "
                    f"clipped={metrics['clipped']}"
                )

            if batches == 0:
                raise ValueError(
                    "DataLoader produced zero batches"
                )

            average_loss = epoch_loss / batches

            epoch_metrics = {
                "epoch": epoch,
                "loss": average_loss,
                "steps": batches,
                "lr": self.scheduler.get_lr(),
            }

            history.append(epoch_metrics)

            print(
                f"Epoch {epoch} complete | "
                f"loss={average_loss:.6f} | "
                f"lr={self.scheduler.get_lr():.8f}"
            )

        return history


# -------------------------------------------------------------
# Self Test
# -------------------------------------------------------------

if __name__ == "__main__":

    from ..model.causal_lm import CausalLM
    from .dataset import CausalLMDataset
    from .dataloader import CausalLMDataLoader
    from .optimizer import AdamW

    print("Neura Trainer")
    print("=============")

    PARAMETERS.weights.clear()
    PARAMETERS.grads.clear()

    model = CausalLM(
        vocab_size=16,
        hidden_size=8,
        heads=2,
        num_layers=1,
        context_length=8,
    )

    token_ids = list(range(16))

    dataset = CausalLMDataset(
        token_ids=token_ids,
        sequence_length=4,
        stride=4,
    )

    dataloader = CausalLMDataLoader(
        dataset=dataset,
        batch_size=2,
        shuffle=False,
    )

    optimizer = AdamW(
        learning_rate=0.001,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        epochs=2,
        max_grad_norm=1.0,
        warmup_steps=1,
        min_learning_rate=0.0001,
    )

    # 2 epochs * 2 batches = 4 steps.
    assert trainer.total_steps == 4

    history = trainer.train()

    print()
    print("History:")
    print(history)

    assert len(history) == 2
    assert history[0]["steps"] == 2
    assert history[1]["steps"] == 2

    assert history[0]["loss"] > 0.0
    assert history[1]["loss"] > 0.0

    assert optimizer.step_count == 4
    assert trainer.global_step == 4
    assert trainer.scheduler.step_count == 4

    assert abs(
        trainer.scheduler.get_lr() - 0.0001
    ) < 1e-12

    print()
    print("TRAINER + SCHEDULER TEST: PASS")
