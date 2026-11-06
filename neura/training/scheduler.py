"""
Neura Learning Rate Scheduler
Linear Warmup + Cosine Decay
"""

import math


class WarmupCosineScheduler:

    def __init__(
        self,
        optimizer,
        warmup_steps,
        total_steps,
        min_learning_rate=0.0,
    ):
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")

        if total_steps <= 0:
            raise ValueError("total_steps must be > 0")

        if warmup_steps > total_steps:
            raise ValueError(
                "warmup_steps cannot exceed total_steps"
            )

        if min_learning_rate < 0:
            raise ValueError(
                "min_learning_rate must be >= 0"
            )

        if min_learning_rate > optimizer.learning_rate:
            raise ValueError(
                "min_learning_rate cannot exceed "
                "initial learning rate"
            )

        self.optimizer = optimizer

        self.base_learning_rate = optimizer.learning_rate
        self.min_learning_rate = min_learning_rate

        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

        self.step_count = 0

        self._set_learning_rate(
            self._calculate_learning_rate(0)
        )

    # ---------------------------------------------------------
    # Learning Rate
    # ---------------------------------------------------------

    def _calculate_learning_rate(self, step):
        if step < 0:
            step = 0

        if step <= self.warmup_steps and self.warmup_steps > 0:
            return (
                self.base_learning_rate
                * step
                / self.warmup_steps
            )

        if step >= self.total_steps:
            return self.min_learning_rate

        decay_steps = (
            self.total_steps
            - self.warmup_steps
        )

        progress = (
            step - self.warmup_steps
        ) / decay_steps

        cosine = 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

        return (
            self.min_learning_rate
            + (
                self.base_learning_rate
                - self.min_learning_rate
            )
            * cosine
        )

    def _set_learning_rate(self, learning_rate):
        self.optimizer.learning_rate = learning_rate

    def get_lr(self):
        return self.optimizer.learning_rate

    # ---------------------------------------------------------
    # Step
    # ---------------------------------------------------------

    def step(self):
        self.step_count += 1

        learning_rate = self._calculate_learning_rate(
            self.step_count
        )

        self._set_learning_rate(learning_rate)

        return learning_rate

    # ---------------------------------------------------------
    # State
    # ---------------------------------------------------------

    def state_dict(self):
        return {
            "step_count": self.step_count,
            "base_learning_rate": self.base_learning_rate,
            "min_learning_rate": self.min_learning_rate,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
        }

    def load_state_dict(self, state):
        self.step_count = int(
            state["step_count"]
        )

        self._set_learning_rate(
            self._calculate_learning_rate(
                self.step_count
            )
        )


# -------------------------------------------------------------
# Self Test
# -------------------------------------------------------------

if __name__ == "__main__":

    from .optimizer import AdamW

    print("Neura Warmup + Cosine Scheduler")
    print("================================")

    optimizer = AdamW(
        learning_rate=0.001,
    )

    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        warmup_steps=2,
        total_steps=10,
        min_learning_rate=0.0001,
    )

    expected = [
        0.0000,
        0.0005,
        0.0010,
    ]

    for step in range(11):

        lr = scheduler.get_lr()

        print(
            f"step={step:02d} "
            f"lr={lr:.8f}"
        )

        if step < len(expected):
            assert abs(lr - expected[step]) < 1e-12

        if step < 10:
            scheduler.step()

    assert abs(
        scheduler.get_lr() - 0.0001
    ) < 1e-12

    state = scheduler.state_dict()

    scheduler2 = WarmupCosineScheduler(
        optimizer=AdamW(
            learning_rate=0.001,
        ),
        warmup_steps=2,
        total_steps=10,
        min_learning_rate=0.0001,
    )

    scheduler2.load_state_dict(state)

    assert (
        scheduler2.step_count
        == scheduler.step_count
    )

    assert abs(
        scheduler2.get_lr()
        - scheduler.get_lr()
    ) < 1e-12

    print()
    print("SCHEDULER TEST: PASS")
