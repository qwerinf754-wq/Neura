"""
Neura Checkpoint Manager
"""

import os
import pickle
import tempfile


class CheckpointManager:

    FORMAT_VERSION = 1

    def __init__(self, directory="checkpoints"):
        self.directory = directory
        os.makedirs(self.directory, exist_ok=True)

    # ---------------------------------------------------------
    # Path
    # ---------------------------------------------------------

    def _path(self, name):
        if not name:
            raise ValueError("Checkpoint name cannot be empty")

        if os.path.basename(name) != name:
            raise ValueError(
                "Checkpoint name must not contain directories"
            )

        return os.path.join(
            self.directory,
            f"{name}.ckpt",
        )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    def save(
        self,
        name,
        parameters,
        optimizer=None,
        scheduler=None,
        epoch=0,
        global_step=0,
        metadata=None,
    ):

        path = self._path(name)

        state = {
            "format_version": self.FORMAT_VERSION,

            "parameters": parameters.state_dict(),

            "optimizer": (
                optimizer.state_dict()
                if optimizer is not None
                and hasattr(optimizer, "state_dict")
                else self._optimizer_state(optimizer)
            ),

            "scheduler": (
                scheduler.state_dict()
                if scheduler is not None
                and hasattr(scheduler, "state_dict")
                else None
            ),

            "epoch": int(epoch),
            "global_step": int(global_step),

            "metadata": metadata or {},
        }

        # Atomic save:
        # write temporary file -> replace final file.
        fd, temp_path = tempfile.mkstemp(
            prefix=".checkpoint_",
            suffix=".tmp",
            dir=self.directory,
        )

        try:
            with os.fdopen(fd, "wb") as file:
                pickle.dump(
                    state,
                    file,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
                file.flush()
                os.fsync(file.fileno())

            os.replace(
                temp_path,
                path,
            )

        except Exception:
            try:
                os.remove(temp_path)
            except OSError:
                pass
            raise

        return path

    # ---------------------------------------------------------
    # Optimizer Compatibility
    # ---------------------------------------------------------

    def _optimizer_state(self, optimizer):

        if optimizer is None:
            return None

        state = {
            "learning_rate": optimizer.learning_rate,
            "weight_decay": optimizer.weight_decay,
            "beta1": optimizer.beta1,
            "beta2": optimizer.beta2,
            "epsilon": optimizer.epsilon,
            "step_count": optimizer.step_count,
            "m": optimizer.m,
            "v": optimizer.v,
        }

        return state

    def _load_optimizer_state(
        self,
        optimizer,
        state,
    ):

        if optimizer is None or state is None:
            return

        optimizer.learning_rate = state["learning_rate"]
        optimizer.weight_decay = state["weight_decay"]
        optimizer.beta1 = state["beta1"]
        optimizer.beta2 = state["beta2"]
        optimizer.epsilon = state["epsilon"]
        optimizer.step_count = state["step_count"]
        optimizer.m = state["m"]
        optimizer.v = state["v"]

    # ---------------------------------------------------------
    # Load
    # ---------------------------------------------------------

    def load(
        self,
        name,
        parameters,
        optimizer=None,
        scheduler=None,
    ):

        path = self._path(name)

        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Checkpoint not found: {path}"
            )

        with open(path, "rb") as file:
            state = pickle.load(file)

        version = state.get("format_version")

        if version != self.FORMAT_VERSION:
            raise ValueError(
                f"Unsupported checkpoint version: {version}"
            )

        # Restore parameters.
        parameters.load_state_dict(
            state["parameters"]
        )

        # Restore optimizer.
        self._load_optimizer_state(
            optimizer,
            state.get("optimizer"),
        )

        # Restore scheduler.
        if (
            scheduler is not None
            and state.get("scheduler") is not None
        ):
            scheduler.load_state_dict(
                state["scheduler"]
            )

        return {
            "epoch": state["epoch"],
            "global_step": state["global_step"],
            "metadata": state.get(
                "metadata",
                {},
            ),
            "format_version": version,
            "path": path,
        }

    # ---------------------------------------------------------
    # Exists
    # ---------------------------------------------------------

    def exists(self, name):
        return os.path.isfile(
            self._path(name)
        )

    # ---------------------------------------------------------
    # Delete
    # ---------------------------------------------------------

    def delete(self, name):

        path = self._path(name)

        if os.path.isfile(path):
            os.remove(path)
            return True

        return False


# -------------------------------------------------------------
# Self Test
# -------------------------------------------------------------

if __name__ == "__main__":

    from ..model.parameters import PARAMETERS
    from .optimizer import AdamW
    from .scheduler import WarmupCosineScheduler

    print("Neura Checkpoint Manager")
    print("========================")

    PARAMETERS.weights.clear()
    PARAMETERS.grads.clear()

    # Create test parameters.
    PARAMETERS.add(
        "test.weight",
        [[1.0, 2.0], [3.0, 4.0]],
    )

    PARAMETERS.set_gradient(
        "test.weight",
        [[0.1, 0.2], [0.3, 0.4]],
    )

    optimizer = AdamW(
        learning_rate=0.001,
        weight_decay=0.01,
    )

    # Perform one optimizer step so m/v are populated.
    optimizer.step(PARAMETERS)

    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        warmup_steps=1,
        total_steps=4,
        min_learning_rate=0.0001,
    )

    scheduler.step()

    manager = CheckpointManager(
        directory="checkpoints/test",
    )

    save_path = manager.save(
        name="checkpoint_test",
        parameters=PARAMETERS,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=3,
        global_step=7,
        metadata={
            "model": "Neura-Test",
            "purpose": "checkpoint-test",
        },
    )

    print(f"Saved: {save_path}")

    assert manager.exists(
        "checkpoint_test"
    )

    # Save original state for comparison.
    expected_parameters = PARAMETERS.state_dict()
    expected_step = optimizer.step_count
    expected_scheduler_step = scheduler.step_count
    expected_epoch = 3
    expected_global_step = 7

    # Destroy current state.
    PARAMETERS.weights.clear()
    PARAMETERS.grads.clear()

    optimizer2 = AdamW(
        learning_rate=0.5,
        weight_decay=0.5,
    )

    scheduler2 = WarmupCosineScheduler(
        optimizer=optimizer2,
        warmup_steps=1,
        total_steps=4,
        min_learning_rate=0.0001,
    )

    restored = manager.load(
        name="checkpoint_test",
        parameters=PARAMETERS,
        optimizer=optimizer2,
        scheduler=scheduler2,
    )

    print(f"Restored epoch: {restored['epoch']}")
    print(
        f"Restored step: {restored['global_step']}"
    )
    print(
        f"Optimizer step: {optimizer2.step_count}"
    )
    print(
        f"Scheduler step: {scheduler2.step_count}"
    )

    assert PARAMETERS.state_dict() == expected_parameters

    assert (
        optimizer2.step_count
        == expected_step
    )

    assert (
        scheduler2.step_count
        == expected_scheduler_step
    )

    assert (
        restored["epoch"]
        == expected_epoch
    )

    assert (
        restored["global_step"]
        == expected_global_step
    )

    assert (
        restored["metadata"]["model"]
        == "Neura-Test"
    )

    # Cleanup test checkpoint.
    manager.delete("checkpoint_test")

    try:
        os.rmdir("checkpoints/test")
    except OSError:
        pass

    print()
    print("CHECKPOINT TEST: PASS")
