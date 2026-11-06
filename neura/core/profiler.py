"""
neura.core.profiler

Performance profiling utilities for the Neura AI Platform.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter


@dataclass(frozen=True)
class ProfileResult:
    """
    Result of a profiling operation.
    """

    name: str
    elapsed_seconds: float

    @property
    def elapsed_milliseconds(self) -> float:
        """
        Return elapsed time in milliseconds.
        """

        return self.elapsed_seconds * 1000.0


class Timer:
    """
    High-resolution execution timer.
    """

    def __init__(self, name: str = "operation") -> None:
        self.name = name
        self._start_time: float | None = None
        self._elapsed_seconds: float = 0.0

    def start(self) -> None:
        """
        Start the timer.
        """

        self._start_time = perf_counter()

    def stop(self) -> ProfileResult:
        """
        Stop the timer and return the result.
        """

        if self._start_time is None:
            raise RuntimeError(
                "Timer has not been started."
            )

        self._elapsed_seconds = (
            perf_counter() - self._start_time
        )

        self._start_time = None

        return ProfileResult(
            name=self.name,
            elapsed_seconds=self._elapsed_seconds,
        )

    @property
    def elapsed_seconds(self) -> float:
        """
        Return the latest elapsed time in seconds.
        """

        return self._elapsed_seconds
def profile_function(
    name: str = "operation",
):
    """
    Decorator for measuring function execution time.
    """

    def decorator(function):
        def wrapper(*args, **kwargs):
            timer = Timer(name)
            timer.start()

            result = function(
                *args,
                **kwargs,
            )

            profile = timer.stop()

            print(
                f"[PROFILE] {profile.name}: "
                f"{profile.elapsed_milliseconds:.3f} ms"
            )

            return result

        return wrapper

    return decorator


def measure(
    name: str = "operation",
) -> Timer:
    """
    Create a new performance timer.
    """

    return Timer(name)


if __name__ == "__main__":
    timer = Timer("test_operation")

    timer.start()

    total = sum(range(100000))

    result = timer.stop()

    print("Profile:", result.name)
    print(
        "Elapsed:",
        f"{result.elapsed_milliseconds:.3f} ms",
    )
    print("Result:", total)
