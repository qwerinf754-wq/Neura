"""
neura.core.distributed

Distributed execution utilities for the Neura AI Platform.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DistributedInfo:
    """
    Information about the distributed execution environment.
    """

    enabled: bool
    world_size: int
    rank: int
    backend: str


def get_distributed_info() -> DistributedInfo:
    """
    Detect the current distributed environment.

    If PyTorch distributed is unavailable or not initialized,
    Neura falls back to a single-process environment.
    """

    try:
        import torch.distributed as dist
    except ImportError:
        return DistributedInfo(
            enabled=False,
            world_size=1,
            rank=0,
            backend="none",
        )

    if not dist.is_available():
        return DistributedInfo(
            enabled=False,
            world_size=1,
            rank=0,
            backend="none",
        )

    if not dist.is_initialized():
        return DistributedInfo(
            enabled=False,
            world_size=1,
            rank=0,
            backend="none",
        )

    return DistributedInfo(
        enabled=True,
        world_size=dist.get_world_size(),
        rank=dist.get_rank(),
        backend=dist.get_backend(),
    )
def is_distributed_available() -> bool:
    """
    Return True when PyTorch distributed is available.
    """

    try:
        import torch.distributed as dist
    except ImportError:
        return False

    return dist.is_available()


def is_distributed_initialized() -> bool:
    """
    Return True when a distributed process group is initialized.
    """

    try:
        import torch.distributed as dist
    except ImportError:
        return False

    if not dist.is_available():
        return False

    return dist.is_initialized()


def is_main_process() -> bool:
    """
    Return True for the main process.

    In a non-distributed environment, the current process
    is always considered the main process.
    """

    info = get_distributed_info()

    return info.rank == 0


def get_world_size() -> int:
    """
    Return the number of distributed processes.
    """

    return get_distributed_info().world_size


def get_rank() -> int:
    """
    Return the current process rank.
    """

    return get_distributed_info().rank


def get_backend() -> str:
    """
    Return the active distributed backend.
    """

    return get_distributed_info().backend


def initialize_distributed(
    backend: str | None = None,
) -> DistributedInfo:
    """
    Initialize PyTorch distributed when possible.

    This function intentionally does not initialize a process
    group automatically. Distributed launch configuration will
    be handled by the training system later.

    Returns the current distributed environment information.
    """

    if is_distributed_initialized():
        return get_distributed_info()

    return DistributedInfo(
        enabled=False,
        world_size=1,
        rank=0,
        backend=backend or "none",
    )


if __name__ == "__main__":
    info = get_distributed_info()

    print("Distributed:", info.enabled)
    print("World Size:", info.world_size)
    print("Rank:", info.rank)
    print("Backend:", info.backend)
    print("Main Process:", is_main_process())
