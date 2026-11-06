"""
neura.core

Core infrastructure for the Neura AI Platform.

This package provides:
- Version information
- Configuration
- Logging
- Device detection
- Random seed management
- Checkpoint management
- Performance profiling
- Distributed execution utilities
"""

from __future__ import annotations

from .checkpoint import (
    build_checkpoint,
    checkpoint_exists,
    load_checkpoint,
    save_checkpoint,
)
from .config import (
    ModelConfig,
    NeuraConfig,
    RuntimeConfig,
    TrainingConfig,
    create_default_config,
    validate_config,
)
from .device import (
    DeviceInfo,
    detect_device,
    get_device_info,
    get_device_name,
)
from .distributed import (
    DistributedInfo,
    get_backend,
    get_distributed_info,
    get_rank,
    get_world_size,
    is_distributed_available,
    is_distributed_initialized,
    is_main_process,
)
from .logger import (
    create_logger,
    get_logger,
    log_system_info,
    set_log_level,
)
from .profiler import (
    ProfileResult,
    Timer,
    measure,
    profile_function,
)
from .seed import (
    set_global_seed,
    set_numpy_seed,
    set_python_seed,
    set_seed,
    set_torch_seed,
)
from .version import (
    VERSION,
    VERSION_INFO,
    banner,
    get_full_version,
    get_version,
)

__all__ = [
    "VERSION",
    "VERSION_INFO",
    "banner",
    "get_version",
    "get_full_version",
    "ModelConfig",
    "TrainingConfig",
    "RuntimeConfig",
    "NeuraConfig",
    "create_default_config",
    "validate_config",
    "DeviceInfo",
    "detect_device",
    "get_device_info",
    "get_device_name",
    "create_logger",
    "get_logger",
    "set_log_level",
    "log_system_info",
    "ProfileResult",
    "Timer",
    "measure",
    "profile_function",
    "set_python_seed",
    "set_numpy_seed",
    "set_torch_seed",
    "set_seed",
    "set_global_seed",
    "build_checkpoint",
    "save_checkpoint",
    "load_checkpoint",
    "checkpoint_exists",
    "DistributedInfo",
    "get_distributed_info",
    "is_distributed_available",
    "is_distributed_initialized",
    "is_main_process",
    "get_world_size",
    "get_rank",
    "get_backend",
]
