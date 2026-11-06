"""
neura.core.device

Hardware device management for Neura AI Platform.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass


@dataclass(frozen=True)
class DeviceInfo:
    """
    Information about available hardware.
    """

    device_type: str
    system: str
    architecture: str
    available: bool
def detect_device() -> DeviceInfo:
    """
    Detect the best available execution device.

    Currently supports:
    - CPU detection
    - Future CUDA/MPS integration
    """

    system = platform.system()
    architecture = platform.machine()

    # Base implementation.
    # GPU backends will be added later.

    return DeviceInfo(
        device_type="cpu",
        system=system,
        architecture=architecture,
        available=True,
    )


def get_device_name() -> str:
    """
    Return current device name.
    """

    return detect_device().device_type


def get_device_info() -> DeviceInfo:
    """
    Return complete device information.
    """

    return detect_device()


if __name__ == "__main__":

    info = get_device_info()

    print(
        "Device:",
        info.device_type
    )

    print(
        "System:",
        info.system
    )

    print(
        "Architecture:",
        info.architecture
    )
