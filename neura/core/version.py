"""
Neura AI Platform
=================

Core version information.

This module defines the official version metadata
for the Neura project.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import platform
import sys


# ---------------------------------------------------------------------
# Project Information
# ---------------------------------------------------------------------

PROJECT_NAME = "Neura"
ENGINE_NAME = "Neura Core"

VERSION = "0.1.0"
STAGE = "Alpha"

AUTHOR = "Neura Team"
LICENSE = "Apache-2.0"

HOMEPAGE = "https://github.com/neura-ai/neura"
REPOSITORY = "https://github.com/neura-ai/neura"

MINIMUM_PYTHON = (3, 12)


# ---------------------------------------------------------------------
# Version Data
# ---------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class VersionInfo:
    """Immutable version information."""

    project: str
    engine: str
    version: str
    stage: str
    author: str
    license: str
    homepage: str
    repository: str
    python_version: str
    operating_system: str
    architecture: str
    build_date: str


VERSION_INFO = VersionInfo(
    project=PROJECT_NAME,
    engine=ENGINE_NAME,
    version=VERSION,
    stage=STAGE,
    author=AUTHOR,
    license=LICENSE,
    homepage=HOMEPAGE,
    repository=REPOSITORY,
    python_version=platform.python_version(),
    operating_system=platform.system(),
    architecture=platform.machine(),
    build_date=datetime.now(UTC).strftime("%Y-%m-%d"),
)


def check_python_version() -> None:
    """Ensure the running Python version is supported."""

    if sys.version_info < MINIMUM_PYTHON:
        raise RuntimeError(
            f"Neura requires Python "
            f"{MINIMUM_PYTHON[0]}.{MINIMUM_PYTHON[1]} "
            f"or newer."
        )


def get_version() -> str:
    """Return the semantic version."""

    return VERSION_INFO.version


def get_full_version() -> str:
    """Return the complete version string."""

    return (
        f"{VERSION_INFO.project} "
        f"{VERSION_INFO.version} "
        f"({VERSION_INFO.stage})"
    )


def banner() -> str:
    """Return the Neura startup banner."""

    line = "=" * 60

    return f"""
{line}
                    NEURA AI PLATFORM
{line}

Project      : {VERSION_INFO.project}
Engine       : {VERSION_INFO.engine}
Version      : {VERSION_INFO.version}
Stage        : {VERSION_INFO.stage}

Python       : {VERSION_INFO.python_version}
OS           : {VERSION_INFO.operating_system}
Architecture : {VERSION_INFO.architecture}

Author       : {VERSION_INFO.author}
License      : {VERSION_INFO.license}

Homepage     : {VERSION_INFO.homepage}
Repository   : {VERSION_INFO.repository}

Build Date   : {VERSION_INFO.build_date}

{line}
"""


check_python_version()


if __name__ == "__main__":
    print(banner())
