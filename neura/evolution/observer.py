"""Project observer for Neura Evolution Engine.

Collects read-only information about the current project state.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class ProjectSnapshot:
    python_version: str
    project_root: str
    git_branch: str | None
    git_commit: str | None
    git_status: list[str]
    evolution_version: str
    python_files: int
    checkpoint_files: int
    tokenizer_files: int
    test_files: int


class ProjectObserver:
    """Read-only observer of the Neura project."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()

    def _git(self, *args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=self.root,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None

        if result.returncode != 0:
            return None

        return result.stdout.strip()

    def _count_files(self, directory: str, suffix: str | None = None) -> int:
        path = self.root / directory

        if not path.exists():
            return 0

        count = 0

        for item in path.rglob("*"):
            if not item.is_file():
                continue

            if suffix is not None and item.suffix != suffix:
                continue

            count += 1

        return count

    def collect(self) -> dict[str, Any]:
        snapshot = ProjectSnapshot(
            python_version=sys.version.split()[0],
            project_root=str(self.root),
            git_branch=self._git("branch", "--show-current"),
            git_commit=self._git("rev-parse", "HEAD"),
            git_status=(
                self._git("status", "--short") or ""
            ).splitlines(),
            evolution_version="0.1.0",
            python_files=self._count_files("neura", ".py"),
            checkpoint_files=self._count_files("checkpoints"),
            tokenizer_files=self._count_files("data", ".json"),
            test_files=self._count_files("tests", ".py"),
        )

        return asdict(snapshot)
