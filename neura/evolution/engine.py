"""Neura Evolution Engine - Phase 1.

Observation-only evolutionary infrastructure.

This module intentionally does NOT modify project files.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .metrics import generation_metrics
from .observer import ProjectObserver
from .policy import DEFAULT_POLICY, EvolutionPolicy


class EvolutionEngine:
    """Collects project state and records evolutionary observations."""

    VERSION = "0.1.0"

    def __init__(
        self,
        root: str | Path,
        policy: EvolutionPolicy = DEFAULT_POLICY,
    ):
        self.root = Path(root).resolve()
        self.policy = policy

        self.observer = ProjectObserver(self.root)

        self.history_dir = (
            self.root / "data" / "evolution" / "history"
        )
        self.metrics_dir = (
            self.root / "data" / "evolution" / "metrics"
        )

        self.history_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

    @property
    def history_file(self) -> Path:
        return self.history_dir / "observations.jsonl"

    def observe(self) -> dict[str, Any]:
        """Collect one immutable project observation."""

        observation = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "engine_version": self.VERSION,
            "mode": "observation_only",
            "policy": {
                "source_modification": (
                    self.policy.allow_source_modification
                ),
                "checkpoint_modification": (
                    self.policy.allow_checkpoint_modification
                ),
                "config_modification": (
                    self.policy.allow_config_modification
                ),
                "dependency_installation": (
                    self.policy.allow_dependency_installation
                ),
                "file_deletion": (
                    self.policy.allow_file_deletion
                ),
                "shell_execution": (
                    self.policy.allow_shell_execution
                ),
            },
            "snapshot": self.observer.collect(),
        }

        self._append_history(observation)

        return observation

    def record_generation(
        self,
        prompt: str,
        response: str,
    ) -> dict[str, Any]:
        """Record generation metrics without changing generation behavior."""

        result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "engine_version": self.VERSION,
            "prompt_metrics": generation_metrics(prompt),
            "response_metrics": generation_metrics(response),
        }

        self._append_history({
            "type": "generation_observation",
            **result,
        })

        return result

    def _append_history(self, data: dict[str, Any]) -> None:
        with self.history_file.open(
            "a",
            encoding="utf-8",
        ) as handle:
            handle.write(
                json.dumps(
                    data,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )


def main() -> int:
    root = Path(__file__).resolve().parents[2]

    engine = EvolutionEngine(root)
    observation = engine.observe()

    print("=" * 70)
    print("NEURA EVOLUTION ENGINE")
    print("=" * 70)
    print("Version:", engine.VERSION)
    print("Mode:", observation["mode"])
    print()
    print("Project:")
    print(" ", observation["snapshot"]["project_root"])
    print()
    print("Python:")
    print(" ", observation["snapshot"]["python_version"])
    print()
    print("Git branch:")
    print(" ", observation["snapshot"]["git_branch"])
    print()
    print("Git commit:")
    print(" ", observation["snapshot"]["git_commit"])
    print()
    print("Python files:")
    print(" ", observation["snapshot"]["python_files"])
    print("Checkpoint files:")
    print(" ", observation["snapshot"]["checkpoint_files"])
    print("Tokenizer JSON files:")
    print(" ", observation["snapshot"]["tokenizer_files"])
    print("Test files:")
    print(" ", observation["snapshot"]["test_files"])
    print()
    print("Automatic modification: DISABLED")
    print("Observation recorded:")
    print(" ", engine.history_file)
    print("=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
