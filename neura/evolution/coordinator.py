"""Evolution coordinator with validation planning.

Phase 0.6:
Observation -> Diagnosis -> Proposal -> Validation Plan.

No automatic mutation is performed.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
from typing import Any

from .analyzer import EvolutionAnalyzer
from .observer import ProjectObserver
from .proposer import EvolutionProposer
from .validator import EvolutionValidator


class EvolutionCoordinator:
    """Coordinates safe evolution analysis without mutation."""

    VERSION = "0.6.0"

    def __init__(self, project_root: Path | None = None):
        self.project_root = (
            project_root
            if project_root is not None
            else Path.cwd()
        )

        self.observer = ProjectObserver(self.project_root)
        self.analyzer = EvolutionAnalyzer()
        self.proposer = EvolutionProposer()
        self.validator = EvolutionValidator()

        self.history_dir = (
            self.project_root
            / "data"
            / "evolution"
            / "history"
        )

        self.history_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def inspect_generation(
        self,
        prompt: str,
        response: str,
    ) -> dict[str, Any]:

        observation = self.observer.collect()

        analysis = self.analyzer.analyze(
            prompt=prompt,
            response=response,
        )

        proposals = self.proposer.propose_all(
            analysis["diagnoses"]
        )

        proposal_reports = []

        for proposal in proposals:
            proposal_dict = {
                "code": proposal.code,
                "priority": proposal.priority,
                "title": proposal.title,
                "reason": proposal.reason,
                "actions": list(proposal.actions),
                "requires_validation": (
                    proposal.requires_validation
                ),
                "automatic_apply_allowed": (
                    proposal.automatic_apply_allowed
                ),
            }

            plan = self.validator.build_plan(
                proposal_dict
            )

            validation = self.validator.validate_plan(
                plan
            )

            proposal_reports.append(
                {
                    "proposal": proposal_dict,
                    "validation": validation,
                }
            )

        report = {
            "timestamp": datetime.now(
                timezone.utc
            ).isoformat(),

            "coordinator_version": self.VERSION,

            "mode": "observation_only",

            "prompt": prompt,

            "response": response,

            "observation": observation,

            "analysis": analysis,

            "proposals": proposal_reports,

            "automatic_modification": False,
        }

        report_path = (
            self.history_dir
            / "validation_reports.jsonl"
        )

        with report_path.open(
            "a",
            encoding="utf-8",
        ) as handle:
            handle.write(
                json.dumps(
                    report,
                    ensure_ascii=False,
                )
                + "\n"
            )

        return report


if __name__ == "__main__":
    coordinator = EvolutionCoordinator()

    response = (
        "Neura. I Neura. Neura. "
        "respond. Neura. learn Neura."
    )

    report = coordinator.inspect_generation(
        prompt="Hello",
        response=response,
    )

    print("=" * 70)
    print("NEURA EVOLUTION COORDINATOR")
    print("=" * 70)

    print(
        "Version:",
        report["coordinator_version"],
    )

    print(
        "Mode:",
        report["mode"],
    )

    print()
    print("DIAGNOSES:")

    for diagnosis in report["analysis"]["diagnoses"]:
        print(
            f"  [{diagnosis['severity']}] "
            f"{diagnosis['code']}"
        )

    print()
    print("PROPOSALS:")

    for item in report["proposals"]:
        proposal = item["proposal"]
        validation = item["validation"]

        print(
            f"  [{proposal['priority']}] "
            f"{proposal['code']}"
        )

        print(
            "    Validation checks:",
            validation["check_count"],
        )

        print(
            "    Automatic apply:",
            validation[
                "automatic_apply_allowed"
            ],
        )

    print()
    print(
        "AUTOMATIC MODIFICATION:",
        report["automatic_modification"],
    )

    print()
    print("REPORT:")
    print(report_path if "report_path" in locals() else (
        coordinator.history_dir
        / "validation_reports.jsonl"
    ))

    print("=" * 70)
