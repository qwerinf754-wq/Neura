"""Read-only repair proposal engine for Neura Evolution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Proposal:
    code: str
    priority: str
    title: str
    reason: str
    actions: tuple[str, ...]
    requires_validation: bool = True
    automatic_apply_allowed: bool = False


class EvolutionProposer:
    """Converts diagnoses into safe, non-applied repair proposals."""

    VERSION = "0.3.0"

    def propose(
        self,
        diagnosis: dict[str, Any],
    ) -> Proposal | None:

        code = diagnosis.get("code")
        evidence = diagnosis.get("evidence", {})

        if code == "UNK_PRESENT":
            return Proposal(
                code="CHECK_TOKENIZER_COMPATIBILITY",
                priority="critical",
                title="Verify tokenizer/checkpoint compatibility",
                reason=(
                    "The generation contains unknown tokens. "
                    "The tokenizer vocabulary and checkpoint vocabulary "
                    "should be inspected before changing decoding parameters."
                ),
                actions=(
                    "Inspect active tokenizer vocabulary size.",
                    "Inspect checkpoint vocabulary size.",
                    "Compare tokenizer special-token mappings.",
                    "Verify checkpoint and tokenizer artifact compatibility.",
                    "Run a direct encode/decode compatibility test.",
                ),
            )

        if code == "HIGH_REPETITION":
            return Proposal(
                code="INVESTIGATE_GENERATION_REPETITION",
                priority="high",
                title="Investigate repetitive generation",
                reason=(
                    "The response has unusually low lexical diversity. "
                    "The underlying model/checkpoint/tokenizer state should "
                    "be investigated before modifying decoding parameters."
                ),
                actions=(
                    "Inspect active checkpoint metadata.",
                    "Inspect active tokenizer vocabulary.",
                    "Measure unknown-token frequency.",
                    "Run deterministic generation diagnostics.",
                    "Compare generation against a known-good checkpoint.",
                ),
            )

        if code == "REPETITION_RATIO_HIGH":
            return Proposal(
                code="MEASURE_REPETITION_PATTERN",
                priority="high",
                title="Measure repeated-token pattern",
                reason=(
                    "A large fraction of generated tokens are repetitions. "
                    "Determine whether repetition originates from model state, "
                    "tokenizer mismatch, prompt formatting, or decoding."
                ),
                actions=(
                    "Capture the generated token IDs.",
                    "Inspect repeated token IDs.",
                    "Decode token IDs individually.",
                    "Check prompt/tokenizer alignment.",
                    "Compare multiple decoding configurations.",
                ),
            )

        if code == "MODERATE_REPETITION":
            return Proposal(
                code="MONITOR_REPETITION",
                priority="medium",
                title="Monitor lexical diversity",
                reason=(
                    "The response shows reduced lexical diversity, "
                    "but the signal is not sufficient to justify an automatic change."
                ),
                actions=(
                    "Record the metric.",
                    "Compare against previous generations.",
                    "Monitor the same prompt across multiple runs.",
                ),
            )

        return None

    def propose_all(
        self,
        diagnoses: list[dict[str, Any]],
    ) -> list[Proposal]:

        proposals: list[Proposal] = []
        seen: set[str] = set()

        for diagnosis in diagnoses:
            proposal = self.propose(diagnosis)

            if proposal is None:
                continue

            if proposal.code in seen:
                continue

            seen.add(proposal.code)
            proposals.append(proposal)

        return proposals
