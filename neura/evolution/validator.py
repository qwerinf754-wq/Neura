"""Safety validator for Neura Evolution proposals.

Phase 0.5:
Validation planning only.

No automatic mutation is performed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ValidationCheck:
    code: str
    description: str
    required: bool = True


@dataclass(frozen=True)
class ValidationPlan:
    proposal_code: str
    checks: tuple[ValidationCheck, ...]
    automatic_apply_allowed: bool = False


class EvolutionValidator:
    """Creates safe validation plans for evolution proposals."""

    VERSION = "0.5.0"

    COMMON_CHECKS = (
        ValidationCheck(
            code="PYTHON_COMPILE",
            description="Compile affected Python modules.",
        ),
        ValidationCheck(
            code="PROJECT_TESTS",
            description="Run the project's existing test suite.",
        ),
    )

    def build_plan(
        self,
        proposal: dict[str, Any],
    ) -> ValidationPlan:

        code = str(proposal.get("code", ""))

        checks = list(self.COMMON_CHECKS)

        if code == "CHECK_TOKENIZER_COMPATIBILITY":
            checks.extend(
                [
                    ValidationCheck(
                        code="TOKENIZER_VOCAB_CHECK",
                        description=(
                            "Verify active tokenizer vocabulary size "
                            "and token mappings."
                        ),
                    ),
                    ValidationCheck(
                        code="CHECKPOINT_VOCAB_CHECK",
                        description=(
                            "Verify checkpoint vocabulary size "
                            "and model configuration."
                        ),
                    ),
                    ValidationCheck(
                        code="TOKENIZER_CHECKPOINT_COMPATIBILITY",
                        description=(
                            "Verify tokenizer/checkpoint compatibility "
                            "before generation."
                        ),
                    ),
                ]
            )

        elif code == "INVESTIGATE_GENERATION_REPETITION":
            checks.extend(
                [
                    ValidationCheck(
                        code="GENERATION_SMOKE_TEST",
                        description=(
                            "Run deterministic generation on "
                            "a fixed diagnostic prompt."
                        ),
                    ),
                    ValidationCheck(
                        code="REPETITION_METRICS",
                        description=(
                            "Measure lexical diversity and repetition."
                        ),
                    ),
                    ValidationCheck(
                        code="UNKNOWN_TOKEN_METRICS",
                        description=(
                            "Measure unknown-token occurrence."
                        ),
                    ),
                ]
            )

        elif code == "MEASURE_REPETITION_PATTERN":
            checks.extend(
                [
                    ValidationCheck(
                        code="TOKEN_ID_TRACE",
                        description=(
                            "Capture generated token IDs "
                            "for diagnostic analysis."
                        ),
                    ),
                    ValidationCheck(
                        code="TOKEN_DECODE_TRACE",
                        description=(
                            "Decode generated token IDs individually."
                        ),
                    ),
                    ValidationCheck(
                        code="PROMPT_ALIGNMENT_CHECK",
                        description=(
                            "Verify prompt/tokenizer alignment."
                        ),
                    ),
                ]
            )

        return ValidationPlan(
            proposal_code=code,
            checks=tuple(checks),
            automatic_apply_allowed=False,
        )

    def validate_plan(
        self,
        plan: ValidationPlan,
    ) -> dict[str, Any]:

        return {
            "validator_version": self.VERSION,
            "proposal_code": plan.proposal_code,
            "check_count": len(plan.checks),
            "checks": [
                {
                    "code": check.code,
                    "description": check.description,
                    "required": check.required,
                }
                for check in plan.checks
            ],
            "automatic_apply_allowed": (
                plan.automatic_apply_allowed
            ),
        }
