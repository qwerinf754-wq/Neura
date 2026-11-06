"""Read-only diagnostics for the Neura Evolution Engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .metrics import generation_metrics


@dataclass(frozen=True)
class Diagnosis:
    code: str
    severity: str
    message: str
    evidence: dict[str, Any]


class EvolutionAnalyzer:
    """Analyzes observed generation behavior without modifying the project."""

    VERSION = "0.2.0"

    def analyze_generation(
        self,
        prompt: str,
        response: str,
    ) -> list[Diagnosis]:
        metrics = generation_metrics(response)
        diagnoses: list[Diagnosis] = []

        words = metrics["words"]
        unique_ratio = metrics["unique_ratio"]
        repeated = metrics["repeated_word_count"]
        max_frequency = metrics["max_word_frequency"]

        if words >= 4 and unique_ratio <= 0.50:
            diagnoses.append(
                Diagnosis(
                    code="HIGH_REPETITION",
                    severity="high",
                    message="Generation contains unusually low lexical diversity.",
                    evidence={
                        "words": words,
                        "unique_ratio": unique_ratio,
                        "repeated_word_count": repeated,
                        "max_word_frequency": max_frequency,
                    },
                )
            )

        elif words >= 6 and unique_ratio <= 0.70:
            diagnoses.append(
                Diagnosis(
                    code="MODERATE_REPETITION",
                    severity="medium",
                    message="Generation shows reduced lexical diversity.",
                    evidence={
                        "words": words,
                        "unique_ratio": unique_ratio,
                        "repeated_word_count": repeated,
                        "max_word_frequency": max_frequency,
                    },
                )
            )

        if words > 0 and repeated / words >= 0.50:
            diagnoses.append(
                Diagnosis(
                    code="REPETITION_RATIO_HIGH",
                    severity="high",
                    message="Repeated tokens make up a large fraction of the response.",
                    evidence={
                        "words": words,
                        "repeated_word_count": repeated,
                        "repetition_ratio": round(
                            repeated / words,
                            6,
                        ),
                    },
                )
            )

        if "<UNK>" in response:
            unk_count = response.split().count("<UNK>")

            diagnoses.append(
                Diagnosis(
                    code="UNK_PRESENT",
                    severity="high",
                    message="Generation contains unknown tokens.",
                    evidence={
                        "unk_count": unk_count,
                        "response_words": words,
                    },
                )
            )

        return diagnoses

    def analyze(
        self,
        prompt: str,
        response: str,
    ) -> dict[str, Any]:
        metrics = generation_metrics(response)
        diagnoses = self.analyze_generation(prompt, response)

        return {
            "analyzer_version": self.VERSION,
            "metrics": metrics,
            "diagnoses": [
                {
                    "code": item.code,
                    "severity": item.severity,
                    "message": item.message,
                    "evidence": item.evidence,
                }
                for item in diagnoses
            ],
        }
