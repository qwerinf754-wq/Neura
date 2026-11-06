"""Read-only validation executor for Neura Evolution.

Phase 0.7:
Execute diagnostic validation checks without mutation.

Safety:
- No source modification.
- No checkpoint modification.
- No tokenizer modification.
- No file deletion.
- No dependency installation.
- No shell execution.
- No automatic apply.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import re
import subprocess
import sys

from .metrics import generation_metrics


@dataclass(frozen=True)
class CheckResult:
    code: str
    status: str
    message: str
    details: dict[str, Any]


class EvolutionValidationExecutor:
    """Executes safe, read-only validation checks."""

    VERSION = "0.7.0"

    def __init__(self, project_root: Path | None = None):
        self.project_root = (
            project_root
            if project_root is not None
            else Path.cwd()
        )

    def _result(
        self,
        code: str,
        status: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> CheckResult:
        return CheckResult(
            code=code,
            status=status,
            message=message,
            details=details or {},
        )

    def python_compile(self) -> CheckResult:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "compileall",
                "-q",
                "neura/evolution",
            ],
            cwd=self.project_root,
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            return self._result(
                "PYTHON_COMPILE",
                "PASS",
                "Evolution modules compiled successfully.",
            )

        return self._result(
            "PYTHON_COMPILE",
            "FAIL",
            "Python compilation failed.",
            {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            },
        )

    def project_tests(self) -> CheckResult:
        """Run the existing lightweight project test runner if present."""

        runner = self.project_root / "tests" / "run_all.py"

        if not runner.exists():
            return self._result(
                "PROJECT_TESTS",
                "SKIP",
                "Existing project test runner was not found.",
            )

        result = subprocess.run(
            [
                sys.executable,
                str(runner),
            ],
            cwd=self.project_root,
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            return self._result(
                "PROJECT_TESTS",
                "PASS",
                "Project test runner completed successfully.",
            )

        return self._result(
            "PROJECT_TESTS",
            "FAIL",
            "Project test runner reported failure.",
            {
                "returncode": result.returncode,
                "stdout_tail": result.stdout[-4000:],
                "stderr_tail": result.stderr[-4000:],
            },
        )

    def repetition_metrics(
        self,
        response: str,
    ) -> CheckResult:

        metrics = generation_metrics(response)

        repeated = metrics.get(
            "repeated_word_count",
            0,
        )

        words = metrics.get(
            "words",
            0,
        )

        ratio = (
            repeated / words
            if words
            else 0.0
        )

        status = (
            "FAIL"
            if ratio >= 0.50
            else "PASS"
        )

        return self._result(
            "REPETITION_METRICS",
            status,
            "Generation repetition metrics evaluated.",
            {
                **metrics,
                "repetition_ratio": ratio,
            },
        )

    def unknown_token_metrics(
        self,
        response: str,
    ) -> CheckResult:

        unknown_count = response.count("<UNK>")

        tokens = re.findall(
            r"\S+",
            response,
        )

        token_count = len(tokens)

        ratio = (
            unknown_count / token_count
            if token_count
            else 0.0
        )

        status = (
            "FAIL"
            if unknown_count > 0
            else "PASS"
        )

        return self._result(
            "UNKNOWN_TOKEN_METRICS",
            status,
            "Unknown-token occurrence evaluated.",
            {
                "unknown_count": unknown_count,
                "token_count": token_count,
                "unknown_ratio": ratio,
            },
        )

    def generation_smoke_test(
        self,
        response: str,
    ) -> CheckResult:

        if not response.strip():
            return self._result(
                "GENERATION_SMOKE_TEST",
                "FAIL",
                "Generation returned an empty response.",
            )

        return self._result(
            "GENERATION_SMOKE_TEST",
            "PASS",
            "Diagnostic generation returned non-empty output.",
            {
                "characters": len(response),
            },
        )

    def tokenizer_checkpoint_scan(self) -> CheckResult:
        """Scan tokenizer/checkpoint artifacts without loading or modifying them."""

        tokenizer_files = sorted(
            self.project_root.glob(
                "data/processed/**/tokenizer.json"
            )
        )

        checkpoint_files = sorted(
            list(
                self.project_root.glob(
                    "data/checkpoints/**/*.pt"
                )
            )
            + list(
                self.project_root.glob(
                    "checkpoints/**/*.pt"
                )
            )
        )

        return self._result(
            "TOKENIZER_CHECKPOINT_SCAN",
            "PASS",
            "Tokenizer and checkpoint artifacts discovered.",
            {
                "tokenizer_files": [
                    str(path.relative_to(self.project_root))
                    for path in tokenizer_files
                ],
                "checkpoint_files": [
                    str(path.relative_to(self.project_root))
                    for path in checkpoint_files
                ],
                "tokenizer_count": len(tokenizer_files),
                "checkpoint_count": len(checkpoint_files),
            },
        )

    def execute_generation_checks(
        self,
        response: str,
    ) -> list[CheckResult]:

        return [
            self.generation_smoke_test(response),
            self.repetition_metrics(response),
            self.unknown_token_metrics(response),
            self.tokenizer_checkpoint_scan(),
        ]

    def execute(
        self,
        response: str,
    ) -> dict[str, Any]:

        results = self.execute_generation_checks(
            response
        )

        return {
            "executor_version": self.VERSION,
            "mode": "read_only",
            "automatic_apply_allowed": False,
            "results": [
                {
                    "code": result.code,
                    "status": result.status,
                    "message": result.message,
                    "details": result.details,
                }
                for result in results
            ],
        }


if __name__ == "__main__":
    executor = EvolutionValidationExecutor()

    response = (
        "Neura. I Neura. Neura. "
        "respond. Neura. learn Neura."
    )

    result = executor.execute(response)

    print("=" * 70)
    print("NEURA EVOLUTION VALIDATION EXECUTOR")
    print("=" * 70)

    print(
        "Version:",
        result["executor_version"],
    )

    print(
        "Mode:",
        result["mode"],
    )

    print()

    for item in result["results"]:
        print(
            f"[{item['status']}] "
            f"{item['code']}"
        )

        print(
            f"  {item['message']}"
        )

        if item["details"]:
            print(
                "  Details:",
                json.dumps(
                    item["details"],
                    ensure_ascii=False,
                ),
            )

    print()
    print(
        "AUTOMATIC APPLY:",
        result["automatic_apply_allowed"],
    )

    print("=" * 70)
