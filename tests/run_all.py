from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

TESTS = [
    "tests/test_adamw.py",
    "tests/test_attention_grad.py",
    "tests/test_causal_lm_grad.py",
    "tests/test_lm_head_grad.py",
    "tests/test_reference_gradient.py",
    "tests/test_rope.py",
    "tests/test_swiglu_grad.py",
    "tests/test_training_resume.py",
    "tests/test_transformer_block_grad.py",
    "tests/test_transformer_grad.py",
    "tests/test_full_training_integration.py",
    "tests/test_security_bounds.py",
]


def main() -> int:
    print("=" * 72)
    print("NEURA TEST SUITE")
    print("=" * 72)

    passed = 0
    failed = 0

    for test in TESTS:
        print()
        print("=" * 72)
        print(f"RUN: {test}")
        print("=" * 72)

        result = subprocess.run(
            [sys.executable, test],
            cwd=ROOT,
        )

        if result.returncode == 0:
            passed += 1
            print()
            print(f"[PASS] {test}")
        else:
            failed += 1
            print()
            print(f"[FAIL] {test}")

    print()
    print("=" * 72)
    print("NEURA TEST SUITE RESULT")
    print("=" * 72)
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total : {passed + failed}")

    if failed:
        print("STATUS: FAIL")
        return 1

    print("STATUS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
