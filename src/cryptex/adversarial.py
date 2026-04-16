"""Compatibility exports for adversarial module."""

from cryptex.analysis.adversarial import *  # noqa: F401,F403
from cryptex.analysis.adversarial import (
    _adversarial_key,
    _legal_text,
    _poetry_text,
    _repetitive_text,
    _short_text,
    _technical_text,
    _uniform_key,
    _very_short_text,
)

__all__ = [
    "StressTestCase",
    "StressTestResult",
    "generate_stress_tests",
    "run_stress_tests",
    "_adversarial_key",
    "_legal_text",
    "_poetry_text",
    "_repetitive_text",
    "_short_text",
    "_technical_text",
    "_uniform_key",
    "_very_short_text",
]
