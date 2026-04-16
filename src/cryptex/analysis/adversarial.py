"""Adversarial stress testing for the cipher cracking system.

Generates deliberately difficult ciphertexts designed to test failure modes:
- Unusual character distributions (legal, technical, poetry)
- Short texts near the phase transition boundary
- Keys creating unusual frequency profiles
- Non-standard text sources
"""

from __future__ import annotations

import string
import time
from dataclasses import dataclass

from cryptex.analysis.evaluation import symbol_error_rate
from cryptex.ciphers import SimpleSubstitution
from cryptex.core.ngram import NgramModel


@dataclass
class StressTestCase:
    """A single adversarial test case."""

    name: str = ""
    description: str = ""
    plaintext: str = ""
    ciphertext: str = ""
    key: str = ""
    difficulty: str = "normal"  # easy, normal, hard, extreme


@dataclass
class StressTestResult:
    """Result of running a stress test case."""

    case_name: str = ""
    ser: float = 0.0
    score: float = 0.0
    time_seconds: float = 0.0
    decrypted: str = ""
    success: bool = False


# Adversarial Text Generators


def _legal_text() -> str:
    """Simulate dense legal/contract language with unusual word distributions."""
    return (
        "whereas the party of the first part hereinafter referred to as the licensor "
        "hereby grants to the party of the second part hereinafter referred to as the "
        "licensee a nonexclusive nontransferable limited license to use copy modify and "
        "distribute the software subject to the terms and conditions set forth herein "
        "the licensee shall not sublicense assign or transfer this license without the "
        "prior written consent of the licensor any attempt to do so shall be void and "
        "of no effect whatsoever the licensor makes no warranties express or implied "
        "including but not limited to the implied warranties of merchantability and "
        "fitness for a particular purpose"
    )


def _technical_text() -> str:
    """Simulate technical/scientific writing with unusual vocabulary."""
    return (
        "the electromagnetic field propagates through the vacuum at a velocity of "
        "approximately three hundred million meters per second the wavelength of the "
        "radiation is inversely proportional to its frequency according to the fundamental "
        "relationship between these quantities the quantum mechanical description of the "
        "hydrogen atom yields discrete energy levels characterized by principal quantum "
        "numbers the schrodinger equation provides the mathematical framework for "
        "calculating the probability amplitudes associated with various electronic "
        "configurations in multielectron systems the perturbation theory approach"
    )


def _poetry_text() -> str:
    """Simulate poetry with rhythmic patterns and unusual word choices."""
    return (
        "shall i compare thee to a summers day thou art more lovely and more temperate "
        "rough winds do shake the darling buds of may and summers lease hath all too short "
        "a date sometime too hot the eye of heaven shines and often is his gold complexion "
        "dimmed and every fair from fair sometime declines by chance or natures changing "
        "course untrimmed but thy eternal summer shall not fade nor lose possession of "
        "that fair thou owest nor shall death brag thou wanderest in his shade"
    )


def _repetitive_text() -> str:
    """Text with high letter repetition — tests algorithm on skewed distributions."""
    return (
        "to be or not to be that is the question whether tis nobler in the mind to suffer "
        "the slings and arrows of outrageous fortune or to take arms against a sea of "
        "troubles and by opposing end them to die to sleep no more and by a sleep to say "
        "we end the heartache and the thousand natural shocks that flesh is heir to tis "
        "a consummation devoutly to be wished to die to sleep to sleep perchance to dream"
    )


def _short_text() -> str:
    """Very short text near phase transition boundary."""
    return "the quick brown fox jumps over the lazy dog near the old oak tree"


def _very_short_text() -> str:
    """Extremely short — should be nearly impossible to crack."""
    return "hello world"


def _uniform_key() -> str:
    """Create a key that maps common letters to common letters (minimal disruption)."""
    return "etaoinshrdlcumwfgypbvkjxqz"


def _adversarial_key() -> str:
    """Key designed to create confusing frequency profile.

    Maps the most common English letters (e, t, a) to rare letters (z, q, x).
    """
    # English frequency order: e t a o i n s h r d l c u m w f g y p b v k j x q z
    common = "etaoinshr"
    rare = "zqxjkvbpy"
    key_list = list(string.ascii_lowercase)
    for c, r in zip(common, rare):
        ci, ri = ord(c) - ord("a"), ord(r) - ord("a")
        key_list[ci], key_list[ri] = key_list[ri], key_list[ci]
    return "".join(key_list)


# Test Case Generation


def generate_stress_tests() -> list[StressTestCase]:
    """Generate a suite of adversarial test cases."""
    cases: list[StressTestCase] = []

    # --- Unusual text distributions ---
    for name, generator, difficulty in [
        ("Legal Text", _legal_text, "hard"),
        ("Technical/Scientific", _technical_text, "hard"),
        ("Poetry (Shakespeare)", _poetry_text, "normal"),
        ("Repetitive Text", _repetitive_text, "normal"),
        ("Short Text (65 chars)", _short_text, "hard"),
        ("Very Short (11 chars)", _very_short_text, "extreme"),
    ]:
        plaintext = generator()
        # Standard random key
        key = SimpleSubstitution.random_key()
        ciphertext = SimpleSubstitution.encrypt(plaintext, key)
        cases.append(
            StressTestCase(
                name=name,
                description=f"Standard random key on {name.lower()} ({len(plaintext)} chars)",
                plaintext=plaintext,
                ciphertext=ciphertext,
                key=key,
                difficulty=difficulty,
            )
        )

    # --- Adversarial keys ---
    normal_text = (
        "it is a truth universally acknowledged that a single man in possession of a good "
        "fortune must be in want of a wife however little known the feelings or views of "
        "such a man may be on his first entering a neighbourhood"
    )

    adv_key = _adversarial_key()
    cases.append(
        StressTestCase(
            name="Adversarial Key",
            description="Key maps common→rare letters to confuse frequency analysis",
            plaintext=normal_text,
            ciphertext=SimpleSubstitution.encrypt(normal_text, adv_key),
            key=adv_key,
            difficulty="hard",
        )
    )

    # --- Uniform key (minimal shuffling) ---
    uni_key = _uniform_key()
    cases.append(
        StressTestCase(
            name="Near-Identity Key",
            description="Key barely differs from identity — subtle cipher",
            plaintext=normal_text,
            ciphertext=SimpleSubstitution.encrypt(normal_text, uni_key),
            key=uni_key,
            difficulty="normal",
        )
    )

    return cases


# Runner


def run_stress_tests(
    model: NgramModel,
    cases: list[StressTestCase] | None = None,
    callback=None,
) -> list[StressTestResult]:
    """Run all stress test cases and return results.

    Parameters
    ----------
    model : NgramModel
    cases : list of StressTestCase, optional
        Defaults to generate_stress_tests().
    callback : callable, optional
        callback(case_name, result)
    """
    from cryptex.solvers.mcmc import MCMCConfig, run_mcmc

    if cases is None:
        cases = generate_stress_tests()

    config = MCMCConfig(
        iterations=40_000,
        num_restarts=6,
        track_trajectory=False,
    )

    results: list[StressTestResult] = []

    for case in cases:
        t0 = time.time()
        mcmc_result = run_mcmc(case.ciphertext, model, config)
        elapsed = time.time() - t0

        ser = symbol_error_rate(case.plaintext, mcmc_result.best_plaintext)
        success = ser < 0.05

        result = StressTestResult(
            case_name=case.name,
            ser=ser,
            score=mcmc_result.best_score,
            time_seconds=elapsed,
            decrypted=mcmc_result.best_plaintext,
            success=success,
        )
        results.append(result)

        if callback:
            callback(case.name, result)

    return results

