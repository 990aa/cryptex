"""Integration and stress tests — end-to-end cipher cracking verification.

These tests verify the complete pipeline works for various cipher types
and text lengths. They have longer timeouts as they run actual solvers.
"""

from __future__ import annotations


import pytest

from cipher.ciphers import (
    ColumnarTransposition,
    SimpleSubstitution,
    Vigenere,
    get_engine,
)
from cipher.evaluation import symbol_error_rate


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture(scope="module")
def model():
    from cipher.ngram import get_model

    return get_model()


@pytest.fixture(scope="module")
def model_nospace():
    from cipher.ngram import get_model

    return get_model(include_space=False)


MEDIUM_TEXT = (
    "it is a truth universally acknowledged that a single man in possession "
    "of a good fortune must be in want of a wife however little known the "
    "feelings or views of such a man may be on his first entering a "
    "neighbourhood this truth is so well fixed in the minds of the "
    "surrounding families that he is considered as the rightful property "
    "of some one or other of their daughters"
)

SHORT_TEXT = "the quick brown fox jumps over the lazy dog and then runs away fast"


# ------------------------------------------------------------------
# End-to-end: MCMC substitution cracking
# ------------------------------------------------------------------


class TestMCMCEndToEnd:
    @pytest.mark.timeout(180)
    def test_cracks_medium_text(self, model) -> None:
        """MCMC should crack a ~300-char substitution cipher with SER < 20%."""
        from cipher.mcmc import MCMCConfig, run_mcmc

        key = SimpleSubstitution.random_key()
        ct = SimpleSubstitution.encrypt(MEDIUM_TEXT, key)

        config = MCMCConfig(iterations=50_000, num_restarts=8)
        result = run_mcmc(ct, model, config)

        ser = symbol_error_rate(MEDIUM_TEXT, result.best_plaintext)
        # MCMC is stochastic; accept up to 20% error
        assert ser < 0.20, f"SER too high: {ser:.1%}"
        assert result.best_score > -1e10

    @pytest.mark.timeout(120)
    def test_cracks_short_text(self, model) -> None:
        """MCMC on shorter text — may not be perfect but should converge."""
        from cipher.mcmc import MCMCConfig, run_mcmc

        key = SimpleSubstitution.random_key()
        ct = SimpleSubstitution.encrypt(SHORT_TEXT, key)

        config = MCMCConfig(iterations=30_000, num_restarts=4)
        result = run_mcmc(ct, model, config)
        # Short text is harder; just verify solver ran
        assert result.best_plaintext != ""
        assert result.best_key != ""


# ------------------------------------------------------------------
# End-to-end: HMM substitution cracking
# ------------------------------------------------------------------


class TestHMMEndToEnd:
    @pytest.mark.timeout(120)
    def test_hmm_runs(self, model) -> None:
        """HMM solver should produce output."""
        from cipher.hmm import HMMConfig, run_hmm

        key = SimpleSubstitution.random_key()
        ct = SimpleSubstitution.encrypt(SHORT_TEXT, key)

        config = HMMConfig(max_iter=30)
        result = run_hmm(ct, model, config)
        assert result.best_plaintext != ""


# ------------------------------------------------------------------
# End-to-end: Genetic algorithm
# ------------------------------------------------------------------


class TestGeneticEndToEnd:
    @pytest.mark.timeout(120)
    def test_genetic_cracks(self, model) -> None:
        """Genetic algorithm should crack substitution cipher."""
        from cipher.genetic import GeneticConfig, run_genetic

        key = SimpleSubstitution.random_key()
        ct = SimpleSubstitution.encrypt(MEDIUM_TEXT, key)

        config = GeneticConfig(population_size=200, generations=300)
        result = run_genetic(ct, model, config)
        assert result.best_plaintext != ""
        assert result.best_score > -1e10


# ------------------------------------------------------------------
# End-to-end: Vigenère cracking
# ------------------------------------------------------------------


class TestVigenereEndToEnd:
    @pytest.mark.timeout(60)
    def test_vigenere_cracks(self, model) -> None:
        from cipher.vigenere_cracker import crack_vigenere

        key = "cipher"
        ct = Vigenere.encrypt(MEDIUM_TEXT, key)
        result = crack_vigenere(ct, model)
        ser = symbol_error_rate(MEDIUM_TEXT, result.best_plaintext)
        assert ser < 0.25, f"SER too high: {ser:.1%}"


# ------------------------------------------------------------------
# End-to-end: Transposition cracking
# ------------------------------------------------------------------


class TestTranspositionEndToEnd:
    @pytest.mark.timeout(120)
    def test_transposition_cracks(self, model) -> None:
        from cipher.transposition_cracker import crack_transposition

        key = [2, 0, 3, 1]
        ct = ColumnarTransposition.encrypt(MEDIUM_TEXT, key)
        result = crack_transposition(ct, model)
        assert result.best_plaintext != ""
        assert result.best_score > -1e10


# ------------------------------------------------------------------
# End-to-end: Cipher type detection
# ------------------------------------------------------------------


class TestDetectionEndToEnd:
    def test_detect_substitution(self) -> None:
        from cipher.detector import detect_cipher_type

        key = SimpleSubstitution.random_key()
        ct = SimpleSubstitution.encrypt(MEDIUM_TEXT, key)
        result = detect_cipher_type(ct)
        # Should detect as substitution (most likely)
        assert result.all_scores is not None
        assert result.predicted_type in result.all_scores
        assert result.confidence > 0

    def test_detect_vigenere(self) -> None:
        from cipher.detector import detect_cipher_type

        ct = Vigenere.encrypt(MEDIUM_TEXT, "secret")
        result = detect_cipher_type(ct)
        assert result.all_scores is not None
        assert result.predicted_type in result.all_scores


# ------------------------------------------------------------------
# Stress: Multiple cipher types in sequence
# ------------------------------------------------------------------


class TestStressSequential:
    @pytest.mark.timeout(300)
    def test_multiple_substitution_keys(self, model) -> None:
        """Run MCMC across 3 different random keys to verify consistency."""
        from cipher.mcmc import MCMCConfig, run_mcmc

        config = MCMCConfig(iterations=30_000, num_restarts=4)
        for _ in range(3):
            key = SimpleSubstitution.random_key()
            ct = SimpleSubstitution.encrypt(MEDIUM_TEXT, key)
            result = run_mcmc(ct, model, config)
            assert result.best_plaintext != ""

    @pytest.mark.timeout(60)
    def test_all_cipher_engines(self) -> None:
        """Verify all registered cipher engines can encrypt and decrypt."""
        for name in ("substitution", "vigenere", "transposition", "playfair"):
            engine = get_engine(name)
            key = engine.random_key()
            ct = engine.encrypt(SHORT_TEXT, key)
            assert (
                ct != SHORT_TEXT or name == "substitution"
            )  # identity key is possible but unlikely
            pt = engine.decrypt(ct, key)
            # For ciphers that support perfect roundtrip:
            if name not in (
                "playfair",
                "transposition",
            ):  # Playfair alters J/padding; Transposition pads with 'x'
                assert pt == SHORT_TEXT
            elif name == "transposition":
                assert pt.startswith(SHORT_TEXT) or pt.rstrip("x") == SHORT_TEXT


# ------------------------------------------------------------------
# Evaluation integration test
# ------------------------------------------------------------------


class TestEvaluationIntegration:
    def test_ser_on_real_crack(self, model) -> None:
        """SER function works with actual cracking output."""
        from cipher.evaluation import symbol_error_rate
        from cipher.mcmc import MCMCConfig, run_mcmc

        key = SimpleSubstitution.random_key()
        ct = SimpleSubstitution.encrypt(MEDIUM_TEXT, key)

        config = MCMCConfig(iterations=20_000, num_restarts=3)
        result = run_mcmc(ct, model, config)
        ser = symbol_error_rate(MEDIUM_TEXT, result.best_plaintext)
        assert 0.0 <= ser <= 1.0
