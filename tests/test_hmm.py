"""Tests for the HMM solver (hmm.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest


class TestHMMConfig:
    def test_defaults(self) -> None:
        from cryptex.hmm import HMMConfig

        c = HMMConfig()
        assert c.max_iter == 100
        assert c.tol == 1e-4
        assert c.callback_interval == 1


class TestHMMResult:
    def test_defaults(self) -> None:
        from cryptex.hmm import HMMResult

        r = HMMResult()
        assert r.best_key == ""
        assert r.best_score == -math.inf
        assert r.log_likelihoods == []


class TestHMMHelpers:
    def test_logsumexp_matches_numpy(self) -> None:
        from cryptex.hmm import _logsumexp

        arr = np.array([-1.0, -2.0, -3.0])
        result = _logsumexp(arr)
        expected = np.log(np.exp(-1.0) + np.exp(-2.0) + np.exp(-3.0))
        assert abs(float(result) - expected) < 1e-10

    def test_logsumexp_axis(self) -> None:
        from cryptex.hmm import _logsumexp

        arr = np.array([[-1.0, -2.0], [-3.0, -4.0]])
        result = _logsumexp(arr, axis=1)
        assert result.shape == (2,)

    def test_normalise_log(self) -> None:
        from cryptex.hmm import _normalise_log

        log_probs = np.array([-1.0, -2.0, -3.0])
        result = _normalise_log(log_probs)
        probs = np.exp(result)
        assert abs(probs.sum() - 1.0) < 1e-6


class TestHMMSolver:
    @pytest.mark.timeout(60)
    def test_runs_and_returns(self, trained_model) -> None:
        from cryptex.ciphers import SimpleSubstitution
        from cryptex.hmm import HMMConfig, run_hmm

        pt = "the quick brown fox jumps"
        key = SimpleSubstitution.random_key()
        ct = SimpleSubstitution.encrypt(pt, key)

        config = HMMConfig(max_iter=10)
        result = run_hmm(ct, trained_model, config)
        assert result.best_key != ""
        assert result.best_plaintext != ""
        assert result.iterations > 0
        assert len(result.log_likelihoods) > 0

    @pytest.mark.timeout(60)
    def test_callback_invoked(self, trained_model) -> None:
        from cryptex.ciphers import SimpleSubstitution
        from cryptex.hmm import HMMConfig, run_hmm

        pt = "hello world test"
        ct = SimpleSubstitution.encrypt(pt, SimpleSubstitution.random_key())

        config = HMMConfig(max_iter=5)
        calls = []

        def cb(iteration, plaintext, ll):
            calls.append(iteration)

        run_hmm(ct, trained_model, config, callback=cb)
        assert len(calls) > 0

    @pytest.mark.timeout(60)
    def test_log_likelihood_non_decreasing(self, trained_model) -> None:
        """EM should generally not decrease the log-likelihood."""
        from cryptex.ciphers import SimpleSubstitution
        from cryptex.hmm import HMMConfig, run_hmm

        pt = "the cat sat on the mat by the hat"
        ct = SimpleSubstitution.encrypt(pt, SimpleSubstitution.random_key())
        config = HMMConfig(max_iter=20)
        result = run_hmm(ct, trained_model, config)

        lls = result.log_likelihoods
        # Allow small numerical errors
        for i in range(1, len(lls)):
            assert lls[i] >= lls[i - 1] - 1e-3, (
                f"LL decreased at step {i}: {lls[i - 1]:.4f} → {lls[i]:.4f}"
            )

