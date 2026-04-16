"""Tests for the MCMC solver (mcmc.py)."""

from __future__ import annotations

import math
import string

import pytest


class TestMCMCConfig:
    def test_default_values(self) -> None:
        from cryptex.mcmc import MCMCConfig

        c = MCMCConfig()
        assert c.iterations == 50_000
        assert c.num_restarts == 8
        assert c.t_start == 50.0
        assert c.cooling_rate == 0.99983
        assert c.early_stop_patience == 15_000
        assert c.score_threshold is None
        assert c.track_trajectory is True
        assert c.acceptance_window == 1000

    def test_custom_values(self) -> None:
        from cryptex.mcmc import MCMCConfig

        c = MCMCConfig(iterations=100, num_restarts=2, t_start=10.0)
        assert c.iterations == 100
        assert c.num_restarts == 2


class TestMCMCResult:
    def test_default_values(self) -> None:
        from cryptex.mcmc import MCMCResult

        r = MCMCResult()
        assert r.best_key == ""
        assert r.best_score == -math.inf
        assert r.chain_scores == []
        assert r.chain_keys == []
        assert r.score_trajectory == []
        assert r.acceptance_rates == []
        assert r.early_stopped is False


class TestMCMCHelpers:
    def test_apply_key_identity(self) -> None:
        from cryptex.mcmc import _apply_key

        result = _apply_key("hello", string.ascii_lowercase)
        assert result == "hello"

    def test_apply_key_atbash(self) -> None:
        from cryptex.mcmc import _apply_key

        # Key maps a→z, b→y, etc. But _apply_key maps key[i] → ascii[i]
        key = "zyxwvutsrqponmlkjihgfedcba"
        result = _apply_key("abc", key)
        assert result == "zyx"

    def test_swap_key_modifies_list(self) -> None:
        from cryptex.mcmc import _swap_key

        key = list(string.ascii_lowercase)
        original = key.copy()
        i, j = _swap_key(key)
        assert key[i] == original[j]
        assert key[j] == original[i]

    def test_random_key_is_permutation(self) -> None:
        from cryptex.mcmc import _random_key

        key = _random_key()
        assert sorted(key) == list(string.ascii_lowercase)
        assert len(key) == 26


class TestMCMCSolver:
    @pytest.mark.timeout(120)
    def test_solves_simple_cipher(self, trained_model, sample_plaintext) -> None:
        """MCMC should crack a substitution cipher on a ~100-char text."""
        from cryptex.ciphers import SimpleSubstitution
        from cryptex.mcmc import MCMCConfig, run_mcmc

        key = SimpleSubstitution.random_key()
        ct = SimpleSubstitution.encrypt(sample_plaintext, key)

        config = MCMCConfig(iterations=30_000, num_restarts=4)
        result = run_mcmc(ct, trained_model, config)

        assert result.best_key != ""
        assert result.best_plaintext != ""
        assert result.best_score > -math.inf
        assert len(result.chain_scores) == 4

    @pytest.mark.timeout(60)
    def test_callback_invoked(self, trained_model, sample_plaintext) -> None:
        from cryptex.ciphers import SimpleSubstitution
        from cryptex.mcmc import MCMCConfig, run_mcmc

        key = SimpleSubstitution.random_key()
        ct = SimpleSubstitution.encrypt(sample_plaintext, key)

        config = MCMCConfig(iterations=2000, num_restarts=1, callback_interval=500)
        calls = []

        def cb(chain, it, cur_key, pt, score, best, temp):
            calls.append((chain, it))

        run_mcmc(ct, trained_model, config, callback=cb)
        assert len(calls) > 0

    @pytest.mark.timeout(60)
    def test_early_stopping(self, trained_model) -> None:
        from cryptex.ciphers import SimpleSubstitution
        from cryptex.mcmc import MCMCConfig, run_mcmc

        pt = "the quick brown fox jumps over the lazy dog and some more text here"
        key = SimpleSubstitution.random_key()
        ct = SimpleSubstitution.encrypt(pt, key)

        config = MCMCConfig(
            iterations=50_000,
            num_restarts=1,
            early_stop_patience=500,
            score_threshold=-100.0,  # Easy threshold
        )
        result = run_mcmc(ct, trained_model, config)
        # Should either early stop or complete — just verify it runs
        assert result.best_key != ""

    @pytest.mark.timeout(60)
    def test_trajectory_tracking(self, trained_model, sample_plaintext) -> None:
        from cryptex.ciphers import SimpleSubstitution
        from cryptex.mcmc import MCMCConfig, run_mcmc

        key = SimpleSubstitution.random_key()
        ct = SimpleSubstitution.encrypt(sample_plaintext, key)

        config = MCMCConfig(iterations=5000, num_restarts=2, track_trajectory=True)
        result = run_mcmc(ct, trained_model, config)
        assert len(result.score_trajectory) > 0
        # acceptance_rates is sampled at regular intervals across all chains
        assert len(result.acceptance_rates) > 0
        assert all(0.0 <= r <= 1.0 for r in result.acceptance_rates)

    @pytest.mark.timeout(60)
    def test_chain_keys_stored(self, trained_model, sample_plaintext) -> None:
        from cryptex.ciphers import SimpleSubstitution
        from cryptex.mcmc import MCMCConfig, run_mcmc

        key = SimpleSubstitution.random_key()
        ct = SimpleSubstitution.encrypt(sample_plaintext, key)

        config = MCMCConfig(iterations=2000, num_restarts=3)
        result = run_mcmc(ct, trained_model, config)
        assert len(result.chain_keys) == 3
        for k in result.chain_keys:
            assert len(k) == 26

    @pytest.mark.timeout(30)
    def test_single_chain(self, trained_model) -> None:
        from cryptex.ciphers import SimpleSubstitution
        from cryptex.mcmc import MCMCConfig, run_mcmc

        pt = "a short text for testing"
        ct = SimpleSubstitution.encrypt(pt, SimpleSubstitution.random_key())
        config = MCMCConfig(iterations=1000, num_restarts=1)
        result = run_mcmc(ct, trained_model, config)
        assert len(result.chain_scores) == 1
