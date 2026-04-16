"""Tests for known-plaintext constrained cracking mode."""

from __future__ import annotations

import pytest


class TestKnownPlaintextCracking:
    @pytest.mark.timeout(120)
    def test_kpa_with_one_known_word(self, trained_model, sample_plaintext) -> None:
        from cryptex.analysis.evaluation import symbol_error_rate
        from cryptex.ciphers import SimpleSubstitution
        from cryptex.solvers.kpa import crack_with_known_plaintext
        from cryptex.solvers.mcmc import MCMCConfig, run_mcmc

        key = SimpleSubstitution.random_key()
        ct = SimpleSubstitution.encrypt(sample_plaintext, key)

        pt_alpha = "".join(ch for ch in sample_plaintext.lower() if ch.isalpha())
        ct_alpha = "".join(ch for ch in ct.lower() if ch.isalpha())

        baseline_cfg = MCMCConfig(random_seed=7, num_restarts=2, iterations=8_000)
        baseline = run_mcmc(ct_alpha, trained_model, baseline_cfg)
        baseline_ser = symbol_error_rate(pt_alpha, baseline.best_plaintext)

        anchor = "truth"
        start = pt_alpha.index(anchor)
        known = [(anchor, ct_alpha[start : start + len(anchor)])]
        kpa_cfg = MCMCConfig(random_seed=7, num_restarts=2, iterations=10_000)
        constrained = crack_with_known_plaintext(
            ct_alpha, known, trained_model, config=kpa_cfg
        )
        constrained_ser = symbol_error_rate(pt_alpha, constrained.best_plaintext)

        assert constrained_ser <= baseline_ser
        assert constrained_ser < 0.10
