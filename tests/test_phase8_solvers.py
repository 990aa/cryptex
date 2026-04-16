"""Phase 8 solver coverage: affine and rail fence crackers."""

from __future__ import annotations

import math

import pytest


class TestAffineCracker:
    @pytest.mark.timeout(60)
    def test_crack_affine_exhaustive_search(self, trained_model) -> None:
        from cryptex.analysis.evaluation import symbol_error_rate
        from cryptex.ciphers import Affine
        from cryptex.solvers.affine import crack_affine

        plaintext = (
            "the quick brown fox jumps over the lazy dog and then runs through the field "
            "with a bundle of letters tied with string"
        )
        key = (5, 8)
        ciphertext = Affine.encrypt(plaintext, key)

        result = crack_affine(ciphertext, trained_model)
        assert result.best_plaintext != ""
        assert result.best_a in {1, 3, 5, 7, 9, 11, 15, 17, 19, 21, 23, 25}
        assert 0 <= result.best_b < 26
        assert result.tested_keys == 312

        ser = symbol_error_rate(plaintext, result.best_plaintext)
        assert ser < 0.20


class TestRailFenceCracker:
    @pytest.mark.timeout(60)
    def test_crack_rail_fence_search(self, trained_model) -> None:
        from cryptex.analysis.evaluation import symbol_error_rate
        from cryptex.ciphers import RailFence
        from cryptex.solvers.railfence import RailFenceConfig, crack_railfence

        plaintext = (
            "it is a truth universally acknowledged that a single man in possession "
            "of a good fortune must be in want of a wife"
        )
        rails = 4
        ciphertext = RailFence.encrypt(plaintext, rails)

        config = RailFenceConfig(min_rails=2, max_rails=10)
        result = crack_railfence(ciphertext, trained_model, config=config)
        assert result.best_plaintext != ""
        assert 2 <= result.best_rails <= 10
        assert math.isfinite(result.best_score)

        ser = symbol_error_rate(plaintext, result.best_plaintext)
        assert ser < 0.20

