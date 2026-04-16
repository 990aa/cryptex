"""Tests for the Genetic Algorithm solver (genetic.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest


class TestGeneticConfig:
    def test_defaults(self) -> None:
        from cryptex.solvers.genetic import GeneticConfig

        c = GeneticConfig()
        assert c.population_size == 500
        assert c.generations == 1000
        assert c.elite_fraction == 0.10
        assert c.tournament_size == 5
        assert c.crossover_rate == 0.65
        assert c.mutation_rate == 0.08
        assert c.num_mutations == 1
        assert c.callback_interval == 50


class TestGeneticResult:
    def test_defaults(self) -> None:
        from cryptex.solvers.genetic import GeneticResult

        r = GeneticResult()
        assert r.best_key == ""
        assert r.best_score == -math.inf
        assert r.generations_run == 0


class TestGeneticHelpers:
    def test_random_key_is_permutation(self) -> None:
        from cryptex.solvers.genetic import _random_key

        key = _random_key()
        assert sorted(key) == list(range(26))

    def test_order_crossover_preserves_permutation(self) -> None:
        from cryptex.solvers.genetic import _order_crossover

        p1 = list(range(26))
        p2 = list(range(25, -1, -1))
        child = _order_crossover(p1, p2)
        assert sorted(child) == list(range(26))

    def test_order_crossover_identical_parents(self) -> None:
        from cryptex.solvers.genetic import _order_crossover

        p = list(range(26))
        child = _order_crossover(p, p)
        assert sorted(child) == list(range(26))

    def test_mutate_preserves_permutation(self) -> None:
        from cryptex.solvers.genetic import _mutate

        key = list(range(26))
        _mutate(key, n_swaps=3)
        assert sorted(key) == list(range(26))

    def test_decode_identity(self) -> None:
        from cryptex.solvers.genetic import _decode

        ct_idx = np.array([0, 1, 2, 3])
        key = list(range(26))  # identity
        result = _decode(ct_idx, key, include_space=False)
        np.testing.assert_array_equal(result, ct_idx)

    def test_idx_to_text(self) -> None:
        from cryptex.solvers.genetic import _idx_to_text

        idx = np.array([0, 1, 2])
        text = _idx_to_text(idx, include_space=False)
        assert text == "abc"

    def test_idx_to_text_with_space(self) -> None:
        from cryptex.solvers.genetic import _idx_to_text

        # include_space=True: 0=space, 1='a', 2='b', ..., 26='z'
        idx = np.array([1, 0, 2])
        text = _idx_to_text(idx, include_space=True)
        assert text == "a b"


class TestGeneticSolver:
    @pytest.mark.timeout(120)
    def test_runs_and_returns(self, trained_model) -> None:
        from cryptex.ciphers import SimpleSubstitution
        from cryptex.solvers.genetic import GeneticConfig, run_genetic

        pt = "the quick brown fox jumps over the lazy dog and some extra text to help"
        key = SimpleSubstitution.random_key()
        ct = SimpleSubstitution.encrypt(pt, key)

        config = GeneticConfig(population_size=50, generations=50)
        result = run_genetic(ct, trained_model, config)
        assert result.best_key != ""
        assert result.best_plaintext != ""
        assert result.generations_run == 50
        assert len(result.score_trajectory) > 0

    @pytest.mark.timeout(60)
    def test_callback_invoked(self, trained_model) -> None:
        from cryptex.ciphers import SimpleSubstitution
        from cryptex.solvers.genetic import GeneticConfig, run_genetic

        pt = "hello world test"
        ct = SimpleSubstitution.encrypt(pt, SimpleSubstitution.random_key())

        config = GeneticConfig(
            population_size=20, generations=100, callback_interval=25
        )
        calls = []

        def cb(gen, key, plaintext, score):
            calls.append(gen)

        run_genetic(ct, trained_model, config, callback=cb)
        assert len(calls) >= 3  # At least gen 25, 50, 75, 100

    @pytest.mark.timeout(60)
    def test_score_improves_over_generations(self, trained_model) -> None:
        from cryptex.ciphers import SimpleSubstitution
        from cryptex.solvers.genetic import GeneticConfig, run_genetic

        pt = "the cat sat on the mat by the hat near the fat rat"
        ct = SimpleSubstitution.encrypt(pt, SimpleSubstitution.random_key())

        config = GeneticConfig(population_size=50, generations=100)
        result = run_genetic(ct, trained_model, config)

        # Score should improve from first to last in trajectory
        if len(result.score_trajectory) > 1:
            assert result.score_trajectory[-1][1] >= result.score_trajectory[0][1]

    @pytest.mark.timeout(30)
    def test_small_population(self, trained_model) -> None:
        from cryptex.ciphers import SimpleSubstitution
        from cryptex.solvers.genetic import GeneticConfig, run_genetic

        pt = "hello world"
        ct = SimpleSubstitution.encrypt(pt, SimpleSubstitution.random_key())
        config = GeneticConfig(population_size=5, generations=10)
        result = run_genetic(ct, trained_model, config)
        assert result.best_key != ""
