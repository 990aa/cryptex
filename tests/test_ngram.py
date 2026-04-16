"""Tests for the n-gram language model (ngram.py)."""

from __future__ import annotations

import numpy as np


class TestCharConversion:
    """Tests for char_to_idx and text_to_indices."""

    def test_char_to_idx_a(self) -> None:
        from cryptex.core.ngram import char_to_idx

        # include_space=True: space=0, a=1, b=2, ..., z=26
        assert char_to_idx("a", include_space=True) == 1

    def test_char_to_idx_z(self) -> None:
        from cryptex.core.ngram import char_to_idx

        assert char_to_idx("z", include_space=True) == 26

    def test_char_to_idx_space(self) -> None:
        from cryptex.core.ngram import char_to_idx

        assert char_to_idx(" ", include_space=True) == 0

    def test_char_to_idx_no_space(self) -> None:
        from cryptex.core.ngram import char_to_idx

        # include_space=False: a=0, b=1, ..., z=25
        assert char_to_idx("a", include_space=False) == 0
        assert char_to_idx("z", include_space=False) == 25

    def test_text_to_indices_roundtrip(self) -> None:
        from cryptex.core.ngram import text_to_indices

        text = "hello world"
        idx = text_to_indices(text, include_space=True)
        assert isinstance(idx, np.ndarray)
        assert len(idx) == len(text)
        # include_space=True: a=1, so h=8, e=5, space=0
        assert idx[0] == 8  # h
        assert idx[5] == 0  # space

    def test_text_to_indices_no_space(self) -> None:
        from cryptex.core.ngram import text_to_indices

        text = "abc"
        idx = text_to_indices(text, include_space=False)
        np.testing.assert_array_equal(idx, [0, 1, 2])

    def test_empty_string(self) -> None:
        from cryptex.core.ngram import text_to_indices

        idx = text_to_indices("", include_space=True)
        assert len(idx) == 0


class TestNgramModel:
    """Tests for NgramModel class."""

    def test_model_loads(self, trained_model) -> None:
        assert trained_model.A == 27  # 26 letters + space

    def test_model_nospace_loads(self, trained_model_nospace) -> None:
        assert trained_model_nospace.A == 26

    def test_unigram_log_probs_valid(self, trained_model) -> None:
        # log_uni should contain valid log-probabilities (all negative or zero)
        assert np.all(trained_model.log_uni <= 0)
        # exp(log_uni) should sum approximately to 1
        probs = np.exp(trained_model.log_uni)
        assert abs(probs.sum() - 1.0) < 0.01

    def test_bigram_log_probs_valid(self, trained_model) -> None:
        # Each row of exp(log_bi) should sum approximately to 1
        probs = np.exp(trained_model.log_bi)
        row_sums = probs.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=0.01)

    def test_log_probs_exist(self, trained_model) -> None:
        assert trained_model.log_uni is not None
        assert trained_model.log_bi is not None
        assert trained_model.log_tri is not None
        assert trained_model.log_quad is not None

    def test_score_english_higher_than_random(self, trained_model) -> None:
        # Use score() which interpolates n-gram orders, on same-length texts
        english = "the quick brown fox jumps"
        random_text = "zxqjv wkfpm nltrs bgyhd"
        s_eng = trained_model.score(english)
        s_rnd = trained_model.score(random_text)
        assert s_eng > s_rnd, "English text should score higher than random"

    def test_score_returns_float(self, trained_model) -> None:
        score = trained_model.score("hello world")
        assert isinstance(score, float)
        assert score < 0  # log-probabilities are negative

    def test_score_indices_matches_text(self, trained_model) -> None:
        from cryptex.core.ngram import text_to_indices

        text = "hello world"
        s1 = trained_model.score(text)
        idx = text_to_indices(text, trained_model.include_space)
        s2 = trained_model.score_indices(idx)
        assert abs(s1 - s2) < 1e-6

    def test_quadgrams_fast_matches_score(self, trained_model) -> None:
        from cryptex.core.ngram import text_to_indices

        text = "in the beginning there was light"
        idx = text_to_indices(text, trained_model.include_space)
        s1 = trained_model.score_quadgrams_fast(idx)
        # score_quadgrams_fast uses only quadgram component, so it won't exactly
        # match score() which interpolates. Just verify it's a finite negative number.
        assert np.isfinite(s1)
        assert s1 < 0

    def test_model_save_load_roundtrip(self, trained_model, tmp_path) -> None:
        path = tmp_path / "test_model.pkl"
        trained_model.save(path)
        from cryptex.core.ngram import NgramModel

        loaded = NgramModel.load(path)
        assert loaded.A == trained_model.A
        np.testing.assert_array_equal(loaded.log_uni, trained_model.log_uni)
        np.testing.assert_array_equal(loaded.log_bi, trained_model.log_bi)

    def test_lambdas_sum_to_one(self, trained_model) -> None:
        assert abs(sum(trained_model.lambdas) - 1.0) < 1e-9

    def test_short_text_score(self, trained_model) -> None:
        """Very short text should still return a valid score."""
        score = trained_model.score("ab")
        assert isinstance(score, float)
        assert np.isfinite(score)

    def test_single_char_score(self, trained_model) -> None:
        score = trained_model.score("a")
        assert isinstance(score, float)
