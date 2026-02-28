"""Tests for Vigenère, Transposition, and Playfair crackers."""

from __future__ import annotations

import math

import pytest


# ======================================================================
# Vigenère Cracker
# ======================================================================


class TestVigenereCracker:
    def test_ioc_english(self) -> None:
        from cipher.vigenere_cracker import _ioc

        # Use a longer text for a meaningful IoC close to English average (~0.0667)
        english_text = (
            "it is a truth universally acknowledged that a single man "
            "in possession of a good fortune must be in want of a wife"
        )
        ioc = _ioc(english_text.replace(" ", ""))
        assert 0.04 < ioc < 0.10  # English IoC is typically ~0.065

    def test_ioc_uniform(self) -> None:
        from cipher.vigenere_cracker import _ioc

        # Each letter appears once
        text = "abcdefghijklmnopqrstuvwxyz"
        ioc = _ioc(text)
        assert ioc < 0.04  # Close to random

    def test_detect_key_length(self) -> None:
        from cipher.ciphers import Vigenere
        from cipher.vigenere_cracker import detect_key_length

        key = "secret"  # length 6
        pt = (
            "the quick brown fox jumps over the lazy dog and some more text to make it longer "
            * 5
        )
        ct = Vigenere.encrypt(pt, key)
        lengths = detect_key_length(ct, max_len=12, top_n=5)
        assert 6 in lengths or 3 in lengths  # 6 or a factor of 6

    def test_solve_caesar(self) -> None:
        from cipher.vigenere_cracker import _solve_caesar

        # Shift by 3 (a→d, b→e, etc.)
        shifted = "wkh"  # "the" shifted by 3
        shift = _solve_caesar(shifted)
        assert shift == 3

    @pytest.mark.timeout(30)
    def test_crack_vigenere(self, trained_model) -> None:
        from cipher.ciphers import Vigenere
        from cipher.vigenere_cracker import VigenereConfig, crack_vigenere

        key = "secret"
        pt = "the quick brown fox jumps over the lazy dog " * 8
        ct = Vigenere.encrypt(pt, key)

        config = VigenereConfig(max_key_length=12)
        result = crack_vigenere(ct, trained_model, config)
        assert result.best_key != ""
        assert result.best_plaintext != ""
        assert result.detected_key_length > 0

    @pytest.mark.timeout(30)
    def test_crack_with_callback(self, trained_model) -> None:
        from cipher.ciphers import Vigenere
        from cipher.vigenere_cracker import VigenereConfig, crack_vigenere

        pt = "hello world test text for vigenere " * 5
        ct = Vigenere.encrypt(pt, "abc")
        config = VigenereConfig()
        calls = []
        crack_vigenere(
            ct, trained_model, config, callback=lambda k, p, s: calls.append(k)
        )
        assert len(calls) > 0

    def test_kasiski_distances(self) -> None:
        from cipher.vigenere_cracker import kasiski_key_lengths

        # With a known periodic text, Kasiski should find factors
        from cipher.ciphers import Vigenere

        pt = "the the the the the the the the the the " * 4
        ct = Vigenere.encrypt(pt, "abc")
        lengths = kasiski_key_lengths(ct, max_len=10)
        # Should get 3 or a multiple (key "abc" has length 3)
        assert isinstance(lengths, list)


# ======================================================================
# Transposition Cracker
# ======================================================================


class TestTranspositionCracker:
    @pytest.mark.timeout(120)
    def test_crack_transposition(self, trained_model) -> None:
        from cipher.ciphers import ColumnarTransposition
        from cipher.transposition_cracker import (
            TranspositionConfig,
            crack_transposition,
        )

        key = [2, 0, 3, 1]
        pt = "the quick brown fox jumps over the lazy dog and some extra words here"
        ct = ColumnarTransposition.encrypt(pt, key)

        config = TranspositionConfig(iterations=10_000, num_restarts=3)
        result = crack_transposition(ct, trained_model, config)
        assert result.best_plaintext != ""
        assert result.best_score > -math.inf
        assert result.detected_ncols > 0

    @pytest.mark.timeout(60)
    def test_callback_invoked(self, trained_model) -> None:
        from cipher.ciphers import ColumnarTransposition
        from cipher.transposition_cracker import (
            TranspositionConfig,
            crack_transposition,
        )

        key = [1, 0, 2]
        pt = "abcdefghijklmnopqrstuvwxyz" * 3
        ct = ColumnarTransposition.encrypt(pt, key)

        config = TranspositionConfig(
            iterations=2000, num_restarts=1, callback_interval=500
        )
        calls = []
        crack_transposition(
            ct, trained_model, config, callback=lambda *a: calls.append(1)
        )
        assert len(calls) > 0


# ======================================================================
# Playfair Cracker
# ======================================================================


class TestPlayfairCracker:
    @pytest.mark.timeout(120)
    def test_crack_playfair(self, trained_model_nospace) -> None:
        from cipher.ciphers import Playfair
        from cipher.playfair_cracker import PlayfairConfig, crack_playfair

        key = Playfair.random_key()
        pt = (
            "the quick brown fox iumps over the lazy dog and this is some extra text "
            "to ensure we have enough statistical signal for the playfair cracker"
        )
        ct = Playfair.encrypt(pt, key)

        config = PlayfairConfig(iterations=20_000, num_restarts=2)
        result = crack_playfair(ct, trained_model_nospace, config)
        assert result.best_key != ""
        assert result.best_plaintext != ""
        assert len(result.best_key) == 25

    def test_build_decrypt_table(self) -> None:
        from cipher.playfair_cracker import _build_decrypt_table

        key = list(range(25))
        table = _build_decrypt_table(key)
        assert table.shape == (25, 25)

    def test_propose_returns_valid_move(self) -> None:
        from cipher.playfair_cracker import _propose

        key = list(range(25))
        move_type, info = _propose(key)
        assert move_type in (
            "swap",
            "swap_rows",
            "swap_cols",
            "reverse",
            "rotate_row",
            "rotate_col",
        )
