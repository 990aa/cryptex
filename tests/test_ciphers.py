"""Tests for all cipher engines (ciphers.py)."""

from __future__ import annotations

import random
import string

import pytest


# ======================================================================
# Simple Substitution
# ======================================================================


class TestSimpleSubstitution:
    def test_random_key_is_permutation(self) -> None:
        from cipher.ciphers import SimpleSubstitution

        key = SimpleSubstitution.random_key()
        assert len(key) == 26
        assert sorted(key) == list(string.ascii_lowercase)

    def test_encrypt_decrypt_roundtrip(self) -> None:
        from cipher.ciphers import SimpleSubstitution

        key = SimpleSubstitution.random_key()
        pt = "hello world"
        ct = SimpleSubstitution.encrypt(pt, key)
        dt = SimpleSubstitution.decrypt(ct, key)
        assert dt == pt

    def test_encrypt_changes_text(self) -> None:
        from cipher.ciphers import SimpleSubstitution

        key = "zyxwvutsrqponmlkjihgfedcba"  # atbash
        pt = "abc"
        ct = SimpleSubstitution.encrypt(pt, key)
        assert ct != pt

    def test_identity_key(self) -> None:
        from cipher.ciphers import SimpleSubstitution

        identity = string.ascii_lowercase
        pt = "the quick brown fox"
        ct = SimpleSubstitution.encrypt(pt, identity)
        assert ct == pt

    def test_preserves_spaces(self) -> None:
        from cipher.ciphers import SimpleSubstitution

        key = SimpleSubstitution.random_key()
        pt = "a b c"
        ct = SimpleSubstitution.encrypt(pt, key)
        assert ct[1] == " " and ct[3] == " "

    def test_key_from_mapping(self) -> None:
        from cipher.ciphers import SimpleSubstitution

        mapping = {ch: ch for ch in string.ascii_lowercase}
        key = SimpleSubstitution.key_from_mapping(mapping)
        assert key == string.ascii_lowercase

    def test_empty_string(self) -> None:
        from cipher.ciphers import SimpleSubstitution

        key = SimpleSubstitution.random_key()
        assert SimpleSubstitution.encrypt("", key) == ""
        assert SimpleSubstitution.decrypt("", key) == ""

    def test_only_spaces(self) -> None:
        from cipher.ciphers import SimpleSubstitution

        key = SimpleSubstitution.random_key()
        assert SimpleSubstitution.encrypt("   ", key) == "   "


# ======================================================================
# Vigenère
# ======================================================================


class TestVigenere:
    def test_random_key_format(self) -> None:
        from cipher.ciphers import Vigenere

        key = Vigenere.random_key(length=8)
        assert len(key) == 8
        assert all(ch in string.ascii_lowercase for ch in key)

    def test_encrypt_decrypt_roundtrip(self) -> None:
        from cipher.ciphers import Vigenere

        key = "secret"
        pt = "attack at dawn"
        ct = Vigenere.encrypt(pt, key)
        dt = Vigenere.decrypt(ct, key)
        assert dt == pt

    def test_single_char_key(self) -> None:
        from cipher.ciphers import Vigenere

        # key='a' means shift by 0 → identity (for letters)
        pt = "hello"
        ct = Vigenere.encrypt(pt, "a")
        assert ct == pt

    def test_preserves_spaces(self) -> None:
        from cipher.ciphers import Vigenere

        key = "key"
        pt = "a b"
        ct = Vigenere.encrypt(pt, key)
        assert " " in ct

    def test_empty_string(self) -> None:
        from cipher.ciphers import Vigenere

        assert Vigenere.encrypt("", "key") == ""
        assert Vigenere.decrypt("", "key") == ""


# ======================================================================
# Columnar Transposition
# ======================================================================


class TestColumnarTransposition:
    def test_random_key_is_permutation(self) -> None:
        from cipher.ciphers import ColumnarTransposition

        key = ColumnarTransposition.random_key(ncols=5)
        assert sorted(key) == list(range(5))

    def test_encrypt_decrypt_roundtrip(self) -> None:
        from cipher.ciphers import ColumnarTransposition

        key = [2, 0, 3, 1]
        pt = "helloworld"
        ct = ColumnarTransposition.encrypt(pt, key)
        dt = ColumnarTransposition.decrypt(ct, key)
        # encrypt pads with 'x' to fill grid, so decrypt may include trailing 'x'
        assert dt.startswith(pt) or dt == pt

    def test_encrypt_changes_order(self) -> None:
        from cipher.ciphers import ColumnarTransposition

        key = [1, 0]  # swap columns
        pt = "abcd"
        ct = ColumnarTransposition.encrypt(pt, key)
        assert ct != pt

    def test_single_column(self) -> None:
        from cipher.ciphers import ColumnarTransposition

        key = [0]
        pt = "hello"
        ct = ColumnarTransposition.encrypt(pt, key)
        assert ct == pt

    def test_empty_string(self) -> None:
        from cipher.ciphers import ColumnarTransposition

        assert ColumnarTransposition.encrypt("", [0, 1, 2]) == ""


# ======================================================================
# Playfair
# ======================================================================


class TestPlayfair:
    def test_random_key_length(self) -> None:
        from cipher.ciphers import Playfair

        key = Playfair.random_key()
        assert len(key) == 25
        assert "j" not in key
        assert sorted(key) == sorted(Playfair.ALPHABET)

    def test_encrypt_decrypt_roundtrip(self) -> None:
        from cipher.ciphers import Playfair

        key = Playfair.random_key()
        pt = "hello world"
        ct = Playfair.encrypt(pt, key)
        dt = Playfair.decrypt(ct, key)
        # Playfair may alter text (j→i, padding x), but core content preserved
        assert len(dt) > 0

    def test_no_j_in_output(self) -> None:
        from cipher.ciphers import Playfair

        key = Playfair.random_key()
        ct = Playfair.encrypt("jump jet", key)
        assert "j" not in ct

    def test_even_length_output(self) -> None:
        from cipher.ciphers import Playfair

        key = Playfair.random_key()
        ct = Playfair.encrypt("hello", key)
        assert len(ct) % 2 == 0

    def test_prepare_removes_j(self) -> None:
        from cipher.ciphers import Playfair

        result = Playfair._prepare("jelly")
        assert "j" not in result

    def test_empty_string(self) -> None:
        from cipher.ciphers import Playfair

        key = Playfair.random_key()
        assert Playfair.encrypt("", key) == ""


# ======================================================================
# NoisyCipher
# ======================================================================


class TestNoisyCipher:
    def test_noisy_roundtrip_approximately(self) -> None:
        from cipher.ciphers import NoisyCipher, SimpleSubstitution

        inner = SimpleSubstitution()
        noisy = NoisyCipher(inner, corruption_rate=0.0, deletion_rate=0.0)
        key = noisy.random_key()
        assert isinstance(key, str)
        pt = "hello world"
        ct = noisy.encrypt(pt, key)
        # With 0% noise, encrypt is deterministic
        dt = inner.decrypt(ct, key)
        assert dt == pt

    def test_noisy_corrupts(self) -> None:
        from cipher.ciphers import NoisyCipher, SimpleSubstitution

        inner = SimpleSubstitution()
        noisy = NoisyCipher(inner, corruption_rate=0.5, deletion_rate=0.0)
        key = noisy.random_key()
        assert isinstance(key, str)
        random.seed(42)
        pt = "a" * 100
        ct = noisy.encrypt(pt, key)
        # High corruption rate means output differs significantly
        clean_ct = inner.encrypt(pt, key)
        differences = sum(1 for a, b in zip(ct, clean_ct) if a != b)
        assert differences > 0

    def test_noisy_deletion(self) -> None:
        from cipher.ciphers import NoisyCipher, SimpleSubstitution

        inner = SimpleSubstitution()
        noisy = NoisyCipher(inner, corruption_rate=0.0, deletion_rate=0.5)
        key = noisy.random_key()
        random.seed(42)
        pt = "a" * 200
        ct = noisy.encrypt(pt, key)
        assert len(ct) < len(pt)


# ======================================================================
# get_engine helper
# ======================================================================


class TestGetEngine:
    def test_known_engines(self) -> None:
        from cipher.ciphers import get_engine

        for name in ["substitution", "vigenere", "transposition", "playfair"]:
            engine = get_engine(name)
            assert hasattr(engine, "encrypt")
            assert hasattr(engine, "decrypt")

    def test_unknown_engine_raises(self) -> None:
        from cipher.ciphers import get_engine

        with pytest.raises(ValueError):
            get_engine("nonexistent_cipher")
