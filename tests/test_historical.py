"""Tests for historical ciphers (historical.py)."""

from __future__ import annotations

import pytest


class TestHistoricalCipher:
    def test_fields(self) -> None:
        from cipher.historical import HistoricalCipher

        h = HistoricalCipher(
            name="Test",
            description="A test cipher",
            ciphertext="abc",
            cipher_type="substitution",
        )
        assert h.name == "Test"
        assert h.known_plaintext is None
        assert h.known_key is None
        assert h.difficulty == "unknown"


class TestGetHistoricalCiphers:
    def test_returns_list(self) -> None:
        from cipher.historical import get_historical_ciphers

        ciphers = get_historical_ciphers()
        assert isinstance(ciphers, list)
        assert len(ciphers) == 6

    def test_each_has_required_fields(self) -> None:
        from cipher.historical import get_historical_ciphers

        for hc in get_historical_ciphers():
            assert hc.name != ""
            assert hc.description != ""
            assert hc.ciphertext != ""
            assert hc.cipher_type in (
                "substitution",
                "vigenere",
                "transposition",
                "playfair",
            )

    def test_cipher_types_diverse(self) -> None:
        from cipher.historical import get_historical_ciphers

        types = {hc.cipher_type for hc in get_historical_ciphers()}
        assert len(types) >= 3  # At least 3 different cipher types

    def test_some_have_known_plaintext(self) -> None:
        from cipher.historical import get_historical_ciphers

        with_pt = [hc for hc in get_historical_ciphers() if hc.known_plaintext]
        assert len(with_pt) >= 2

    def test_difficulties_varied(self) -> None:
        from cipher.historical import get_historical_ciphers

        difficulties = {hc.difficulty for hc in get_historical_ciphers()}
        assert len(difficulties) >= 2


class TestRunHistoricalChallenge:
    @pytest.mark.timeout(120)
    def test_runs_caesar(self) -> None:
        """Test the easiest historical cipher (Caesar/ROT-13)."""
        from cipher.historical import get_historical_ciphers, run_historical_challenge

        ciphers = get_historical_ciphers()
        # Find the Caesar one
        caesar = next(
            hc
            for hc in ciphers
            if "caesar" in hc.name.lower() or "rot" in hc.name.lower()
        )
        result = run_historical_challenge(caesar)
        assert "decrypted" in result
        assert "time_seconds" in result
        time_seconds = result.get("time_seconds")
        assert isinstance(time_seconds, (int, float))
        assert time_seconds > 0

    @pytest.mark.timeout(120)
    def test_returns_required_keys(self) -> None:
        from cipher.historical import get_historical_ciphers, run_historical_challenge

        ciphers = get_historical_ciphers()
        result = run_historical_challenge(ciphers[0])
        assert "name" in result
        assert "cipher_type" in result
        assert "decrypted" in result
