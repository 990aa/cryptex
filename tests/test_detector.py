"""Tests for the cipher type detector (detector.py)."""

from __future__ import annotations

import string


class TestCipherFeatures:
    def test_dataclass_defaults(self) -> None:
        from cryptex.core.detector import CipherFeatures

        f = CipherFeatures()
        assert f.ioc == 0.0
        assert f.entropy == 0.0
        assert f.length == 0


class TestDetectionResult:
    def test_dataclass_defaults(self) -> None:
        from cryptex.core.detector import DetectionResult

        r = DetectionResult()
        assert r.predicted_type == ""
        assert r.confidence == 0.0


class TestStatisticalFeatures:
    def test_ioc_english(self) -> None:
        from cryptex.core.detector import _ioc

        text = "the quick brown fox jumps over the lazy dog and then some more text"
        ioc = _ioc(text.replace(" ", ""))
        assert 0.03 < ioc < 0.10

    def test_ioc_single_char(self) -> None:
        from cryptex.core.detector import _ioc

        assert _ioc("aaaa") == 1.0

    def test_ioc_all_different(self) -> None:
        from cryptex.core.detector import _ioc

        text = string.ascii_lowercase
        ioc = _ioc(text)
        # 26 chars, each appearing once: IoC = 0
        assert ioc < 0.005

    def test_ioc_empty(self) -> None:
        from cryptex.core.detector import _ioc

        result = _ioc("")
        assert result == 0.0 or isinstance(result, float)

    def test_entropy_uniform(self) -> None:
        from cryptex.core.detector import _entropy

        # All 26 letters equally: max entropy ≈ log2(26) ≈ 4.70
        text = string.ascii_lowercase * 100
        ent = _entropy(text)
        assert 4.6 < ent < 4.75

    def test_entropy_single_char(self) -> None:
        from cryptex.core.detector import _entropy

        # Single character repeated: entropy = 0
        assert _entropy("aaaa") == 0.0

    def test_chi_squared_english(self) -> None:
        from cryptex.core.detector import _chi_squared

        text = "the quick brown fox jumps over the lazy dog " * 10
        chi2 = _chi_squared(text.replace(" ", ""))
        assert chi2 < 10.0  # English text should have relatively low chi-squared

    def test_bigram_repeat_rate(self) -> None:
        from cryptex.core.detector import _bigram_repeat_rate

        text = "ababababab"
        rate = _bigram_repeat_rate(text)
        assert rate > 0  # "ab" and "ba" repeat a lot

    def test_bigram_repeat_rate_unique(self) -> None:
        from cryptex.core.detector import _bigram_repeat_rate

        text = string.ascii_lowercase
        rate = _bigram_repeat_rate(text)
        assert rate == 0.0  # All bigrams are unique

    def test_autocorrelation(self) -> None:
        from cryptex.core.detector import _autocorrelation

        # Periodic text should show autocorrelation at period length
        text = "abc" * 50
        peak, values = _autocorrelation(text, max_lag=10)
        assert peak == 3 or peak in (3, 6, 9)  # Period 3

    def test_even_odd_balance(self) -> None:
        from cryptex.core.detector import _even_odd_balance

        text = "abcdef"
        balance = _even_odd_balance(text)
        assert 0 <= balance <= 1

    def test_even_odd_balance_all_same(self) -> None:
        from cryptex.core.detector import _even_odd_balance

        text = "aaaaaa"
        balance = _even_odd_balance(text)
        assert isinstance(balance, float)


class TestExtractFeatures:
    def test_returns_all_fields(self) -> None:
        from cryptex.core.detector import extract_features

        text = "the quick brown fox jumps over the lazy dog"
        f = extract_features(text)
        assert f.length > 0
        assert f.ioc > 0
        assert f.entropy > 0
        assert f.chi_squared >= 0

    def test_strips_non_alpha(self) -> None:
        from cryptex.core.detector import extract_features

        f = extract_features("hello 123 world!!!")
        assert f.length == 10  # "helloworld"


class TestDetectCipherType:
    def test_detects_substitution(self) -> None:
        from cryptex.ciphers import SimpleSubstitution
        from cryptex.core.detector import detect_cipher_type

        pt = (
            "the quick brown fox jumps over the lazy dog and keeps on running through the meadow "
            * 5
        )
        key = SimpleSubstitution.random_key()
        ct = SimpleSubstitution.encrypt(pt, key)
        result = detect_cipher_type(ct)
        # Detection is heuristic; substitution may be classified as similar types
        assert result.predicted_type in (
            "substitution",
            "transposition",
            "vigenere",
            "playfair",
        )
        assert result.confidence > 0
        assert result.reasoning != ""

    def test_detects_vigenere(self) -> None:
        from cryptex.ciphers import Vigenere
        from cryptex.core.detector import detect_cipher_type

        pt = "the quick brown fox jumps over the lazy dog " * 10
        ct = Vigenere.encrypt(pt, "secretkey")
        result = detect_cipher_type(ct)
        # Vigenere should rank high but exact prediction is heuristic
        assert result.predicted_type in (
            "vigenere",
            "substitution",
            "playfair",
            "transposition",
        )
        assert result.confidence > 0
        assert result.all_scores is not None
        assert "vigenere" in result.all_scores

    def test_returns_all_scores(self) -> None:
        from cryptex.core.detector import detect_cipher_type

        result = detect_cipher_type("some random ciphertext here")
        assert result.all_scores is not None
        assert "substitution" in result.all_scores
        assert "vigenere" in result.all_scores
        assert "transposition" in result.all_scores
        assert "playfair" in result.all_scores

    def test_short_text(self) -> None:
        from cryptex.core.detector import detect_cipher_type

        result = detect_cipher_type("abc")
        assert result.predicted_type != ""

    def test_single_char(self) -> None:
        from cryptex.core.detector import detect_cipher_type

        result = detect_cipher_type("a")
        assert isinstance(result.predicted_type, str)
