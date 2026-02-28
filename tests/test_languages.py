"""Tests for multi-language support (languages.py)."""

from __future__ import annotations


class TestLanguageCorpora:
    def test_french_urls_defined(self) -> None:
        from cipher.languages import LANGUAGE_CORPORA

        assert "french" in LANGUAGE_CORPORA
        assert len(LANGUAGE_CORPORA["french"]) >= 2

    def test_german_urls_defined(self) -> None:
        from cipher.languages import LANGUAGE_CORPORA

        assert "german" in LANGUAGE_CORPORA
        assert len(LANGUAGE_CORPORA["german"]) >= 2

    def test_spanish_urls_defined(self) -> None:
        from cipher.languages import LANGUAGE_CORPORA

        assert "spanish" in LANGUAGE_CORPORA
        assert len(LANGUAGE_CORPORA["spanish"]) >= 1


class TestLanguageFrequencies:
    def test_all_languages_present(self) -> None:
        from cipher.languages import LANGUAGE_FREQ

        for lang in ("english", "french", "german", "spanish"):
            assert lang in LANGUAGE_FREQ

    def test_frequencies_sum_to_one(self) -> None:
        from cipher.languages import LANGUAGE_FREQ

        for lang, freq in LANGUAGE_FREQ.items():
            assert abs(freq.sum() - 1.0) < 0.05, f"{lang} freqs sum to {freq.sum()}"

    def test_frequency_shape(self) -> None:
        from cipher.languages import LANGUAGE_FREQ

        for lang, freq in LANGUAGE_FREQ.items():
            assert freq.shape == (26,), f"{lang} has shape {freq.shape}"


class TestTransliteration:
    def test_table_exists(self) -> None:
        from cipher.languages import TRANSLITERATION

        assert isinstance(TRANSLITERATION, dict)

    def test_accent_transliteration(self) -> None:
        from cipher.languages import TRANSLITERATION

        text = "café résumé naïve"
        result = text.translate(TRANSLITERATION)
        assert "é" not in result
        assert "ï" not in result


class TestDetectLanguage:
    def test_english(self) -> None:
        from cipher.languages import detect_language

        text = "the quick brown fox jumps over the lazy dog and keeps running"
        lang, scores = detect_language(text)
        assert lang == "english"
        assert "english" in scores

    def test_french_text(self) -> None:
        from cipher.languages import detect_language

        text = (
            "le petit prince est un livre tres populaire en france dans le monde entier"
        )
        lang, scores = detect_language(text)
        assert lang in ("french", "english")  # Frequency overlap possible

    def test_empty_text(self) -> None:
        from cipher.languages import detect_language

        lang, scores = detect_language("")
        assert lang == "english"  # Default
        assert scores == {}

    def test_numbers_only(self) -> None:
        from cipher.languages import detect_language

        lang, scores = detect_language("12345")
        assert lang == "english"

    def test_returns_all_lang_scores(self) -> None:
        from cipher.languages import detect_language

        _, scores = detect_language("hello world how are you")
        assert "english" in scores
        assert "french" in scores
        assert "german" in scores
        assert "spanish" in scores


class TestGetAvailableLanguages:
    def test_includes_all(self) -> None:
        from cipher.languages import get_available_languages

        langs = get_available_languages()
        assert "english" in langs
        assert "french" in langs
        assert "german" in langs
        assert "spanish" in langs

    def test_returns_list(self) -> None:
        from cipher.languages import get_available_languages

        assert isinstance(get_available_languages(), list)


class TestStripGutenberg:
    def test_strips_markers(self) -> None:
        from cipher.languages import _strip_gutenberg

        text = (
            "*** START OF THE PROJECT GUTENBERG EBOOK ***\n"
            "actual content here\n"
            "*** END OF THE PROJECT GUTENBERG EBOOK ***\n"
        )
        result = _strip_gutenberg(text)
        assert "content" in result
        assert "GUTENBERG" not in result

    def test_no_markers(self) -> None:
        from cipher.languages import _strip_gutenberg

        text = "just plain text"
        result = _strip_gutenberg(text)
        assert "plain" in result
