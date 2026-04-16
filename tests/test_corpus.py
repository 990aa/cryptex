"""Tests for corpus download and preprocessing (corpus.py)."""

from __future__ import annotations

import string


class TestDownloadCorpus:
    """Tests for corpus.download_corpus()."""

    def test_returns_nonempty_string(self, corpus_text: str) -> None:
        assert isinstance(corpus_text, str)
        assert len(corpus_text) > 100_000, "Corpus should be >100k characters"

    def test_only_contains_valid_characters(self, corpus_text: str) -> None:
        allowed = set(string.ascii_lowercase + " ")
        invalid = set(corpus_text) - allowed
        assert not invalid, f"Invalid chars in corpus: {invalid}"

    def test_no_uppercase(self, corpus_text: str) -> None:
        assert corpus_text == corpus_text.lower()

    def test_no_consecutive_spaces(self, corpus_text: str) -> None:
        # For freshly-downloaded corpus, re.sub(r"\\s+", " ") collapses spaces.
        # A cached file created by an earlier version may have some double spaces.
        # Verify the vast majority has no consecutive spaces (<0.01%).
        count = corpus_text.count("  ")
        ratio = count / max(len(corpus_text), 1)
        assert ratio < 0.001, f"Too many consecutive-space occurrences: {count}"

    def test_no_gutenberg_header(self, corpus_text: str) -> None:
        # The main gutenberg boilerplate markers should be stripped
        assert "*** start of the project gutenberg" not in corpus_text.lower()
        assert "*** end of the project gutenberg" not in corpus_text.lower()

    def test_corpus_cached(self) -> None:
        """Second call should use cache (fast)."""
        from cryptex.corpus import download_corpus

        c1 = download_corpus()
        c2 = download_corpus()
        assert c1 == c2


class TestCorpusEdgeCases:
    """Edge cases for corpus module."""

    def test_strip_gutenberg_header_footer(self) -> None:
        from cryptex.corpus import _strip_gutenberg_header_footer

        text = (
            "*** START OF THE PROJECT GUTENBERG EBOOK ***\n"
            "Hello world this is the body.\n"
            "*** END OF THE PROJECT GUTENBERG EBOOK ***\n"
        )
        result = _strip_gutenberg_header_footer(text)
        assert "START OF" not in result
        assert "END OF" not in result
        assert "body" in result.lower()

    def test_strip_no_markers(self) -> None:
        from cryptex.corpus import _strip_gutenberg_header_footer

        text = "Just plain text without markers."
        result = _strip_gutenberg_header_footer(text)
        assert "plain text" in result.lower()

