"""Tests for robust text I/O helpers (ghost mapping)."""

from __future__ import annotations


class TestGhostMapping:
    def test_extract_core_text(self) -> None:
        from cryptex.io import ghost_map_text

        mapping = ghost_map_text("HeLLo, World! Cafe.")
        assert mapping.core_text == "hello world cafe"
        assert len(mapping.alpha_positions) == len("HeLLoWorldCafe")
        assert len(mapping.uppercase_mask) == len("HeLLoWorldCafe")

    def test_restore_preserves_layout_and_case(self) -> None:
        from cryptex.io import ghost_map_text, restore_ghost_text

        mapping = ghost_map_text("HeLLo, World! Cafe.")
        restored = restore_ghost_text("hello world cafe", mapping)
        assert restored == "HeLLo, World! Cafe."

    def test_unicode_fold_in_core(self) -> None:
        from cryptex.io import ghost_map_text

        mapping = ghost_map_text("Caf\u00e9 d\u00e9j\u00e0 vu")
        assert mapping.core_text == "cafe deja vu"


class TestAlphabetDiscovery:
    def test_discover_effective_alphabet(self) -> None:
        from cryptex.io import discover_effective_alphabet

        alphabet = discover_effective_alphabet("abAB!! 22")
        assert alphabet == ["!", "2", "a", "b"]

    def test_likely_homophonic_cipher(self) -> None:
        from cryptex.io import likely_homophonic_cipher

        small = "abcdefg"
        large = "".join(chr(33 + i) for i in range(50))
        assert likely_homophonic_cipher(small, threshold=40) is False
        assert likely_homophonic_cipher(large, threshold=40) is True
