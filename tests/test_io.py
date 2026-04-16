"""Tests for robust text I/O helpers (ghost mapping)."""

from __future__ import annotations

import string

from hypothesis import given, settings
from hypothesis import strategies as st


class TestGhostMapping:
    def test_extract_core_text(self) -> None:
        from cryptex.core.io import ghost_map_text

        mapping = ghost_map_text("HeLLo, World! Cafe.")
        assert mapping.core_text == "hello world cafe"
        assert len(mapping.alpha_positions) == len("HeLLoWorldCafe")
        assert len(mapping.uppercase_mask) == len("HeLLoWorldCafe")

    def test_restore_preserves_layout_and_case(self) -> None:
        from cryptex.core.io import ghost_map_text, restore_ghost_text

        mapping = ghost_map_text("HeLLo, World! Cafe.")
        restored = restore_ghost_text("hello world cafe", mapping)
        assert restored == "HeLLo, World! Cafe."

    def test_unicode_fold_in_core(self) -> None:
        from cryptex.core.io import ghost_map_text

        mapping = ghost_map_text("Caf\u00e9 d\u00e9j\u00e0 vu")
        assert mapping.core_text == "cafe deja vu"

    def test_restore_length_mismatch_pads_with_question_mark(self) -> None:
        from cryptex.core.io import ghost_map_text, restore_ghost_text

        mapping = ghost_map_text("Hello, World!")
        restored = restore_ghost_text("abc", mapping)
        assert restored == "Abc??, ?????!"

    def test_roundtrip_invariant(self) -> None:
        from cryptex.core.io import test_ghost_roundtrip_invariant

        assert test_ghost_roundtrip_invariant("HeLLo, World! Cafe 123") is True


class TestAlphabetDiscovery:
    def test_discover_effective_alphabet(self) -> None:
        from cryptex.core.io import discover_effective_alphabet

        alphabet = discover_effective_alphabet("abAB!! 22")
        assert alphabet == ["!", "2", "a", "b"]

    def test_likely_homophonic_cipher(self) -> None:
        from cryptex.core.io import likely_homophonic_cipher

        small = "abcdefg"
        large = "".join(chr(33 + i) for i in range(50))
        assert likely_homophonic_cipher(small, threshold=40) is False
        assert likely_homophonic_cipher(large, threshold=40) is True


class TestGhostMappingProperty:
    @given(
        st.text(
            alphabet=string.ascii_lowercase + string.ascii_uppercase + " ",
            min_size=50,
            max_size=500,
        )
    )
    @settings(max_examples=100)
    def test_ghost_map_roundtrip_property(self, text: str) -> None:
        from cryptex.core.io import ghost_map_text, restore_ghost_text

        mapping = ghost_map_text(text)
        restored = restore_ghost_text(mapping.core_text, mapping)

        orig_letters = [c.lower() for c in text if c.isalpha()]
        rest_letters = [c for c in restored if c.isalpha()]
        assert orig_letters == [c.lower() for c in rest_letters]
