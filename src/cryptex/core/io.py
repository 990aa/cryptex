"""Robust input/output helpers for real-world ciphertext.

Implements a ghost-mapping pipeline:
1) Extract a normalized cryptographic core for solvers.
2) Preserve positional metadata (case + non-alpha symbols).
3) Reconstruct readable plaintext in the original layout.
"""

from __future__ import annotations

import string
import unicodedata
from dataclasses import dataclass


@dataclass
class GhostMapping:
    """Metadata required to reconstruct plaintext into original text layout."""

    original_text: str
    core_text: str
    alpha_positions: list[int]
    uppercase_mask: list[bool]


def _ascii_letter_from_char(ch: str) -> str | None:
    """Best-effort map from Unicode character to a-z."""
    if not ch:
        return None
    if "a" <= ch <= "z":
        return ch
    if "A" <= ch <= "Z":
        return ch.lower()

    folded = unicodedata.normalize("NFKD", ch)
    for c in folded:
        if "a" <= c <= "z":
            return c
        if "A" <= c <= "Z":
            return c.lower()
    return None


def discover_effective_alphabet(text: str, normalize_case: bool = True) -> list[str]:
    """Return sorted unique symbols that appear in text.

    This is used for automatic solver routing, e.g. detecting potential
    homophonic substitution when the symbol inventory is very large.
    """
    if normalize_case:
        text = text.casefold()
    symbols = {ch for ch in text if not ch.isspace()}
    return sorted(symbols)


def ghost_map_text(text: str) -> GhostMapping:
    """Extract solver-friendly core text while preserving reconstruction metadata."""
    core_chars: list[str] = []
    alpha_positions: list[int] = []
    uppercase_mask: list[bool] = []

    last_was_space = True
    for idx, ch in enumerate(text):
        mapped = _ascii_letter_from_char(ch)
        if mapped is not None:
            core_chars.append(mapped)
            alpha_positions.append(idx)
            uppercase_mask.append(ch.isupper())
            last_was_space = False
            continue

        if ch.isspace() or ch in string.punctuation or ch.isdigit():
            if not last_was_space:
                core_chars.append(" ")
                last_was_space = True

    core_text = "".join(core_chars).strip()
    return GhostMapping(
        original_text=text,
        core_text=core_text,
        alpha_positions=alpha_positions,
        uppercase_mask=uppercase_mask,
    )


def restore_ghost_text(plaintext_core: str, mapping: GhostMapping) -> str:
    """Restore case and punctuation into decrypted text.

    Only alphabetic characters in the original text are replaced by decrypted
    letters; symbols and punctuation stay exactly where they were.
    """
    letters = [ch for ch in plaintext_core if "a" <= ch <= "z"]
    out = list(mapping.original_text)

    li = 0
    for pos, is_upper in zip(mapping.alpha_positions, mapping.uppercase_mask):
        if li < len(letters):
            ch = letters[li]
            li += 1
        else:
            fallback = _ascii_letter_from_char(mapping.original_text[pos])
            ch = fallback if fallback is not None else "x"

        out[pos] = ch.upper() if is_upper else ch

    return "".join(out)


def likely_homophonic_cipher(text: str, threshold: int = 40) -> bool:
    """Heuristic: large effective alphabet often indicates homophonic cipher."""
    alphabet = discover_effective_alphabet(text)
    return len(alphabet) >= threshold

