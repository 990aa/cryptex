"""Known-plaintext attack helpers for classical ciphers."""

from __future__ import annotations

import string
from dataclasses import dataclass, field


@dataclass
class KPAResult:
    substitutions: list[dict[str, str]] = field(default_factory=list)
    caesar_shifts: list[int] = field(default_factory=list)
    affine_keys: list[tuple[int, int]] = field(default_factory=list)
    vigenere_key_guesses: list[str] = field(default_factory=list)


def _sanitize_letters(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch in string.ascii_lowercase)


def _inverse_mod(a: int, m: int) -> int | None:
    for x in range(1, m):
        if (a * x) % m == 1:
            return x
    return None


def _check_substitution_pair(ct: str, pt: str) -> dict[str, str] | None:
    c2p: dict[str, str] = {}
    p_used: set[str] = set()
    for c, p in zip(ct, pt):
        if c in c2p and c2p[c] != p:
            return None
        if c not in c2p and p in p_used:
            return None
        c2p[c] = p
        p_used.add(p)
    return c2p


def _caesar_shift(ct: str, pt: str) -> int | None:
    if not ct:
        return None
    shifts = {(ord(c) - ord(p)) % 26 for c, p in zip(ct, pt)}
    if len(shifts) == 1:
        return shifts.pop()
    return None


def _affine_key(ct: str, pt: str) -> tuple[int, int] | None:
    if len(ct) < 2:
        return None
    pairs = [(ord(c) - ord("a"), ord(p) - ord("a")) for c, p in zip(ct, pt)]

    for a in range(26):
        inv = _inverse_mod(a, 26)
        if inv is None:
            continue
        for b in range(26):
            ok = True
            for c, p in pairs:
                if (inv * (c - b)) % 26 != p:
                    ok = False
                    break
            if ok:
                return (a, b)
    return None


def _vigenere_guess(ct: str, pt: str, max_key_len: int = 20) -> list[str]:
    if not ct:
        return []

    shifts = [(ord(c) - ord(p)) % 26 for c, p in zip(ct, pt)]
    guesses: list[str] = []
    for kl in range(1, min(max_key_len, len(shifts)) + 1):
        ok = True
        for i, s in enumerate(shifts):
            if s != shifts[i % kl]:
                ok = False
                break
        if ok:
            key = "".join(chr(ord("a") + shifts[i]) for i in range(kl))
            guesses.append(key)
    return guesses


def known_plaintext_attack(ciphertext: str, known_plaintext: str) -> KPAResult:
    """Slide known plaintext over ciphertext and extract compatible key clues."""
    ct = _sanitize_letters(ciphertext)
    pt = _sanitize_letters(known_plaintext)
    result = KPAResult()

    if not ct or not pt or len(pt) > len(ct):
        return result

    seen_subs: set[tuple[tuple[str, str], ...]] = set()
    seen_caesar: set[int] = set()
    seen_affine: set[tuple[int, int]] = set()
    seen_vig: set[str] = set()

    for start in range(0, len(ct) - len(pt) + 1):
        window = ct[start : start + len(pt)]

        sub = _check_substitution_pair(window, pt)
        if sub is not None:
            fingerprint = tuple(sorted(sub.items()))
            if fingerprint not in seen_subs:
                seen_subs.add(fingerprint)
                result.substitutions.append(sub)

        shift = _caesar_shift(window, pt)
        if shift is not None and shift not in seen_caesar:
            seen_caesar.add(shift)
            result.caesar_shifts.append(shift)

        affine = _affine_key(window, pt)
        if affine is not None and affine not in seen_affine:
            seen_affine.add(affine)
            result.affine_keys.append(affine)

        for key in _vigenere_guess(window, pt):
            if key not in seen_vig:
                seen_vig.add(key)
                result.vigenere_key_guesses.append(key)

    return result

