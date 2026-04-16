"""Vigenere solver with adaptive key-length and n-gram scoring."""

from __future__ import annotations

import math
import string
from collections import Counter
from dataclasses import dataclass
from typing import Callable

import numpy as np

from cryptex.core.ngram import NgramModel, text_to_indices


@dataclass
class VigenereConfig:
    max_key_length: int = 20
    top_key_lengths: int = 8
    callback_interval: int = 1


@dataclass
class VigenereResult:
    best_key: str = ""
    best_plaintext: str = ""
    best_score: float = -math.inf
    detected_key_length: int = 0


VigenereCallback = Callable[[str, str, float], None]

ENGLISH_IOC = 0.0667

# Expected English letter frequencies (a-z)
ENGLISH_FREQ = np.array(
    [
        0.08167,
        0.01492,
        0.02782,
        0.04253,
        0.12702,
        0.02228,
        0.02015,
        0.06094,
        0.06966,
        0.00153,
        0.00772,
        0.04025,
        0.02406,
        0.06749,
        0.07507,
        0.01929,
        0.00095,
        0.05987,
        0.06327,
        0.09056,
        0.02758,
        0.00978,
        0.02360,
        0.00150,
        0.01974,
        0.00074,
    ]
)

COMMON_KEY_WORDS = [
    "key",
    "cipher",
    "secret",
    "delta",
    "alpha",
    "omega",
    "matrix",
    "vector",
    "shadow",
    "bridge",
    "signal",
    "echo",
    "north",
    "south",
    "east",
    "west",
    "atlas",
    "phoenix",
    "falcon",
    "apollo",
    "london",
    "paris",
    "berlin",
    "madrid",
]


def _letters_only(text: str) -> str:
    return "".join(ch for ch in text if ch in string.ascii_lowercase)


def _ioc(text: str) -> float:
    counts = Counter(ch for ch in text if ch in string.ascii_lowercase)
    n = sum(counts.values())
    if n <= 1:
        return 0.0
    return sum(c * (c - 1) for c in counts.values()) / (n * (n - 1))


def _average_ioc(ciphertext: str, key_len: int) -> float:
    subseqs: list[list[str]] = [[] for _ in range(key_len)]
    idx = 0
    for ch in ciphertext:
        if ch in string.ascii_lowercase:
            subseqs[idx % key_len].append(ch)
            idx += 1
    iocs = [_ioc("".join(s)) for s in subseqs if len(s) > 1]
    return sum(iocs) / len(iocs) if iocs else 0.0


def detect_key_length(ciphertext: str, max_len: int = 20, top_n: int = 3) -> list[int]:
    scores: list[tuple[int, float]] = []
    for kl in range(1, max_len + 1):
        avg = _average_ioc(ciphertext, kl)
        scores.append((kl, abs(avg - ENGLISH_IOC)))
    scores.sort(key=lambda x: x[1])
    return [kl for kl, _ in scores[:top_n]]


def _kasiski_distances(ciphertext: str, ngram_len: int = 3) -> list[int]:
    letters = _letters_only(ciphertext)
    positions: dict[str, list[int]] = {}
    for i in range(len(letters) - ngram_len + 1):
        gram = letters[i : i + ngram_len]
        positions.setdefault(gram, []).append(i)

    gaps: list[int] = []
    for poslist in positions.values():
        if len(poslist) >= 2:
            for i in range(len(poslist) - 1):
                gaps.append(poslist[i + 1] - poslist[i])
    return gaps


def _kasiski_length_vote(ciphertext: str, key_len: int) -> float:
    gaps = _kasiski_distances(ciphertext)
    if not gaps:
        return 0.0
    divisible = sum(1 for g in gaps if g % key_len == 0)
    return divisible / max(len(gaps), 1)


def kasiski_key_lengths(ciphertext: str, max_len: int = 20) -> list[int]:
    gaps = _kasiski_distances(ciphertext)
    if not gaps:
        return []
    factor_counts: Counter[int] = Counter()
    for g in gaps:
        for f in range(2, max_len + 1):
            if g % f == 0:
                factor_counts[f] += 1
    return [kl for kl, _ in factor_counts.most_common(5)]


def _solve_caesar(subseq: str) -> int:
    """Legacy chi-squared Caesar solver kept for compatibility."""
    counts = np.zeros(26)
    for ch in subseq:
        if ch in string.ascii_lowercase:
            counts[ord(ch) - ord("a")] += 1
    n = counts.sum()
    if n == 0:
        return 0

    best_shift = 0
    best_chi2 = math.inf
    for shift in range(26):
        shifted = np.roll(counts, -shift)
        freq = shifted / n
        chi2 = ((freq - ENGLISH_FREQ) ** 2 / (ENGLISH_FREQ + 1e-10)).sum()
        if chi2 < best_chi2:
            best_chi2 = chi2
            best_shift = shift
    return best_shift


def _solve_caesar_ngram(subseq: str, model: NgramModel) -> int:
    """Find the Caesar shift maximizing adaptive n-gram score."""
    best_shift = 0
    best_score = -math.inf

    clean = _letters_only(subseq)
    if not clean:
        return 0

    for shift in range(26):
        decrypted = "".join(
            chr((ord(c) - ord("a") - shift) % 26 + ord("a")) for c in clean
        )
        idx = text_to_indices(decrypted, include_space=False)
        score = model.score_adaptive(idx)
        if score > best_score:
            best_score = score
            best_shift = shift
    return best_shift


def _get_subseq(ciphertext: str, key_len: int, position: int) -> str:
    letters = _letters_only(ciphertext)
    return "".join(letters[i] for i in range(position, len(letters), key_len))


def _apply_vigenere_key(ciphertext: str, shifts: list[int]) -> str:
    out: list[str] = []
    ki = 0
    key_len = max(len(shifts), 1)
    for ch in ciphertext:
        if ch in string.ascii_lowercase:
            shift = shifts[ki % key_len]
            out.append(chr((ord(ch) - ord("a") - shift) % 26 + ord("a")))
            ki += 1
        else:
            out.append(ch)
    return "".join(out)


def _refine_to_word_key(shifts: list[int], word_list: list[str]) -> list[int]:
    """Snap shift sequence to a nearby dictionary word key when plausible."""
    kl = len(shifts)
    candidates = [w for w in word_list if len(w) == kl and w.isalpha()]
    if not candidates:
        return shifts

    best_word: str | None = None
    best_dist = math.inf
    for word in candidates:
        dist = 0
        for s, ch in zip(shifts, word.lower()):
            target = ord(ch) - ord("a")
            dist += min((s - target) % 26, (target - s) % 26)
        if dist < best_dist:
            best_dist = dist
            best_word = word.lower()

    if best_word and best_dist <= kl:
        return [ord(c) - ord("a") for c in best_word]
    return shifts


def _score_key_length(ciphertext: str, key_len: int, model: NgramModel) -> float:
    ioc_score = -abs(_average_ioc(ciphertext, key_len) - ENGLISH_IOC)
    kasiski_score = _kasiski_length_vote(ciphertext, key_len)

    shifts = [
        _solve_caesar_ngram(_get_subseq(ciphertext, key_len, pos), model)
        for pos in range(key_len)
    ]
    plaintext = _apply_vigenere_key(ciphertext, shifts)
    pt_idx = text_to_indices(plaintext, include_space=model.include_space)
    ngram_score = model.score_adaptive(pt_idx)

    return (
        0.3 * ioc_score
        + 0.1 * kasiski_score
        + 0.6 * (ngram_score / max(len(pt_idx), 1))
    )


def crack_vigenere(
    ciphertext: str,
    model: NgramModel,
    config: VigenereConfig | None = None,
    callback: VigenereCallback | None = None,
) -> VigenereResult:
    """Crack Vigenere using IoC+Kasiski+adaptive n-gram scoring."""
    if config is None:
        config = VigenereConfig()

    result = VigenereResult()
    letters = _letters_only(ciphertext)
    if not letters:
        return result

    max_len = min(max(config.max_key_length, 1), len(letters))
    length_scores = [
        (kl, _score_key_length(letters, kl, model)) for kl in range(1, max_len + 1)
    ]
    length_scores.sort(key=lambda x: x[1], reverse=True)

    top_n = min(max(config.top_key_lengths, 1), len(length_scores))
    for rank, (kl, _combined_score) in enumerate(length_scores[:top_n], start=1):
        shifts = [
            _solve_caesar_ngram(_get_subseq(letters, kl, pos), model)
            for pos in range(kl)
        ]
        shifts = _refine_to_word_key(shifts, COMMON_KEY_WORDS)

        plaintext = _apply_vigenere_key(ciphertext, shifts)
        pt_idx = text_to_indices(plaintext, include_space=model.include_space)
        score = model.score_adaptive(pt_idx)
        key = "".join(chr(ord("a") + s) for s in shifts)

        if callback and rank % max(config.callback_interval, 1) == 0:
            callback(key, plaintext[:120], score)

        if score > result.best_score:
            result.best_score = score
            result.best_key = key
            result.best_plaintext = plaintext
            result.detected_key_length = kl

    return result
