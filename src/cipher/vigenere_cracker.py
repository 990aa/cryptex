"""Vigenère cipher cracker.

Strategy:
1. Determine key length using Index of Coincidence (IoC) / Kasiski examination.
2. Split ciphertext into `key_length` interleaved subsequences.
3. Solve each subsequence independently as a Caesar shift (frequency analysis).
"""

from __future__ import annotations

import math
import string
from collections import Counter
from dataclasses import dataclass
from typing import Callable

import numpy as np

from cipher.ngram import NgramModel


# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------


@dataclass
class VigenereConfig:
    max_key_length: int = 20
    """Try key lengths from 1 to this value."""

    top_key_lengths: int = 3
    """Evaluate the top-N candidate key-lengths by IoC."""

    callback_interval: int = 1


@dataclass
class VigenereResult:
    best_key: str = ""
    best_plaintext: str = ""
    best_score: float = -math.inf
    detected_key_length: int = 0


VigenereCallback = Callable[[str, str, float], None]

# English Index of Coincidence ≈ 0.0667
ENGLISH_IOC = 0.0667


# ------------------------------------------------------------------
# Index of Coincidence
# ------------------------------------------------------------------


def _ioc(text: str) -> float:
    """Compute the Index of Coincidence for *text* (letters only)."""
    counts = Counter(ch for ch in text if ch in string.ascii_lowercase)
    n = sum(counts.values())
    if n <= 1:
        return 0.0
    return sum(c * (c - 1) for c in counts.values()) / (n * (n - 1))


def _average_ioc(ciphertext: str, key_len: int) -> float:
    """Split cipher into *key_len* subsequences and average their IoC."""
    subseqs: list[list[str]] = [[] for _ in range(key_len)]
    idx = 0
    for ch in ciphertext:
        if ch in string.ascii_lowercase:
            subseqs[idx % key_len].append(ch)
            idx += 1
    iocs = [_ioc("".join(s)) for s in subseqs if len(s) > 1]
    return sum(iocs) / len(iocs) if iocs else 0.0


def detect_key_length(ciphertext: str, max_len: int = 20, top_n: int = 3) -> list[int]:
    """Return the *top_n* most likely key lengths ranked by closeness of
    average IoC to expected English IoC."""
    scores: list[tuple[int, float]] = []
    for kl in range(1, max_len + 1):
        avg = _average_ioc(ciphertext, kl)
        scores.append((kl, abs(avg - ENGLISH_IOC)))
    scores.sort(key=lambda x: x[1])
    return [kl for kl, _ in scores[:top_n]]


# ------------------------------------------------------------------
# Kasiski Examination (supporting evidence)
# ------------------------------------------------------------------


def _kasiski_distances(ciphertext: str, ngram_len: int = 3) -> list[int]:
    """Find repeated n-grams and return the gaps between their positions."""
    letters = "".join(ch for ch in ciphertext if ch in string.ascii_lowercase)
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


def kasiski_key_lengths(ciphertext: str, max_len: int = 20) -> list[int]:
    """Use Kasiski examination to suggest key lengths (by GCD of gaps)."""
    gaps = _kasiski_distances(ciphertext)
    if not gaps:
        return []
    factor_counts: Counter[int] = Counter()
    for g in gaps:
        for f in range(2, max_len + 1):
            if g % f == 0:
                factor_counts[f] += 1
    if not factor_counts:
        return []
    return [kl for kl, _ in factor_counts.most_common(5)]


# ------------------------------------------------------------------
# Frequency-based Caesar solver for a single subsequence
# ------------------------------------------------------------------

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


def _solve_caesar(subseq: str) -> int:
    """Return the best Caesar shift (0-25) for *subseq* using chi-squared."""
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


# ------------------------------------------------------------------
# Full Vigenère cracker
# ------------------------------------------------------------------


def crack_vigenere(
    ciphertext: str,
    model: NgramModel,
    config: VigenereConfig | None = None,
    callback: VigenereCallback | None = None,
) -> VigenereResult:
    """Crack a Vigenère cipher.

    1. Detect key length via IoC (+ Kasiski for cross-validation).
    2. For each candidate key length, solve each Caesar sub-problem.
    3. Score full decryptions; return the best.
    """
    if config is None:
        config = VigenereConfig()

    letters_only = "".join(ch for ch in ciphertext if ch in string.ascii_lowercase)
    result = VigenereResult()

    candidate_lengths = detect_key_length(
        letters_only, config.max_key_length, config.top_key_lengths
    )
    kasiski_lengths = kasiski_key_lengths(letters_only, config.max_key_length)

    # Merge and deduplicate, preferring IoC order
    seen: set[int] = set()
    all_lengths: list[int] = []
    for kl in candidate_lengths + kasiski_lengths:
        if kl not in seen:
            seen.add(kl)
            all_lengths.append(kl)

    from cipher.ciphers import Vigenere
    from cipher.ngram import text_to_indices

    for kl in all_lengths:
        # Split into subsequences
        subseqs: list[str] = ["" for _ in range(kl)]
        idx = 0
        for ch in letters_only:
            subseqs[idx % kl] += ch
            idx += 1

        # Solve each Caesar
        shifts = [_solve_caesar(s) for s in subseqs]
        key = "".join(chr(ord("a") + s) for s in shifts)

        # Decrypt
        plaintext = Vigenere.decrypt(ciphertext, key)

        # Score
        scorable = "".join(
            ch for ch in plaintext if ch in string.ascii_lowercase or ch == " "
        )
        score = model.score_quadgrams_fast(
            text_to_indices(scorable, model.include_space)
        )

        if callback:
            callback(key, plaintext[:120], score)

        if score > result.best_score:
            result.best_score = score
            result.best_key = key
            result.best_plaintext = plaintext
            result.detected_key_length = kl

    return result
