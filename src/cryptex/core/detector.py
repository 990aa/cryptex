"""Cipher type detection — classify unknown ciphertext.

Uses statistical features (IoC, entropy, bigram stats, autocorrelation)
to predict the cipher type without any key knowledge.
"""

from __future__ import annotations

import math
import string
from collections import Counter
from dataclasses import dataclass

import numpy as np

from cryptex.core.ngram import NgramModel, get_model, text_to_indices


_DETECTOR_MODEL: NgramModel | None = None


# Expected English IoC
ENGLISH_IOC = 0.0667
# Random IoC (uniform 26 letters)
RANDOM_IOC = 1 / 26  # ≈ 0.0385


@dataclass
class CipherFeatures:
    """Statistical features extracted from ciphertext."""

    ioc: float = 0.0
    entropy: float = 0.0
    chi_squared: float = 0.0
    bigram_repeat_rate: float = 0.0
    autocorrelation_peak: int = 0
    autocorrelation_values: list[float] | None = None
    even_odd_balance: float = 0.0
    playfair_score: float = 0.0
    length: int = 0


@dataclass
class DetectionResult:
    """Result of cipher type detection."""

    predicted_type: str = ""
    confidence: float = 0.0
    features: CipherFeatures | None = None
    all_scores: dict[str, float] | None = None
    reasoning: str = ""


# Feature extraction


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


def _ioc(text: str) -> float:
    """Index of Coincidence."""
    counts = Counter(ch for ch in text if ch in string.ascii_lowercase)
    n = sum(counts.values())
    if n <= 1:
        return 0.0
    return sum(c * (c - 1) for c in counts.values()) / (n * (n - 1))


def _entropy(text: str) -> float:
    """Shannon entropy of the character distribution."""
    counts = Counter(ch for ch in text if ch in string.ascii_lowercase)
    n = sum(counts.values())
    if n == 0:
        return 0.0
    return -sum((c / n) * math.log2(c / n) for c in counts.values() if c > 0)


def _chi_squared(text: str) -> float:
    """Chi-squared statistic vs expected English frequencies."""
    counts = np.zeros(26)
    for ch in text:
        if ch in string.ascii_lowercase:
            counts[ord(ch) - ord("a")] += 1
    n = counts.sum()
    if n == 0:
        return 0.0
    observed = counts / n
    return float(((observed - ENGLISH_FREQ) ** 2 / (ENGLISH_FREQ + 1e-10)).sum())


def _bigram_repeat_rate(text: str) -> float:
    """Fraction of repeated bigrams (digraph repetition rate)."""
    letters = "".join(ch for ch in text if ch in string.ascii_lowercase)
    if len(letters) < 2:
        return 0.0
    bigrams = [letters[i : i + 2] for i in range(len(letters) - 1)]
    counts = Counter(bigrams)
    n = len(bigrams)
    return sum(c * (c - 1) for c in counts.values()) / max(n * (n - 1), 1)


def _autocorrelation(text: str, max_lag: int = 30) -> tuple[int, list[float]]:
    """Autocorrelation of letter equality at different lags.

    A peak at lag k suggests Vigenère with key length k.
    """
    letters = "".join(ch for ch in text if ch in string.ascii_lowercase)
    n = len(letters)
    values = []

    for lag in range(1, min(max_lag + 1, n)):
        matches = sum(1 for i in range(n - lag) if letters[i] == letters[i + lag])
        values.append(matches / max(n - lag, 1))

    if not values:
        return 0, []

    peak_lag = int(np.argmax(values)) + 1
    return peak_lag, values


def _even_odd_balance(text: str) -> float:
    """For Playfair detection: check if text length is even and letter-only.

    Returns 1.0 if length is even and no 'j', lower otherwise.
    """
    letters = "".join(ch for ch in text if ch in string.ascii_lowercase)
    score = 0.0
    if len(letters) % 2 == 0:
        score += 0.5
    if "j" not in letters:
        score += 0.3
    # Playfair tends to have more uniform bigram distribution
    if len(letters) >= 4:
        bigrams = [letters[i : i + 2] for i in range(0, len(letters) - 1, 2)]
        unique_ratio = len(set(bigrams)) / max(len(bigrams), 1)
        if unique_ratio > 0.85:
            score += 0.2
    return min(score, 1.0)


def _playfair_score(text: str) -> float:
    """Playfair-specific structure score in [0, 1]."""
    letters = "".join(ch for ch in text if ch in string.ascii_lowercase)
    if len(letters) < 4:
        return 0.0

    score = 0.0
    if len(letters) % 2 == 0:
        score += 0.35
    if "j" not in letters:
        score += 0.25

    digraphs = [letters[i : i + 2] for i in range(0, len(letters) - 1, 2)]
    unique_ratio = len(set(digraphs)) / max(len(digraphs), 1)
    score += 0.25 * min(1.0, unique_ratio / 0.85)

    repeated_pair_ratio = sum(1 for dg in digraphs if dg[0] == dg[1]) / max(
        len(digraphs), 1
    )
    score += 0.15 * max(0.0, 1.0 - repeated_pair_ratio / 0.05)
    return float(min(1.0, max(0.0, score)))


def _softmax_confidence(scores: dict[str, float]) -> dict[str, float]:
    """Convert raw classifier scores into calibrated probabilities."""
    if not scores:
        return {}
    labels = list(scores.keys())
    values = np.asarray([scores[k] for k in labels], dtype=np.float64)
    values -= np.max(values)
    probs = np.exp(values)
    denom = probs.sum()
    if denom <= 0:
        return {k: 0.0 for k in labels}
    probs /= denom
    return {k: float(p) for k, p in zip(labels, probs)}


def _score_per_char(text: str, model: NgramModel) -> float:
    """Average language-model score per character."""
    filtered = "".join(
        ch
        for ch in text.lower()
        if ch in string.ascii_lowercase or (model.include_space and ch == " ")
    )
    idx = text_to_indices(filtered, include_space=model.include_space)
    if len(idx) == 0:
        return -math.inf
    return float(model.score_adaptive(idx) / max(len(idx), 1))


def _is_plaintext(
    text: str,
    model: NgramModel,
    features: CipherFeatures,
    threshold: float = -2.5,
) -> bool:
    """Heuristic plaintext gate before cipher-type classification."""
    score = _score_per_char(text, model)
    return (
        score > threshold
        and 0.05 <= features.ioc <= 0.085
        and features.chi_squared < 1.2
        and features.entropy < 4.45
    )


def _get_detector_model() -> NgramModel:
    global _DETECTOR_MODEL
    if _DETECTOR_MODEL is None:
        _DETECTOR_MODEL = get_model(include_space=True)
    return _DETECTOR_MODEL


def extract_features(ciphertext: str) -> CipherFeatures:
    """Extract statistical features from ciphertext."""
    letters = "".join(ch for ch in ciphertext if ch in string.ascii_lowercase)
    autocorr_peak, autocorr_values = _autocorrelation(letters)

    return CipherFeatures(
        ioc=_ioc(letters),
        entropy=_entropy(letters),
        chi_squared=_chi_squared(letters),
        bigram_repeat_rate=_bigram_repeat_rate(letters),
        autocorrelation_peak=autocorr_peak,
        autocorrelation_values=autocorr_values,
        even_odd_balance=_even_odd_balance(letters),
        playfair_score=_playfair_score(letters),
        length=len(letters),
    )


# Rule-based classifier


def detect_cipher_type(
    ciphertext: str,
    model: NgramModel | None = None,
) -> DetectionResult:
    """Detect the type of cipher used on the given ciphertext.

    Uses a rule-based classifier with statistical features.
    More robust than a trained ML classifier for this domain since
    we know exactly what features distinguish each cipher type.

    Returns
    -------
    DetectionResult
        Predicted cipher type and confidence.
    """
    if model is None:
        try:
            model = _get_detector_model()
        except Exception:  # pragma: no cover - defensive fallback
            model = None

    features = extract_features(ciphertext)
    scores: dict[str, float] = {
        "substitution": 0.0,
        "vigenere": 0.0,
        "transposition": 0.0,
        "playfair": 0.0,
    }
    reasoning_parts: list[str] = []

    ioc = features.ioc
    entropy = features.entropy

    # --- IoC Analysis ---
    # Substitution preserves English IoC (~0.067)
    # Vigenère flattens IoC toward random (~0.038) for long keys
    # Transposition preserves IoC perfectly
    # Playfair sits between substitution and random

    if ioc > 0.060:
        # High IoC → substitution or transposition
        scores["substitution"] += 3.0
        scores["transposition"] += 3.0
        reasoning_parts.append(
            f"High IoC ({ioc:.4f}) suggests mono-alphabetic or transposition"
        )
    elif ioc > 0.045:
        # Medium IoC → Playfair or short-key Vigenère
        scores["playfair"] += 2.0
        scores["vigenere"] += 1.5
        reasoning_parts.append(
            f"Medium IoC ({ioc:.4f}) suggests Playfair or short Vigenère"
        )
    else:
        # Low IoC → Vigenère with longer key
        scores["vigenere"] += 3.0
        reasoning_parts.append(f"Low IoC ({ioc:.4f}) suggests Vigenère")

    # --- Entropy Analysis ---
    # Max entropy for 26 letters = log2(26) ≈ 4.70
    # English ≈ 4.0-4.2
    # Random ≈ 4.65-4.70
    if entropy > 4.5:
        scores["vigenere"] += 1.0
        reasoning_parts.append(f"High entropy ({entropy:.2f}) suggests polyalphabetic")
    elif entropy < 4.2:
        scores["substitution"] += 1.0
        scores["transposition"] += 0.5
        reasoning_parts.append(
            f"Low entropy ({entropy:.2f}) suggests preserved letter distribution"
        )

    # --- Autocorrelation (Vigenère detection) ---
    if features.autocorrelation_values:
        values = features.autocorrelation_values
        mean_ac = sum(values) / len(values) if values else 0
        if len(values) >= 3:
            # Check for periodic peaks
            max_val = max(values)
            if max_val > mean_ac * 1.3 and ioc < 0.060:
                scores["vigenere"] += 2.5
                reasoning_parts.append(
                    f"Autocorrelation peak at lag {features.autocorrelation_peak} "
                    f"({max_val:.4f} vs mean {mean_ac:.4f})"
                )

    # --- Spaces presence (distinguishes transposition) ---
    has_spaces = " " in ciphertext
    letters_only = "".join(ch for ch in ciphertext if ch in string.ascii_lowercase)

    if has_spaces and ioc > 0.055:
        # Transposition often preserves spaces (or scrambles them)
        # But substitution also has spaces
        scores["transposition"] += 1.0
        reasoning_parts.append("Spaces present with high IoC")

    # --- Playfair indicators ---
    playfair_shape = features.playfair_score
    if playfair_shape > 0.55:
        scores["playfair"] += 2.5 * playfair_shape
        reasoning_parts.append(
            f"Playfair structural score {playfair_shape:.2f} (even digraph behavior)"
        )
    elif features.even_odd_balance > 0.7:
        scores["playfair"] += 1.5
        reasoning_parts.append("Even length, no 'j' — weak Playfair indicators")

    # No spaces + even length + medium IoC → strong Playfair
    if not has_spaces and len(letters_only) % 2 == 0 and 0.040 < ioc < 0.060:
        scores["playfair"] += 2.0

    # --- Chi-squared test ---
    # Low chi-squared = close to English frequencies (substitution/transposition)
    if features.chi_squared < 0.5:
        scores["transposition"] += 2.0
        reasoning_parts.append(
            f"Low chi² ({features.chi_squared:.3f}) — letter frequencies match English"
        )
    elif features.chi_squared < 2.0:
        scores["substitution"] += 1.0

    # --- Bigram repeat rate ---
    # Transposition preserves bigram patterns somewhat
    if features.bigram_repeat_rate > 0.001:
        scores["transposition"] += 0.5
        scores["substitution"] += 0.5

    if model is not None and _is_plaintext(ciphertext, model, features):
        scores["plaintext"] = max(scores.values(), default=0.0) + 4.0
        reasoning_parts.insert(0, "Language model strongly matches plaintext")

    probs = _softmax_confidence(scores)
    predicted = max(probs, key=lambda k: probs[k]) if probs else ""
    confidence = probs.get(predicted, 0.0)

    return DetectionResult(
        predicted_type=predicted,
        confidence=confidence,
        features=features,
        all_scores=probs,
        reasoning="; ".join(reasoning_parts),
    )
