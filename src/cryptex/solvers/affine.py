"""Affine cipher solver via exhaustive key search."""

from __future__ import annotations

import math
import string
from dataclasses import dataclass
from threading import Event
from typing import Callable

from cryptex.core.ngram import NgramModel, text_to_indices


@dataclass
class AffineConfig:
    """Configuration for affine brute-force search."""

    callback_interval: int = 26


@dataclass
class AffineResult:
    best_a: int = 1
    best_b: int = 0
    best_key: str = "a=1,b=0"
    best_plaintext: str = ""
    best_score: float = -math.inf
    tested_keys: int = 0


AffineCallback = Callable[[int, int, str, float], None]


def _mod_inv(a: int, m: int) -> int | None:
    for x in range(1, m):
        if (a * x) % m == 1:
            return x
    return None


def _decrypt_affine(ciphertext: str, a: int, b: int) -> str:
    inv_a = _mod_inv(a, 26)
    if inv_a is None:
        return ciphertext
    out: list[str] = []
    for ch in ciphertext:
        if ch in string.ascii_lowercase:
            x = ord(ch) - ord("a")
            p = (inv_a * (x - b)) % 26
            out.append(chr(ord("a") + p))
        else:
            out.append(ch)
    return "".join(out)


def crack_affine(
    ciphertext: str,
    model: NgramModel,
    config: AffineConfig | None = None,
    callback: AffineCallback | None = None,
    stop_event: Event | None = None,
) -> AffineResult:
    """Solve affine cipher by scoring all valid keys."""
    if config is None:
        config = AffineConfig()

    result = AffineResult()
    valid_a = [a for a in range(26) if math.gcd(a, 26) == 1]

    trial = 0
    for a in valid_a:
        if stop_event is not None and stop_event.is_set():
            break
        for b in range(26):
            if stop_event is not None and stop_event.is_set():
                break
            trial += 1
            pt = _decrypt_affine(ciphertext, a, b)
            idx = text_to_indices(pt, include_space=model.include_space)
            score = model.score_adaptive(idx)

            if score > result.best_score:
                result.best_score = score
                result.best_a = a
                result.best_b = b
                result.best_key = f"a={a},b={b}"
                result.best_plaintext = pt

            if callback and trial % max(config.callback_interval, 1) == 0:
                callback(a, b, pt[:120], score)

    result.tested_keys = trial
    return result

