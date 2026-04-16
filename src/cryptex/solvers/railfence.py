"""Rail Fence cipher solver using rail-count search."""

from __future__ import annotations

import math
import string
from dataclasses import dataclass
from threading import Event
from typing import Callable

from cryptex.core.ngram import NgramModel, text_to_indices


@dataclass
class RailFenceConfig:
    min_rails: int = 2
    max_rails: int = 12


@dataclass
class RailFenceResult:
    best_rails: int = 2
    best_key: str = "rails=2"
    best_plaintext: str = ""
    best_score: float = -math.inf


RailFenceCallback = Callable[[int, str, float], None]


def _rail_pattern(n: int, rails: int) -> list[int]:
    if rails <= 1:
        return [0] * n
    row = 0
    step = 1
    pattern: list[int] = []
    for _ in range(n):
        pattern.append(row)
        row += step
        if row == 0 or row == rails - 1:
            step *= -1
    return pattern


def _decrypt_railfence(ciphertext: str, rails: int) -> str:
    letters_only = [ch for ch in ciphertext if ch in string.ascii_lowercase]
    n = len(letters_only)
    if n == 0:
        return ciphertext

    pattern = _rail_pattern(n, rails)
    counts = [pattern.count(r) for r in range(rails)]

    rail_slices: list[list[str]] = []
    idx = 0
    for c in counts:
        rail_slices.append(letters_only[idx : idx + c])
        idx += c

    rail_offsets = [0] * rails
    recovered: list[str] = []
    for r in pattern:
        recovered.append(rail_slices[r][rail_offsets[r]])
        rail_offsets[r] += 1

    out: list[str] = []
    rec_idx = 0
    for ch in ciphertext:
        if ch in string.ascii_lowercase:
            out.append(recovered[rec_idx])
            rec_idx += 1
        else:
            out.append(ch)
    return "".join(out)


def crack_railfence(
    ciphertext: str,
    model: NgramModel,
    config: RailFenceConfig | None = None,
    callback: RailFenceCallback | None = None,
    stop_event: Event | None = None,
) -> RailFenceResult:
    if config is None:
        config = RailFenceConfig()

    result = RailFenceResult()

    for rails in range(max(2, config.min_rails), max(2, config.max_rails) + 1):
        if stop_event is not None and stop_event.is_set():
            break
        pt = _decrypt_railfence(ciphertext, rails)
        idx = text_to_indices(pt, include_space=model.include_space)
        score = model.score_adaptive(idx)

        if callback:
            callback(rails, pt[:120], score)

        if score > result.best_score:
            result.best_score = score
            result.best_rails = rails
            result.best_key = f"rails={rails}"
            result.best_plaintext = pt

    return result

