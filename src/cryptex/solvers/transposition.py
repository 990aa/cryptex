"""Columnar transposition cipher cracker via MCMC.

The search space is over column permutations.  We use the same
Metropolis-Hastings / simulated-annealing framework as the substitution
solver but propose moves by swapping two column indices.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable


from cryptex.ciphers import ColumnarTransposition
from cryptex.ngram import NgramModel, text_to_indices


@dataclass
class TranspositionConfig:
    iterations: int = 25_000
    num_restarts: int = 5
    t_start: float = 50.0
    t_min: float = 0.01
    cooling_rate: float = 0.99966
    callback_interval: int = 500
    min_cols: int = 3
    max_cols: int = 12


@dataclass
class TranspositionResult:
    best_key: list[int] = field(default_factory=list)
    best_plaintext: str = ""
    best_score: float = -math.inf
    detected_ncols: int = 0
    iterations_total: int = 0


TranspositionCallback = Callable[[int, int, str, float, float], None]


def _run_chain(
    ciphertext: str,
    ncols: int,
    model: NgramModel,
    config: TranspositionConfig,
    chain_idx: int,
    callback: TranspositionCallback | None,
) -> tuple[list[int], str, float]:
    """Run a single MCMC chain for a fixed number of columns."""
    key = list(range(ncols))
    random.shuffle(key)

    plaintext = ColumnarTransposition.decrypt(ciphertext, key)
    score = model.score_quadgrams_fast(text_to_indices(plaintext, model.include_space))

    best_key = list(key)
    best_score = score
    best_pt = plaintext
    temperature = config.t_start

    for it in range(1, config.iterations + 1):
        # Propose: swap two columns
        i, j = random.sample(range(ncols), 2)
        key[i], key[j] = key[j], key[i]

        pt = ColumnarTransposition.decrypt(ciphertext, key)
        new_score = model.score_quadgrams_fast(text_to_indices(pt, model.include_space))

        delta = new_score - score
        if delta > 0 or random.random() < math.exp(delta / max(temperature, 1e-12)):
            score = new_score
            plaintext = pt
        else:
            key[i], key[j] = key[j], key[i]

        if score > best_score:
            best_score = score
            best_key = list(key)
            best_pt = plaintext

        if temperature > config.t_min:
            temperature *= config.cooling_rate

        if callback and it % config.callback_interval == 0:
            callback(chain_idx, it, plaintext[:120], score, temperature)

    return best_key, best_pt, best_score


def crack_transposition(
    ciphertext: str,
    model: NgramModel,
    config: TranspositionConfig | None = None,
    callback: TranspositionCallback | None = None,
) -> TranspositionResult:
    """Crack a columnar transposition cipher.

    Tries multiple candidate column counts and runs MCMC for each.
    """
    if config is None:
        config = TranspositionConfig()

    result = TranspositionResult()
    ct_len = len(ciphertext)

    for ncols in range(config.min_cols, config.max_cols + 1):
        # Column count must evenly divide ciphertext length (after padding)
        # We allow slight mismatch — the cipher may have padded
        if ct_len % ncols != 0:
            continue

        for restart in range(config.num_restarts):
            key, pt, score = _run_chain(
                ciphertext, ncols, model, config, restart, callback
            )
            result.iterations_total += config.iterations

            if score > result.best_score:
                result.best_score = score
                result.best_key = key
                result.best_plaintext = pt
                result.detected_ncols = ncols

    return result
