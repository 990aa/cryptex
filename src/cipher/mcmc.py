"""MCMC Metropolis-Hastings solver for substitution ciphers.

Implements simulated annealing with configurable cooling schedule
and multiple random restarts.  Exposes a callback-based interface
so the CLI can display real-time progress.
"""

from __future__ import annotations

import math
import random
import string
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from cipher.ngram import NgramModel, text_to_indices


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------


@dataclass
class MCMCConfig:
    """Knobs for the MCMC solver."""

    iterations: int = 50_000
    """Iterations per chain."""

    num_restarts: int = 8
    """Number of independent random-restart chains."""

    t_start: float = 50.0
    """Starting temperature (must match score-magnitude scale)."""

    t_min: float = 0.008
    """Minimum temperature (stop cooling here)."""

    cooling_rate: float = 0.99983
    """Geometric cooling factor — cools from 50 → ~0.01 over 50k iters."""

    callback_interval: int = 500
    """How often (in iterations) to call the progress callback."""

    early_stop_patience: int = 15_000
    """Stop a chain if no improvement after this many iterations."""

    score_threshold: float | None = None
    """If set, stop all chains once this score is reached."""

    track_trajectory: bool = True
    """Whether to log the best score at each callback interval."""

    acceptance_window: int = 1000
    """Window size for computing recent acceptance rate."""


# ------------------------------------------------------------------
# Result type
# ------------------------------------------------------------------


@dataclass
class MCMCResult:
    best_key: str = ""
    best_plaintext: str = ""
    best_score: float = -math.inf
    iterations_total: int = 0
    chain_scores: list[float] = field(default_factory=list)
    chain_keys: list[str] = field(default_factory=list)
    score_trajectory: list[tuple[int, float]] = field(default_factory=list)
    acceptance_rates: list[float] = field(default_factory=list)
    early_stopped: bool = False


# ------------------------------------------------------------------
# Callback signature
# ------------------------------------------------------------------
# callback(chain_idx, iteration, current_key, current_plaintext, current_score, best_score, temperature)
ProgressCallback = Callable[[int, int, str, str, float, float, float], None]


# ------------------------------------------------------------------
# Core solver
# ------------------------------------------------------------------


def _apply_key(ciphertext: str, key: str) -> str:
    """Decrypt *ciphertext* with the given substitution key."""
    table = str.maketrans(key, string.ascii_lowercase)
    return ciphertext.translate(table)


def _swap_key(key: list[str]) -> tuple[int, int]:
    """Swap two random positions in the key (in-place) and return the indices."""
    i, j = random.sample(range(26), 2)
    key[i], key[j] = key[j], key[i]
    return i, j


def _random_key() -> list[str]:
    k = list(string.ascii_lowercase)
    random.shuffle(k)
    return k


def run_mcmc(
    ciphertext: str,
    model: NgramModel,
    config: MCMCConfig | None = None,
    callback: ProgressCallback | None = None,
) -> MCMCResult:
    """Run multiple MCMC chains and return the best decryption found.

    Parameters
    ----------
    ciphertext : str
        The ciphertext to crack (lowercase letters + optional spaces).
    model : NgramModel
        Trained language model used for scoring.
    config : MCMCConfig, optional
        Solver parameters.
    callback : callable, optional
        Called every ``config.callback_interval`` iterations with progress info.

    Returns
    -------
    MCMCResult
    """
    if config is None:
        config = MCMCConfig()

    result = MCMCResult()

    # Pre-strip ciphertext to scoreable characters
    ct_alpha = "".join(
        ch for ch in ciphertext if ch in string.ascii_lowercase or ch == " "
    )
    ct_idx = text_to_indices(ct_alpha, model.include_space)  # ciphertext as indices

    global_iter = 0
    threshold_reached = False

    for chain_idx in range(config.num_restarts):
        if threshold_reached:
            break

        key = _random_key()

        # Build a full inverse-mapping array for fast re-indexing
        # inv[cipher_idx] → plain_idx
        include_space = model.include_space
        offset = 1 if include_space else 0
        A = model.A

        def _build_plain_idx(k: list[str]) -> np.ndarray:
            inv = np.zeros(A, dtype=np.int8)
            if include_space:
                inv[0] = 0  # space → space
            for ci in range(26):
                cipher_char = k[ci]
                cipher_i = ord(cipher_char) - ord("a") + offset
                plain_i = ci + offset
                inv[cipher_i] = plain_i
            return inv

        inv = _build_plain_idx(key)
        plain_idx = inv[ct_idx]  # fast vectorised decrypt

        current_score = model.score_quadgrams_fast(plain_idx)

        best_chain_score = current_score
        best_chain_key = list(key)
        best_chain_plain: np.ndarray = plain_idx.copy()

        temperature = config.t_start
        no_improve_count = 0
        accept_count = 0
        total_proposals = 0

        for it in range(1, config.iterations + 1):
            # --- Propose: swap two letters in the key ---
            i, j = random.sample(range(26), 2)
            key[i], key[j] = key[j], key[i]

            # Update inverse mapping for the two swapped letters
            ci_i = ord(key[i]) - ord("a") + offset
            ci_j = ord(key[j]) - ord("a") + offset
            pi_i = i + offset
            pi_j = j + offset
            inv[ci_i] = pi_i
            inv[ci_j] = pi_j

            # Re-derive plaintext indices (vectorised)
            proposed_plain_idx = inv[ct_idx]
            proposed_score = model.score_quadgrams_fast(proposed_plain_idx)

            # --- Accept / Reject ---
            delta = proposed_score - current_score
            total_proposals += 1
            if delta > 0 or random.random() < math.exp(delta / max(temperature, 1e-12)):
                # Accept
                current_score = proposed_score
                plain_idx = proposed_plain_idx
                accept_count += 1
            else:
                # Reject — undo swap
                key[i], key[j] = key[j], key[i]
                inv[ci_i] = pi_j
                inv[ci_j] = pi_i

            # Track local best
            if current_score > best_chain_score:
                best_chain_score = current_score
                best_chain_key = list(key)
                best_chain_plain = plain_idx.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1

            # Track global best
            if current_score > result.best_score:
                result.best_score = current_score
                result.best_key = "".join(key)

            # Score trajectory
            if config.track_trajectory and it % config.callback_interval == 0:
                global_iter += 1
                result.score_trajectory.append(
                    (chain_idx * config.iterations + it, result.best_score)
                )
                # Acceptance rate over recent window
                window = min(total_proposals, config.acceptance_window)
                rate = accept_count / max(total_proposals, 1)
                result.acceptance_rates.append(rate)

            # Early stopping: no improvement
            if config.early_stop_patience and no_improve_count >= config.early_stop_patience:
                break

            # Early stopping: threshold reached
            if config.score_threshold is not None and result.best_score >= config.score_threshold:
                threshold_reached = True
                break

            # Cool
            if temperature > config.t_min:
                temperature *= config.cooling_rate

            # Callback
            if callback and it % config.callback_interval == 0:
                # Decode plaintext for display
                chars: list[str] = []
                for v in plain_idx:
                    if include_space and v == 0:
                        chars.append(" ")
                    else:
                        chars.append(chr(ord("a") + v - offset))
                pt_str = "".join(chars)
                callback(
                    chain_idx,
                    it,
                    "".join(key),
                    pt_str,
                    current_score,
                    result.best_score,
                    temperature,
                )

        result.chain_scores.append(best_chain_score)
        result.chain_keys.append("".join(best_chain_key))
        result.iterations_total += config.iterations

    # Decode final best plaintext
    inv_best = _build_plain_idx(list(result.best_key))
    best_plain = inv_best[ct_idx]
    offset = 1 if model.include_space else 0
    chars = []
    for v in best_plain:
        if model.include_space and v == 0:
            chars.append(" ")
        else:
            chars.append(chr(ord("a") + v - offset))
    result.best_plaintext = "".join(chars)

    return result
