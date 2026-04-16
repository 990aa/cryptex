"""MCMC solver for substitution ciphers.

Uses adaptive simulated annealing with optional parallel tempering
(replica exchange) and occasional evolutionary jump proposals.
"""

from __future__ import annotations

import math
import random
import string
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from cipher.ngram import NgramModel, text_to_indices


# Configuration


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

    use_parallel_tempering: bool = True
    """Enable replica exchange with multiple temperatures per restart."""

    num_replicas: int = 6
    """Number of temperature replicas per restart when tempering is enabled."""

    swap_interval: int = 50
    """How often to attempt adjacent-replica exchanges."""

    adapt_interval: int = 250
    """How often to adapt the temperature scale to hit target acceptance."""

    target_acceptance: float = 0.25
    """Target acceptance for adaptive annealing."""

    adaptation_rate: float = 0.08
    """Strength of adaptation updates."""

    crossover_jump_prob: float = 0.08
    """Probability of OX1 crossover jump proposal instead of simple swap."""

    top_k_candidates: int = 5
    """How many top candidates to retain for short ciphertexts."""

    random_seed: int | None = None
    """Optional deterministic seed for repeatable solver runs."""


# Result type


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
    top_candidates: list[tuple[str, str, float]] = field(default_factory=list)
    swap_acceptance_rate: float = 0.0


# Callback signature

# callback(chain_idx, iteration, current_key, current_plaintext, current_score, best_score, temperature)
ProgressCallback = Callable[[int, int, str, str, float, float, float], None]


# Core solver


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


def _random_perm() -> np.ndarray:
    p = np.arange(26, dtype=np.int16)
    np.random.shuffle(p)
    return p


def _perm_to_key(perm: np.ndarray) -> str:
    return "".join(chr(ord("a") + int(v)) for v in perm)


def _build_inv_map(perm: np.ndarray, include_space: bool, A: int) -> np.ndarray:
    """Build inv[cipher_idx] -> plain_idx for current permutation."""
    offset = 1 if include_space else 0
    inv = np.zeros(A, dtype=np.int16)
    if include_space:
        inv[0] = 0
    for plain_i in range(26):
        cipher_i = int(perm[plain_i]) + offset
        inv[cipher_i] = plain_i + offset
    return inv


def _decode_with_inv(ct_idx: np.ndarray, inv: np.ndarray) -> np.ndarray:
    return inv[ct_idx].astype(np.int16, copy=False)


def _idx_to_text(plain_idx: np.ndarray, include_space: bool) -> str:
    offset = 1 if include_space else 0
    chars: list[str] = []
    for v in plain_idx:
        if include_space and v == 0:
            chars.append(" ")
        else:
            chars.append(chr(ord("a") + int(v) - offset))
    return "".join(chars)


def _order_crossover_perm(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Order crossover (OX1) for permutation proposals."""
    n = len(p1)
    start, end = sorted(random.sample(range(n), 2))
    child = np.full(n, -1, dtype=np.int16)
    child[start:end] = p1[start:end]
    used = set(int(v) for v in child[start:end])
    fill = [int(v) for v in p2 if int(v) not in used]
    fill_i = 0
    for i in range(n):
        if child[i] == -1:
            child[i] = fill[fill_i]
            fill_i += 1
    return child


def _temperature_ladder(config: MCMCConfig) -> np.ndarray:
    n = max(1, int(config.num_replicas))
    hi = max(config.t_start, config.t_min + 1e-9)
    lo = max(min(config.t_min, hi), 1e-6)
    if n == 1:
        return np.asarray([hi], dtype=np.float64)
    return np.geomspace(hi, lo, n).astype(np.float64)


def _push_top_candidate(
    result: MCMCResult,
    key: str,
    plaintext: str,
    score: float,
    k: int,
) -> None:
    """Maintain top-k unique plaintext candidates by score."""
    for i, (k_old, pt_old, s_old) in enumerate(result.top_candidates):
        if pt_old == plaintext:
            if score > s_old:
                result.top_candidates[i] = (key, plaintext, score)
            break
    else:
        result.top_candidates.append((key, plaintext, score))

    result.top_candidates.sort(key=lambda x: x[2], reverse=True)
    if len(result.top_candidates) > k:
        del result.top_candidates[k:]


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

    if config.random_seed is not None:
        random.seed(config.random_seed)
        np.random.seed(config.random_seed)

    result = MCMCResult()

    # Preserve only model-supported symbols (letters + optional spaces).
    ct_idx = text_to_indices(ciphertext, model.include_space)
    if len(ct_idx) == 0:
        result.best_key = string.ascii_lowercase
        result.best_plaintext = ""
        return result

    threshold_reached = False
    global_no_improve = 0
    total_swap_attempts = 0
    total_swap_accepts = 0

    include_space = model.include_space
    A = model.A

    best_perm_global: np.ndarray | None = None

    for chain_idx in range(config.num_restarts):
        if threshold_reached:
            break

        n_replicas = (
            max(1, int(config.num_replicas)) if config.use_parallel_tempering else 1
        )
        temperatures = _temperature_ladder(config)[:n_replicas]

        replicas: list[dict[str, object]] = []
        for ridx in range(n_replicas):
            perm = _random_perm()
            inv = _build_inv_map(perm, include_space, A)
            score = model.score_quadgrams_with_mapping(ct_idx, inv)

            replicas.append(
                {
                    "perm": perm,
                    "inv": inv,
                    "score": score,
                    "best_perm": perm.copy(),
                    "best_score": score,
                    "accepts": 0,
                    "proposals": 0,
                    "accepts_window": 0,
                    "proposals_window": 0,
                }
            )

            if score > result.best_score:
                result.best_score = float(score)
                best_perm_global = perm.copy()
                result.best_key = _perm_to_key(perm)
                pt_now = _idx_to_text(_decode_with_inv(ct_idx, inv), include_space)
                result.best_plaintext = pt_now
                _push_top_candidate(
                    result,
                    result.best_key,
                    pt_now,
                    result.best_score,
                    config.top_k_candidates,
                )

        chain_best_score = max(float(r["best_score"]) for r in replicas)
        chain_best_perm = np.array(replicas[0]["best_perm"], copy=True)

        for it in range(1, config.iterations + 1):
            if threshold_reached:
                break

            # Metropolis updates in each replica.
            for ridx, replica in enumerate(replicas):
                temp = float(temperatures[ridx])
                perm = np.array(replica["perm"], copy=False)
                current_score = float(replica["score"])

                use_crossover = (
                    best_perm_global is not None
                    and random.random() < config.crossover_jump_prob
                )

                if use_crossover:
                    proposal_perm = _order_crossover_perm(perm, best_perm_global)
                else:
                    proposal_perm = perm.copy()
                    i, j = random.sample(range(26), 2)
                    proposal_perm[i], proposal_perm[j] = (
                        proposal_perm[j],
                        proposal_perm[i],
                    )

                proposal_inv = _build_inv_map(proposal_perm, include_space, A)
                proposal_score = model.score_quadgrams_with_mapping(
                    ct_idx, proposal_inv
                )

                delta = float(proposal_score - current_score)
                accept = delta >= 0.0 or random.random() < math.exp(
                    delta / max(temp, 1e-12)
                )

                replica["proposals"] = int(replica["proposals"]) + 1
                replica["proposals_window"] = int(replica["proposals_window"]) + 1

                if accept:
                    replica["perm"] = proposal_perm
                    replica["inv"] = proposal_inv
                    replica["score"] = float(proposal_score)
                    replica["accepts"] = int(replica["accepts"]) + 1
                    replica["accepts_window"] = int(replica["accepts_window"]) + 1
                    current_score = float(proposal_score)

                if current_score > float(replica["best_score"]):
                    replica["best_score"] = current_score
                    replica["best_perm"] = np.array(replica["perm"], copy=True)

                if current_score > chain_best_score:
                    chain_best_score = current_score
                    chain_best_perm = np.array(replica["perm"], copy=True)
                    global_no_improve = 0
                else:
                    global_no_improve += 1

                if current_score > result.best_score:
                    result.best_score = current_score
                    best_perm_global = np.array(replica["perm"], copy=True)
                    result.best_key = _perm_to_key(best_perm_global)
                    pt_best = _idx_to_text(
                        _decode_with_inv(ct_idx, np.array(replica["inv"], copy=False)),
                        include_space,
                    )
                    result.best_plaintext = pt_best
                    _push_top_candidate(
                        result,
                        result.best_key,
                        pt_best,
                        result.best_score,
                        config.top_k_candidates,
                    )

            # Replica exchange (parallel tempering).
            if n_replicas > 1 and it % max(config.swap_interval, 1) == 0:
                parity = (it // max(config.swap_interval, 1)) % 2
                start = int(parity)
                for ridx in range(start, n_replicas - 1, 2):
                    r1 = replicas[ridx]
                    r2 = replicas[ridx + 1]

                    t1 = float(temperatures[ridx])
                    t2 = float(temperatures[ridx + 1])
                    s1 = float(r1["score"])
                    s2 = float(r2["score"])

                    total_swap_attempts += 1
                    swap_log_alpha = (1.0 / max(t1, 1e-12) - 1.0 / max(t2, 1e-12)) * (
                        s2 - s1
                    )
                    if math.log(max(random.random(), 1e-12)) < swap_log_alpha:
                        total_swap_accepts += 1
                        r1["perm"], r2["perm"] = r2["perm"], r1["perm"]
                        r1["inv"], r2["inv"] = r2["inv"], r1["inv"]
                        r1["score"], r2["score"] = r2["score"], r1["score"]

            # Adaptive annealing + baseline cooling.
            if it % max(config.adapt_interval, 1) == 0:
                window_rates = []
                for replica in replicas:
                    pw = int(replica["proposals_window"])
                    aw = int(replica["accepts_window"])
                    rate = aw / max(pw, 1)
                    window_rates.append(rate)
                    replica["proposals_window"] = 0
                    replica["accepts_window"] = 0

                mean_rate = float(np.mean(window_rates)) if window_rates else 0.0
                adapt_scale = math.exp(
                    config.adaptation_rate * (mean_rate - config.target_acceptance)
                )
                temperatures = np.clip(
                    temperatures * adapt_scale,
                    config.t_min,
                    max(config.t_start * 4.0, config.t_min),
                )

            temperatures = np.maximum(temperatures * config.cooling_rate, config.t_min)

            # Periodic logging and callback from the coldest replica.
            if it % config.callback_interval == 0:
                cold_idx = int(np.argmin(temperatures))
                cold = replicas[cold_idx]
                cold_perm = np.array(cold["perm"], copy=False)
                cold_inv = np.array(cold["inv"], copy=False)
                cold_pt = _idx_to_text(
                    _decode_with_inv(ct_idx, cold_inv), include_space
                )
                cold_score = float(cold["score"])

                if config.track_trajectory:
                    result.score_trajectory.append(
                        (chain_idx * config.iterations + it, result.best_score)
                    )

                acc_values = [
                    int(rep["accepts"]) / max(int(rep["proposals"]), 1)
                    for rep in replicas
                ]
                result.acceptance_rates.append(float(np.mean(acc_values)))

                if callback:
                    callback(
                        chain_idx,
                        it,
                        _perm_to_key(cold_perm),
                        cold_pt,
                        cold_score,
                        result.best_score,
                        float(temperatures[cold_idx]),
                    )

            if (
                config.early_stop_patience
                and global_no_improve >= config.early_stop_patience
            ):
                result.early_stopped = True
                break

            if (
                config.score_threshold is not None
                and result.best_score >= config.score_threshold
            ):
                threshold_reached = True
                break

        result.chain_scores.append(chain_best_score)
        result.chain_keys.append(_perm_to_key(chain_best_perm))
        result.iterations_total += config.iterations

    if best_perm_global is None:
        best_perm_global = np.arange(26, dtype=np.int16)
        result.best_key = _perm_to_key(best_perm_global)

    best_inv = _build_inv_map(best_perm_global, include_space, A)
    best_plain = _decode_with_inv(ct_idx, best_inv)
    result.best_plaintext = _idx_to_text(best_plain, include_space)

    if len(ct_idx) < 30 and not result.top_candidates:
        _push_top_candidate(
            result,
            result.best_key,
            result.best_plaintext,
            result.best_score,
            config.top_k_candidates,
        )

    if total_swap_attempts > 0:
        result.swap_acceptance_rate = total_swap_accepts / total_swap_attempts
    else:
        result.swap_acceptance_rate = 0.0

    return result
