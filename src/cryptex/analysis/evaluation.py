"""Evaluation framework for cipher cracking.

Provides:
- Symbol Error Rate (SER) metric
- Phase transition analysis (success rate vs ciphertext length)
- Benchmarking against baselines (frequency analysis, genetic algorithm, hill-climb)
"""

from __future__ import annotations

import json
import math
import random
import string
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from cryptex.ciphers import SimpleSubstitution
from cryptex.core.ngram import NgramModel, text_to_indices


ENGLISH_FREQ_RANK = [
    4,
    19,
    0,
    14,
    8,
    13,
    18,
    7,
    17,
    3,
    11,
    2,
    20,
    12,
    22,
    5,
    6,
    24,
    15,
    1,
    21,
    10,
    9,
    23,
    16,
    25,
]


def _write_json_report(
    payload: dict[str, object],
    save_path: str | Path | None,
    default_name: str,
) -> Path:
    path = Path(save_path) if save_path is not None else Path(default_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


# Symbol Error Rate


def symbol_error_rate(plaintext: str, decrypted: str) -> float:
    """Compute Symbol Error Rate: fraction of positions incorrectly decrypted.

    Only compares alphabetic characters (ignores spaces/punctuation).
    """
    errors = 0
    total = 0
    for p, d in zip(plaintext, decrypted):
        if p in string.ascii_lowercase:
            total += 1
            if p != d:
                errors += 1
    return errors / max(total, 1)


def key_accuracy(true_key: str, found_key: str) -> float:
    """Fraction of key positions that are correct."""
    if not true_key or not found_key:
        return 0.0
    correct = sum(1 for a, b in zip(true_key, found_key) if a == b)
    return correct / max(len(true_key), 1)


# Phase Transition Analysis


@dataclass
class PhaseTransitionResult:
    """Results of a phase transition experiment."""

    lengths: list[int] = field(default_factory=list)
    success_rates: list[float] = field(default_factory=list)
    avg_ser: list[float] = field(default_factory=list)
    avg_times: list[float] = field(default_factory=list)
    threshold_length: int = 0


def run_phase_transition(
    model: NgramModel,
    corpus: str,
    lengths: list[int] | None = None,
    trials_per_length: int = 5,
    success_threshold: float = 0.05,
    callback=None,
) -> PhaseTransitionResult:
    """Empirically determine the phase transition in ciphertext length.

    For each length, encrypt random excerpts with random keys and measure
    the success rate of MCMC decryption.

    Parameters
    ----------
    model : NgramModel
        Trained language model.
    corpus : str
        Source text to draw excerpts from.
    lengths : list of int, optional
        Ciphertext lengths to test. Defaults to [50, 75, 100, 150, 200, 300, 500].
    trials_per_length : int
        Number of trials per length.
    success_threshold : float
        SER below which a trial is considered "successful".
    callback : callable, optional
        callback(length, trial, ser, elapsed)
    """
    from cryptex.solvers.mcmc import MCMCConfig, run_mcmc

    if lengths is None:
        lengths = [50, 75, 100, 150, 200, 300, 500]

    result = PhaseTransitionResult()
    result.lengths = lengths

    # Use a fast config for benchmarking
    config = MCMCConfig(
        iterations=30_000,
        num_restarts=4,
        t_start=50.0,
        cooling_rate=0.99980,
        track_trajectory=False,
    )

    corpus_clean = "".join(
        ch for ch in corpus.lower() if ch in string.ascii_lowercase or ch == " "
    )

    for length in lengths:
        successes = 0
        total_ser = 0.0
        total_time = 0.0

        for trial in range(trials_per_length):
            # Extract random excerpt
            max_start = len(corpus_clean) - length
            if max_start <= 0:
                continue
            start = random.randint(0, max_start)
            excerpt = corpus_clean[start : start + length]

            # Encrypt
            key = SimpleSubstitution.random_key()
            ciphertext = SimpleSubstitution.encrypt(excerpt, key)

            # Crack
            t0 = time.time()
            mcmc_result = run_mcmc(ciphertext, model, config)
            elapsed = time.time() - t0

            ser = symbol_error_rate(excerpt, mcmc_result.best_plaintext)
            total_ser += ser
            total_time += elapsed

            if ser < success_threshold:
                successes += 1

            if callback:
                callback(length, trial, ser, elapsed)

        n_trials = trials_per_length
        result.success_rates.append(successes / n_trials)
        result.avg_ser.append(total_ser / n_trials)
        result.avg_times.append(total_time / n_trials)

    # Find threshold (first length with >50% success)
    for i, rate in enumerate(result.success_rates):
        if rate >= 0.5:
            result.threshold_length = result.lengths[i]
            break

    return result


def plot_phase_transition(
    pt_result: PhaseTransitionResult,
    save_path: str | Path | None = None,
) -> Path | None:
    """Write phase-transition metrics as JSON."""
    payload: dict[str, object] = {
        "lengths": [int(v) for v in pt_result.lengths],
        "success_rates": [float(v) for v in pt_result.success_rates],
        "avg_ser": [float(v) for v in pt_result.avg_ser],
        "avg_times": [float(v) for v in pt_result.avg_times],
        "threshold_length": int(pt_result.threshold_length),
    }
    return _write_json_report(payload, save_path, "phase_transition.json")


# Benchmarking Against Baselines


@dataclass
class BenchmarkEntry:
    method: str = ""
    avg_ser: float = 0.0
    avg_score: float = 0.0
    avg_time: float = 0.0
    success_rate: float = 0.0
    trials: int = 0


def _frequency_analysis_baseline(ciphertext: str) -> str:
    """Pure frequency analysis: map most common cipher letters to most common English letters."""
    english_order = "etaoinshrdlcumwfgypbvkjxqz"

    # Count cipher letter frequencies
    counts = {}
    for ch in ciphertext:
        if ch in string.ascii_lowercase:
            counts[ch] = counts.get(ch, 0) + 1

    cipher_order = sorted(counts.keys(), key=lambda c: counts.get(c, 0), reverse=True)

    # Build mapping
    mapping = {}
    for i, cipher_char in enumerate(cipher_order):
        if i < len(english_order):
            mapping[cipher_char] = english_order[i]
        else:
            mapping[cipher_char] = cipher_char

    # Decrypt
    return "".join(mapping.get(ch, ch) for ch in ciphertext)


def _hillclimb_frequency_baseline(
    ciphertext: str,
    model: NgramModel,
    iterations: int = 6000,
) -> str:
    """Literature-style baseline: frequency init + greedy swap hill-climbing."""
    ct_idx = text_to_indices(
        "".join(ch for ch in ciphertext if ch in string.ascii_lowercase or ch == " "),
        model.include_space,
    )
    if len(ct_idx) == 0:
        return ""

    offset = 1 if model.include_space else 0
    A = model.A

    counts = np.bincount(ct_idx, minlength=A).astype(np.float64)
    if model.include_space:
        counts[0] = 0.0
    cipher_rank = np.argsort(-counts[offset : offset + 26])

    perm = np.arange(26, dtype=np.int16)
    for rank_pos, cipher_letter in enumerate(cipher_rank):
        perm[ENGLISH_FREQ_RANK[rank_pos]] = int(cipher_letter)

    def _build_inv_map(cur_perm: np.ndarray) -> np.ndarray:
        inv = np.zeros(A, dtype=np.int16)
        if model.include_space:
            inv[0] = 0
        for plain_i in range(26):
            cipher_i = int(cur_perm[plain_i]) + offset
            inv[cipher_i] = plain_i + offset
        return inv

    inv = _build_inv_map(perm)
    best_score = model.score_quadgrams_with_mapping(ct_idx, inv)

    for _ in range(max(1, iterations)):
        i, j = random.sample(range(26), 2)
        perm[i], perm[j] = perm[j], perm[i]
        trial_inv = _build_inv_map(perm)
        trial_score = model.score_quadgrams_with_mapping(ct_idx, trial_inv)

        if trial_score > best_score:
            inv = trial_inv
            best_score = trial_score
        else:
            perm[i], perm[j] = perm[j], perm[i]

    plain_idx = inv[ct_idx]
    chars = []
    for v in plain_idx:
        if model.include_space and v == 0:
            chars.append(" ")
        else:
            chars.append(chr(ord("a") + v - offset))
    return "".join(chars)


def run_benchmark(
    model: NgramModel,
    corpus: str,
    text_length: int = 300,
    trials: int = 5,
    success_threshold: float = 0.20,
    callback=None,
) -> list[BenchmarkEntry]:
    """Benchmark MCMC vs Genetic Algorithm vs Frequency Analysis vs hill-climb.

    Parameters
    ----------
    model : NgramModel
    corpus : str
        Source text for generating test cases.
    text_length : int
        Length of each test ciphertext.
    trials : int
        Number of trials per method.
    success_threshold : float
        A trial counts as success when SER <= success_threshold.
    callback : callable, optional
        callback(method_name, trial, ser, elapsed)

    Returns
    -------
    list of BenchmarkEntry
    """
    from cryptex.solvers.genetic import GeneticConfig, run_genetic
    from cryptex.solvers.mcmc import MCMCConfig, run_mcmc

    corpus_clean = "".join(
        ch for ch in corpus.lower() if ch in string.ascii_lowercase or ch == " "
    )

    # Generate test cases
    test_cases: list[tuple[str, str, str]] = []  # (plaintext, ciphertext, key)
    for _ in range(trials):
        max_start = len(corpus_clean) - text_length
        start = random.randint(0, max(max_start, 0))
        excerpt = corpus_clean[start : start + text_length]
        key = SimpleSubstitution.random_key()
        ct = SimpleSubstitution.encrypt(excerpt, key)
        test_cases.append((excerpt, ct, key))

    methods: dict[str, BenchmarkEntry] = {
        "MCMC": BenchmarkEntry(method="MCMC", trials=trials),
        "Genetic": BenchmarkEntry(method="Genetic Algorithm", trials=trials),
        "Frequency": BenchmarkEntry(method="Frequency Analysis", trials=trials),
        "Hillclimb": BenchmarkEntry(method="Hill Climb + Freq Init", trials=trials),
    }

    mcmc_config = MCMCConfig(
        iterations=30_000, num_restarts=4, track_trajectory=False, cooling_rate=0.99980
    )
    ga_config = GeneticConfig(population_size=300, generations=500)

    for trial_idx, (plaintext, ciphertext, true_key) in enumerate(test_cases):
        # --- MCMC ---
        t0 = time.time()
        mcmc_result = run_mcmc(ciphertext, model, mcmc_config)
        elapsed = time.time() - t0
        ser = symbol_error_rate(plaintext, mcmc_result.best_plaintext)
        methods["MCMC"].avg_ser += ser
        methods["MCMC"].avg_score += mcmc_result.best_score
        methods["MCMC"].avg_time += elapsed
        if ser <= success_threshold:
            methods["MCMC"].success_rate += 1
        if callback:
            callback("MCMC", trial_idx, ser, elapsed)

        # --- Genetic ---
        t0 = time.time()
        ga_result = run_genetic(ciphertext, model, ga_config)
        elapsed = time.time() - t0
        ser = symbol_error_rate(plaintext, ga_result.best_plaintext)
        methods["Genetic"].avg_ser += ser
        methods["Genetic"].avg_score += ga_result.best_score
        methods["Genetic"].avg_time += elapsed
        if ser <= success_threshold:
            methods["Genetic"].success_rate += 1
        if callback:
            callback("Genetic", trial_idx, ser, elapsed)

        # --- Frequency Analysis ---
        t0 = time.time()
        freq_result = _frequency_analysis_baseline(ciphertext)
        elapsed = time.time() - t0
        ser = symbol_error_rate(plaintext, freq_result)
        freq_score = model.score_quadgrams_fast(
            text_to_indices(
                "".join(
                    ch
                    for ch in freq_result
                    if ch in string.ascii_lowercase or ch == " "
                ),
                model.include_space,
            )
        )
        methods["Frequency"].avg_ser += ser
        methods["Frequency"].avg_score += freq_score
        methods["Frequency"].avg_time += elapsed
        if ser <= success_threshold:
            methods["Frequency"].success_rate += 1
        if callback:
            callback("Frequency", trial_idx, ser, elapsed)

        # --- Hill Climb + Frequency Init ---
        t0 = time.time()
        hill_result = _hillclimb_frequency_baseline(
            ciphertext,
            model,
            iterations=6000,
        )
        elapsed = time.time() - t0
        ser = symbol_error_rate(plaintext, hill_result)
        hill_score = model.score_quadgrams_fast(
            text_to_indices(
                "".join(
                    ch
                    for ch in hill_result
                    if ch in string.ascii_lowercase or ch == " "
                ),
                model.include_space,
            )
        )
        methods["Hillclimb"].avg_ser += ser
        methods["Hillclimb"].avg_score += hill_score
        methods["Hillclimb"].avg_time += elapsed
        if ser <= success_threshold:
            methods["Hillclimb"].success_rate += 1
        if callback:
            callback("Hill Climb + Freq Init", trial_idx, ser, elapsed)

    # Average
    results: list[BenchmarkEntry] = []
    n_trials = max(len(test_cases), 1)
    for entry in methods.values():
        entry.avg_ser /= n_trials
        entry.avg_score /= n_trials
        entry.avg_time /= n_trials
        entry.success_rate /= n_trials
        results.append(entry)

    return results


def plot_benchmark(
    entries: list[BenchmarkEntry],
    save_path: str | Path | None = None,
    success_threshold: float = 0.20,
) -> Path | None:
    """Write benchmark aggregates as JSON."""
    payload: dict[str, object] = {
        "success_threshold": float(success_threshold),
        "entries": [
            {
                "method": e.method,
                "avg_ser": float(e.avg_ser),
                "avg_score": float(e.avg_score),
                "avg_time": float(e.avg_time),
                "success_rate": float(e.success_rate),
                "trials": int(e.trials),
            }
            for e in entries
        ],
    }
    return _write_json_report(payload, save_path, "benchmark.json")

