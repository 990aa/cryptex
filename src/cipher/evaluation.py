"""Evaluation framework for cipher cracking.

Provides:
- Symbol Error Rate (SER) metric
- Phase transition analysis (success rate vs ciphertext length)
- Benchmarking against baselines (frequency analysis, genetic algorithm, random restarts)
"""

from __future__ import annotations

import math
import random
import string
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from cipher.ciphers import SimpleSubstitution
from cipher.ngram import NgramModel, text_to_indices


# ------------------------------------------------------------------
# Symbol Error Rate
# ------------------------------------------------------------------


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


# ------------------------------------------------------------------
# Phase Transition Analysis
# ------------------------------------------------------------------


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
    from cipher.mcmc import MCMCConfig, run_mcmc

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
    """Plot the phase transition graph: success rate vs ciphertext length."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(
        pt_result.lengths,
        pt_result.success_rates,
        "b-o",
        linewidth=2,
        markersize=8,
        label="Success Rate",
    )
    ax1.set_xlabel("Ciphertext Length (characters)")
    ax1.set_ylabel("Success Rate", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax1.set_ylim(-0.05, 1.05)
    ax1.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="50% threshold")

    if pt_result.threshold_length:
        ax1.axvline(
            x=pt_result.threshold_length,
            color="green",
            linestyle="--",
            alpha=0.7,
            label=f"Threshold ≈ {pt_result.threshold_length} chars",
        )

    # SER on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(
        pt_result.lengths,
        pt_result.avg_ser,
        "r--s",
        linewidth=1.5,
        markersize=6,
        alpha=0.7,
        label="Avg SER",
    )
    ax2.set_ylabel("Average Symbol Error Rate", color="red")
    ax2.tick_params(axis="y", labelcolor="red")
    ax2.set_ylim(-0.05, 1.05)

    ax1.set_title("Phase Transition: Decryption Success vs. Ciphertext Length")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    plt.tight_layout()

    if save_path is None:
        save_path = Path("phase_transition.png")
    else:
        save_path = Path(save_path)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


# ------------------------------------------------------------------
# Benchmarking Against Baselines
# ------------------------------------------------------------------


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


def _random_restart_baseline(
    ciphertext: str, model: NgramModel, n_restarts: int = 50
) -> str:
    """Random key restarts without MCMC — just try random keys and keep the best."""
    ct_idx = text_to_indices(
        "".join(ch for ch in ciphertext if ch in string.ascii_lowercase or ch == " "),
        model.include_space,
    )
    offset = 1 if model.include_space else 0
    A = model.A

    best_score = -math.inf
    best_key = None

    for _ in range(n_restarts):
        key = list(string.ascii_lowercase)
        random.shuffle(key)

        inv = np.zeros(A, dtype=np.int8)
        if model.include_space:
            inv[0] = 0
        for ci in range(26):
            inv[ord(key[ci]) - ord("a") + offset] = ci + offset
        plain_idx = inv[ct_idx]
        score = model.score_quadgrams_fast(plain_idx)

        if score > best_score:
            best_score = score
            best_key = key

    # Decode best
    if best_key is None:
        return ciphertext
    inv = np.zeros(A, dtype=np.int8)
    if model.include_space:
        inv[0] = 0
    for ci in range(26):
        inv[ord(best_key[ci]) - ord("a") + offset] = ci + offset
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
    callback=None,
) -> list[BenchmarkEntry]:
    """Benchmark MCMC vs Genetic Algorithm vs Frequency Analysis vs Random Restarts.

    Parameters
    ----------
    model : NgramModel
    corpus : str
        Source text for generating test cases.
    text_length : int
        Length of each test ciphertext.
    trials : int
        Number of trials per method.
    callback : callable, optional
        callback(method_name, trial, ser, elapsed)

    Returns
    -------
    list of BenchmarkEntry
    """
    from cipher.genetic import GeneticConfig, run_genetic
    from cipher.mcmc import MCMCConfig, run_mcmc

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
        "Random": BenchmarkEntry(method="Random Restarts", trials=trials),
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
        if ser < 0.05:
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
        if ser < 0.05:
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
                "".join(ch for ch in freq_result if ch in string.ascii_lowercase or ch == " "),
                model.include_space,
            )
        )
        methods["Frequency"].avg_ser += ser
        methods["Frequency"].avg_score += freq_score
        methods["Frequency"].avg_time += elapsed
        if ser < 0.05:
            methods["Frequency"].success_rate += 1
        if callback:
            callback("Frequency", trial_idx, ser, elapsed)

        # --- Random Restarts ---
        t0 = time.time()
        rand_result = _random_restart_baseline(ciphertext, model, n_restarts=1000)
        elapsed = time.time() - t0
        ser = symbol_error_rate(plaintext, rand_result)
        rand_score = model.score_quadgrams_fast(
            text_to_indices(
                "".join(ch for ch in rand_result if ch in string.ascii_lowercase or ch == " "),
                model.include_space,
            )
        )
        methods["Random"].avg_ser += ser
        methods["Random"].avg_score += rand_score
        methods["Random"].avg_time += elapsed
        if ser < 0.05:
            methods["Random"].success_rate += 1
        if callback:
            callback("Random", trial_idx, ser, elapsed)

    # Average
    results = []
    for entry in methods.values():
        entry.avg_ser /= max(trials, 1)
        entry.avg_score /= max(trials, 1)
        entry.avg_time /= max(trials, 1)
        entry.success_rate /= max(trials, 1)
        results.append(entry)

    return results


def plot_benchmark(
    entries: list[BenchmarkEntry],
    save_path: str | Path | None = None,
) -> Path | None:
    """Plot benchmark comparison as a grouped bar chart."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = [e.method for e in entries]
    sers = [e.avg_ser for e in entries]
    success = [e.success_rate for e in entries]
    times = [e.avg_time for e in entries]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    colors = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0"]

    # SER
    axes[0].bar(methods, sers, color=colors[: len(methods)])
    axes[0].set_ylabel("Avg Symbol Error Rate")
    axes[0].set_title("Accuracy (lower is better)")
    axes[0].set_ylim(0, 1)

    # Success rate
    axes[1].bar(methods, success, color=colors[: len(methods)])
    axes[1].set_ylabel("Success Rate")
    axes[1].set_title("Success Rate (SER < 5%)")
    axes[1].set_ylim(0, 1.05)

    # Time
    axes[2].bar(methods, times, color=colors[: len(methods)])
    axes[2].set_ylabel("Avg Time (s)")
    axes[2].set_title("Speed")

    for ax in axes:
        ax.tick_params(axis="x", rotation=15)

    plt.suptitle("Benchmark: Cipher Cracking Methods", fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path is None:
        save_path = Path("benchmark.png")
    else:
        save_path = Path(save_path)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path
