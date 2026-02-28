"""Genetic algorithm solver for substitution ciphers.

An alternative metaheuristic to MCMC for benchmarking comparison.
Uses tournament selection, order crossover, and swap mutation.
"""

from __future__ import annotations

import math
import random
import string
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from cipher.ngram import NgramModel, text_to_indices


@dataclass
class GeneticConfig:
    """Parameters for the genetic algorithm solver."""

    population_size: int = 500
    """Number of individuals in the population."""

    generations: int = 1000
    """Number of generations to evolve."""

    elite_fraction: float = 0.10
    """Top fraction preserved unchanged each generation."""

    tournament_size: int = 5
    """Tournament selection group size."""

    crossover_rate: float = 0.65
    """Probability of crossover vs direct copy."""

    mutation_rate: float = 0.08
    """Per-individual probability of mutation (swap two key positions)."""

    num_mutations: int = 1
    """Number of swap mutations per mutated individual."""

    callback_interval: int = 50
    """Report progress every N generations."""


@dataclass
class GeneticResult:
    best_key: str = ""
    best_plaintext: str = ""
    best_score: float = -math.inf
    generations_run: int = 0
    score_trajectory: list[tuple[int, float]] = field(default_factory=list)


GeneticCallback = Callable[[int, str, str, float], None]
# callback(generation, current_key, current_plaintext, current_score)


def _random_key() -> list[int]:
    k = list(range(26))
    random.shuffle(k)
    return k


def _decode(ct_idx: np.ndarray, key: list[int], include_space: bool) -> np.ndarray:
    """Vectorised decrypt: cipher index -> plain index via key."""
    offset = 1 if include_space else 0
    A = 27 if include_space else 26
    inv = np.zeros(A, dtype=np.int8)
    if include_space:
        inv[0] = 0
    for ci in range(26):
        cipher_i = key[ci] + offset
        plain_i = ci + offset
        inv[cipher_i] = plain_i
    return inv[ct_idx]


def _score(ct_idx: np.ndarray, key: list[int], model: NgramModel) -> float:
    plain_idx = _decode(ct_idx, key, model.include_space)
    return model.score_quadgrams_fast(plain_idx)


def _order_crossover(p1: list[int], p2: list[int]) -> list[int]:
    """Order crossover (OX1) for permutation chromosomes."""
    n = len(p1)
    start, end = sorted(random.sample(range(n), 2))
    child = [-1] * n
    # Copy segment from p1
    child[start:end] = p1[start:end]
    # Fill remaining from p2 in order
    used = set(child[start:end])
    fill = [g for g in p2 if g not in used]
    fi = 0
    for i in range(n):
        if child[i] == -1:
            child[i] = fill[fi]
            fi += 1
    return child


def _mutate(key: list[int], n_swaps: int = 1) -> None:
    """In-place swap mutation."""
    for _ in range(n_swaps):
        i, j = random.sample(range(26), 2)
        key[i], key[j] = key[j], key[i]


def _idx_to_text(plain_idx: np.ndarray, include_space: bool) -> str:
    offset = 1 if include_space else 0
    chars = []
    for v in plain_idx:
        if include_space and v == 0:
            chars.append(" ")
        else:
            chars.append(chr(ord("a") + v - offset))
    return "".join(chars)


def run_genetic(
    ciphertext: str,
    model: NgramModel,
    config: GeneticConfig | None = None,
    callback: GeneticCallback | None = None,
) -> GeneticResult:
    """Solve a substitution cipher using a genetic algorithm.

    Parameters
    ----------
    ciphertext : str
        Lowercase ciphertext.
    model : NgramModel
        Trained language model.
    config : GeneticConfig, optional
    callback : callable, optional

    Returns
    -------
    GeneticResult
    """
    if config is None:
        config = GeneticConfig()

    result = GeneticResult()

    ct_alpha = "".join(
        ch for ch in ciphertext if ch in string.ascii_lowercase or ch == " "
    )
    ct_idx = text_to_indices(ct_alpha, model.include_space)

    # Initialise population
    population = [_random_key() for _ in range(config.population_size)]
    fitness = np.array([_score(ct_idx, k, model) for k in population])

    n_elite = max(1, int(config.population_size * config.elite_fraction))

    for gen in range(1, config.generations + 1):
        # Sort by fitness (descending)
        order = np.argsort(fitness)[::-1]
        population = [population[i] for i in order]
        fitness = fitness[order]

        best_key = population[0]
        best_fit = float(fitness[0])

        if best_fit > result.best_score:
            result.best_score = best_fit
            result.best_key = "".join(string.ascii_lowercase[i] for i in best_key)

        result.score_trajectory.append((gen, result.best_score))

        # Callback
        if callback and gen % config.callback_interval == 0:
            plain_idx = _decode(ct_idx, best_key, model.include_space)
            pt_str = _idx_to_text(plain_idx, model.include_space)
            callback(gen, result.best_key, pt_str, result.best_score)

        # Elitism: keep top individuals
        new_pop = [list(population[i]) for i in range(n_elite)]

        # Fill rest via tournament selection + crossover + mutation
        while len(new_pop) < config.population_size:
            # Tournament selection
            def _tournament() -> list[int]:
                candidates = random.sample(range(config.population_size), config.tournament_size)
                best_c = max(candidates, key=lambda c: fitness[c])
                return list(population[best_c])

            parent1 = _tournament()
            parent2 = _tournament()

            if random.random() < config.crossover_rate:
                child = _order_crossover(parent1, parent2)
            else:
                child = list(parent1)

            if random.random() < config.mutation_rate:
                _mutate(child, config.num_mutations)

            new_pop.append(child)

        population = new_pop
        fitness = np.array([_score(ct_idx, k, model) for k in population])
        result.generations_run = gen

    # Final best
    best_idx = int(np.argmax(fitness))
    best_key = population[best_idx]
    result.best_key = "".join(string.ascii_lowercase[i] for i in best_key)
    plain_idx = _decode(ct_idx, best_key, model.include_space)
    result.best_plaintext = _idx_to_text(plain_idx, model.include_space)
    result.best_score = float(fitness[best_idx])

    return result
