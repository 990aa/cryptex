"""Playfair cipher cracker using population-based hill climbing.

This implementation keeps a population of candidate keys and performs
temperature-guided local improvements with periodic elite injection.
"""

from __future__ import annotations

import math
import random
import warnings
from dataclasses import dataclass
from threading import Event
from typing import Callable

import numpy as np

from cryptex.ngram import NgramModel
from cryptex.warnings import CryptexWarning


# Alphabet used by Playfair (no j)
PF_ALPHA = "abcdefghiklmnopqrstuvwxyz"  # 25 chars
PF_CH2I = {ch: i for i, ch in enumerate(PF_ALPHA)}
PF_TO_ALPHA_IDX = np.asarray([ord(ch) - ord("a") for ch in PF_ALPHA], dtype=np.int8)


@dataclass
class PlayfairConfig:
    iterations: int = 200_000
    num_restarts: int = 10
    population_size: int = 20
    t_start: float = 20.0
    t_min: float = 0.001
    cooling_rate: float = 0.99995
    use_digraph_scoring: bool = True
    min_recommended_length: int = 100
    callback_interval: int = 5000


@dataclass
class PlayfairResult:
    best_key: str = ""
    best_plaintext: str = ""
    best_score: float = -math.inf
    iterations_total: int = 0


PlayfairCallback = Callable[[int, int, str, float, float], None]


@dataclass
class _Candidate:
    key: list[int]
    table: np.ndarray
    plain_pf: np.ndarray
    score: float


# Fast Playfair decrypt via lookup table


def _build_decrypt_table(key: list[int]) -> np.ndarray:
    """Build a 25×25 lookup table: table[ci, cj] → packed (pi*25+pj).

    key is a list of 25 ints (letter indices), representing the 5×5 matrix
    in row-major order.
    """
    row = [0] * 25
    col = [0] * 25
    for idx in range(25):
        r, c = divmod(idx, 5)
        letter = key[idx]
        row[letter] = r
        col[letter] = c

    table = np.empty((25, 25), dtype=np.int16)
    for a in range(25):
        r1, c1 = row[a], col[a]
        for b in range(25):
            if a == b:
                table[a, b] = a * 25 + b
                continue
            r2, c2 = row[b], col[b]
            if r1 == r2:
                p1 = key[r1 * 5 + (c1 - 1) % 5]
                p2 = key[r2 * 5 + (c2 - 1) % 5]
            elif c1 == c2:
                p1 = key[((r1 - 1) % 5) * 5 + c1]
                p2 = key[((r2 - 1) % 5) * 5 + c2]
            else:
                p1 = key[r1 * 5 + c2]
                p2 = key[r2 * 5 + c1]
            table[a, b] = p1 * 25 + p2
    return table


def _decode_pairs(ct_pairs: np.ndarray, table: np.ndarray) -> np.ndarray:
    """Decode Playfair ciphertext pairs into 25-letter plaintext indices."""
    N = len(ct_pairs)
    packed = table[ct_pairs[:, 0], ct_pairs[:, 1]]
    pi = packed // 25
    pj = packed % 25

    plain_pf = np.empty(N * 2, dtype=np.int8)
    plain_pf[0::2] = pi.astype(np.int8)
    plain_pf[1::2] = pj.astype(np.int8)
    return plain_pf


def _score_playfair(plain: np.ndarray, model: NgramModel) -> float:
    """Combined digraph and quadgram score for Playfair."""
    digraph_score = model.score_digraphs(plain)
    quad_score = model.score_quadgrams_fast(plain)
    return 0.6 * digraph_score + 0.4 * quad_score


def _evaluate_key(
    key: list[int],
    ct_pairs: np.ndarray,
    model: NgramModel,
    use_digraph_scoring: bool,
) -> _Candidate:
    table = _build_decrypt_table(key)
    plain_pf = _decode_pairs(ct_pairs, table)
    plain_alpha = PF_TO_ALPHA_IDX[plain_pf]
    if use_digraph_scoring:
        score = _score_playfair(plain_alpha, model)
    else:
        score = model.score_quadgrams_fast(plain_alpha)
    return _Candidate(
        key=list(key),
        table=table,
        plain_pf=plain_pf,
        score=float(score),
    )


def _idx_to_str(plain: np.ndarray) -> str:
    return "".join(PF_ALPHA[i] for i in plain)


# Proposal moves


def _propose(key: list[int]) -> tuple[str, tuple]:
    move = random.choices(
        ["swap", "swap_rows", "swap_cols", "reverse", "rotate_row", "rotate_col"],
        weights=[50, 15, 15, 5, 10, 10],
        k=1,
    )[0]

    if move == "swap":
        i, j = random.sample(range(25), 2)
        key[i], key[j] = key[j], key[i]
        return move, (i, j)
    elif move == "swap_rows":
        r1, r2 = random.sample(range(5), 2)
        for c in range(5):
            a, b = r1 * 5 + c, r2 * 5 + c
            key[a], key[b] = key[b], key[a]
        return move, (r1, r2)
    elif move == "swap_cols":
        c1, c2 = random.sample(range(5), 2)
        for r in range(5):
            a, b = r * 5 + c1, r * 5 + c2
            key[a], key[b] = key[b], key[a]
        return move, (c1, c2)
    elif move == "reverse":
        key.reverse()
        return move, ()
    elif move == "rotate_row":
        r = random.randint(0, 4)
        d = random.choice([-1, 1])
        row = [key[r * 5 + c] for c in range(5)]
        row = [row[-1]] + row[:-1] if d == 1 else row[1:] + [row[0]]
        for c in range(5):
            key[r * 5 + c] = row[c]
        return move, (r, d)
    else:  # rotate_col
        col = random.randint(0, 4)
        d = random.choice([-1, 1])
        vals = [key[r * 5 + col] for r in range(5)]
        vals = [vals[-1]] + vals[:-1] if d == 1 else vals[1:] + [vals[0]]
        for r in range(5):
            key[r * 5 + col] = vals[r]
        return move, (col, d)


def _undo(key: list[int], move_type: str, info: tuple) -> None:
    if move_type == "swap":
        i, j = info
        key[i], key[j] = key[j], key[i]
    elif move_type == "swap_rows":
        r1, r2 = info
        for c in range(5):
            a, b = r1 * 5 + c, r2 * 5 + c
            key[a], key[b] = key[b], key[a]
    elif move_type == "swap_cols":
        c1, c2 = info
        for r in range(5):
            a, b = r * 5 + c1, r * 5 + c2
            key[a], key[b] = key[b], key[a]
    elif move_type == "reverse":
        key.reverse()
    elif move_type == "rotate_row":
        r, d = info
        row = [key[r * 5 + c] for c in range(5)]
        row = row[1:] + [row[0]] if d == 1 else [row[-1]] + row[:-1]
        for c in range(5):
            key[r * 5 + c] = row[c]
    elif move_type == "rotate_col":
        col, d = info
        vals = [key[r * 5 + col] for r in range(5)]
        vals = vals[1:] + [vals[0]] if d == 1 else [vals[-1]] + vals[:-1]
        for r in range(5):
            key[r * 5 + col] = vals[r]


# Main cracker


def crack_playfair(
    ciphertext: str,
    model: NgramModel,
    config: PlayfairConfig | None = None,
    callback: PlayfairCallback | None = None,
    stop_event: Event | None = None,
) -> PlayfairResult:
    """Crack a Playfair cipher using population-based hill climbing."""
    if config is None:
        config = PlayfairConfig()

    result = PlayfairResult()

    # Clean ciphertext
    ct_letters: list[str] = []
    for ch in ciphertext.lower():
        if ch == "j":
            ct_letters.append("i")
        elif ch in PF_ALPHA:
            ct_letters.append(ch)
    if len(ct_letters) < config.min_recommended_length:
        warnings.warn(
            (
                "Playfair cracking is unreliable below "
                f"{config.min_recommended_length} characters "
                f"(got {len(ct_letters)}). Results may be poor."
            ),
            CryptexWarning,
            stacklevel=2,
        )

    if len(ct_letters) % 2 != 0:
        ct_letters.append("x")
    if not ct_letters:
        return result
    ct = "".join(ct_letters)

    # Convert ciphertext to pairs of integer indices
    ct_pairs = np.array(
        [[PF_CH2I[ct[i]], PF_CH2I[ct[i + 1]]] for i in range(0, len(ct), 2)],
        dtype=np.int8,
    )

    pop_size = max(2, int(config.population_size))

    for chain_idx in range(config.num_restarts):
        if stop_event is not None and stop_event.is_set():
            break
        population: list[_Candidate] = []
        for _ in range(pop_size):
            key = list(range(25))
            random.shuffle(key)
            population.append(
                _evaluate_key(
                    key,
                    ct_pairs,
                    model,
                    use_digraph_scoring=config.use_digraph_scoring,
                )
            )

        temperature = float(config.t_start)

        for it in range(1, config.iterations + 1):
            if stop_event is not None and stop_event.is_set():
                break
            ranked_idx = sorted(
                range(len(population)),
                key=lambda i: population[i].score,
                reverse=True,
            )
            elite_count = max(2, pop_size // 5)
            elite_idx = ranked_idx[:elite_count]

            parent_idx = random.choice(elite_idx)
            parent = population[parent_idx]

            child_key = list(parent.key)
            for _ in range(1 + (1 if random.random() < 0.35 else 0)):
                _propose(child_key)

            child = _evaluate_key(
                child_key,
                ct_pairs,
                model,
                use_digraph_scoring=config.use_digraph_scoring,
            )

            delta = child.score - parent.score
            accept = delta >= 0 or random.random() < math.exp(
                delta / max(temperature, 1e-12)
            )

            if accept:
                low_band = (
                    ranked_idx[elite_count:] if elite_count < pop_size else ranked_idx
                )
                replace_idx = random.choice(low_band)
                population[replace_idx] = child

            # Periodic elite injection: mutate top candidates into weakest slots.
            if it % max(pop_size * 2, 20) == 0:
                ranked_idx = sorted(
                    range(len(population)),
                    key=lambda i: population[i].score,
                    reverse=True,
                )
                weakest = ranked_idx[-elite_count:]
                for w_idx in weakest:
                    seed = population[random.choice(elite_idx)].key.copy()
                    _propose(seed)
                    population[w_idx] = _evaluate_key(
                        seed,
                        ct_pairs,
                        model,
                        use_digraph_scoring=config.use_digraph_scoring,
                    )

            best = max(population, key=lambda c: c.score)
            if best.score > result.best_score:
                result.best_score = best.score
                result.best_key = "".join(PF_ALPHA[k] for k in best.key)
                result.best_plaintext = _idx_to_str(best.plain_pf)

            if callback and it % config.callback_interval == 0:
                callback(
                    chain_idx,
                    it,
                    _idx_to_str(best.plain_pf)[:120],
                    float(best.score),
                    float(temperature),
                )

            temperature = max(config.t_min, temperature * config.cooling_rate)

        result.iterations_total += config.iterations

    return result
