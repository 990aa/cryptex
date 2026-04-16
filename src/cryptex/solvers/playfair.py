"""Playfair cipher cracker via MCMC over the 5×5 key matrix.

Optimised inner loop: builds a 25×25 digraph→digraph lookup table from the
key matrix so that decrypt is a fast array operation rather than rebuilding
the matrix every iteration.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable

import numpy as np

from cryptex.ngram import NgramModel


# Alphabet used by Playfair (no j)
PF_ALPHA = "abcdefghiklmnopqrstuvwxyz"  # 25 chars
PF_CH2I = {ch: i for i, ch in enumerate(PF_ALPHA)}


@dataclass
class PlayfairConfig:
    iterations: int = 80_000
    num_restarts: int = 6
    t_start: float = 50.0
    t_min: float = 0.005
    cooling_rate: float = 0.99989
    callback_interval: int = 5000


@dataclass
class PlayfairResult:
    best_key: str = ""
    best_plaintext: str = ""
    best_score: float = -math.inf
    iterations_total: int = 0


PlayfairCallback = Callable[[int, int, str, float, float], None]


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


def _fast_decrypt_score(
    ct_pairs: np.ndarray,
    table: np.ndarray,
    log_quad: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Decrypt and score using precomputed table."""
    N = len(ct_pairs)
    packed = table[ct_pairs[:, 0], ct_pairs[:, 1]]
    pi = packed // 25
    pj = packed % 25

    plain = np.empty(N * 2, dtype=np.int8)
    plain[0::2] = pi.astype(np.int8)
    plain[1::2] = pj.astype(np.int8)

    n = len(plain)
    if n < 4:
        return 0.0, plain
    score = float(log_quad[plain[:-3], plain[1:-2], plain[2:-1], plain[3:]].sum())
    return score, plain


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
) -> PlayfairResult:
    """Crack a Playfair cipher using MCMC with simulated annealing."""
    if config is None:
        config = PlayfairConfig()

    result = PlayfairResult()

    # Clean ciphertext
    ct = ciphertext.replace("j", "i")
    ct = "".join(ch for ch in ct if ch in PF_ALPHA)
    if len(ct) % 2 != 0:
        ct += "x"

    # Convert ciphertext to pairs of integer indices
    ct_pairs = np.array(
        [[PF_CH2I[ct[i]], PF_CH2I[ct[i + 1]]] for i in range(0, len(ct), 2)],
        dtype=np.int8,
    )

    log_quad = model.log_quad
    assert log_quad is not None

    for chain_idx in range(config.num_restarts):
        perm = list(range(25))
        random.shuffle(perm)
        key = perm

        table = _build_decrypt_table(key)
        score, plain = _fast_decrypt_score(ct_pairs, table, log_quad)

        best_chain_score = score
        temperature = config.t_start

        for it in range(1, config.iterations + 1):
            move_type, undo_info = _propose(key)
            new_table = _build_decrypt_table(key)
            new_score, new_plain = _fast_decrypt_score(ct_pairs, new_table, log_quad)

            delta = new_score - score
            if delta > 0 or random.random() < math.exp(delta / max(temperature, 1e-12)):
                score = new_score
                plain = new_plain
                table = new_table
            else:
                _undo(key, move_type, undo_info)

            if score > best_chain_score:
                best_chain_score = score

            if score > result.best_score:
                result.best_score = score
                result.best_key = "".join(PF_ALPHA[k] for k in key)
                result.best_plaintext = _idx_to_str(plain)

            if temperature > config.t_min:
                temperature *= config.cooling_rate

            if callback and it % config.callback_interval == 0:
                pt_str = _idx_to_str(plain)
                callback(chain_idx, it, pt_str[:120], score, temperature)

        result.iterations_total += config.iterations

    return result

