"""HMM / Expectation-Maximisation (Baum-Welch) solver for substitution ciphers.

Frames cipher-cracking as unsupervised sequence labelling:
  hidden states  = plaintext characters (a-z, optionally space)
  observations   = ciphertext characters
  transitions    = bigram language model
  emissions      = cipher mapping (learned via EM)

This approach naturally handles noisy / probabilistic ciphers.
"""

from __future__ import annotations

import math
import string
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from cipher.ngram import NgramModel


# Config


@dataclass
class HMMConfig:
    """Parameters for the HMM/EM solver."""

    max_iter: int = 100
    """Maximum EM iterations."""

    tol: float = 1e-4
    """Convergence tolerance on log-likelihood change."""

    callback_interval: int = 1
    """Call progress callback every N EM iterations."""


@dataclass
class HMMResult:
    best_key: str = ""
    best_plaintext: str = ""
    best_score: float = -math.inf
    iterations: int = 0
    log_likelihoods: list[float] = field(default_factory=list)


# callback(iteration, plaintext, log_likelihood)
HMMCallback = Callable[[int, str, float], None]


# Helpers


def _logsumexp(a: np.ndarray, axis: int | None = None) -> np.ndarray:
    """Numerically stable log-sum-exp."""
    a_max = np.max(a, axis=axis, keepdims=True)
    # Guard against -inf
    a_max = np.where(np.isfinite(a_max), a_max, 0.0)
    out = np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True)) + a_max
    if axis is not None:
        return out.squeeze(axis=axis)
    return out.squeeze()


def _normalise_log(log_probs: np.ndarray, axis: int = -1) -> np.ndarray:
    """Normalise log-probabilities along *axis* so they sum to 1 in probability space."""
    lse = _logsumexp(log_probs, axis=axis)
    if lse.ndim < log_probs.ndim:
        lse = np.expand_dims(lse, axis=axis)
    return log_probs - lse


# Core solver


def run_hmm(
    ciphertext: str,
    model: NgramModel,
    config: HMMConfig | None = None,
    callback: HMMCallback | None = None,
) -> HMMResult:
    """Crack a substitution cipher using the HMM/EM (Baum-Welch) algorithm.

    Parameters
    ----------
    ciphertext : str
        Lowercased ciphertext (letters + optional spaces).
    model : NgramModel
        Trained n-gram model (uses bigram log-probabilities for transitions).
    config : HMMConfig, optional
    callback : callable, optional
        Progress callback receiving (iteration, current_plaintext, log_likelihood).

    Returns
    -------
    HMMResult
    """
    if config is None:
        config = HMMConfig()

    A = model.A  # alphabet size (26 or 27)
    include_space = model.include_space

    # Map ciphertext to integer observations
    obs_chars = [
        ch
        for ch in ciphertext
        if ch in string.ascii_lowercase or (include_space and ch == " ")
    ]
    T = len(obs_chars)
    if T == 0:
        return HMMResult()

    obs = np.empty(T, dtype=np.int32)
    for t, ch in enumerate(obs_chars):
        if ch == " " and include_space:
            obs[t] = 0
        else:
            obs[t] = ord(ch) - ord("a") + (1 if include_space else 0)

    # Initialise HMM parameters

    # Transition matrix (log): use bigram log-probs from the language model
    # log_trans[i, j] = log P(hidden_j | hidden_i)
    assert model.log_bi is not None
    log_trans = model.log_bi.copy()  # (A, A)

    # Initial state distribution (log): use unigram
    assert model.log_uni is not None
    log_pi = model.log_uni.copy()  # (A,)

    # Emission matrix (log): start near-uniform with slight random perturbation
    # log_emit[hidden_state, observed_symbol]
    log_emit = np.random.dirichlet(np.ones(A) * 10, size=A)
    log_emit = np.log(log_emit + 1e-30)

    result = HMMResult()
    prev_ll = -np.inf

    for em_iter in range(1, config.max_iter + 1):
        # ==============================================================
        # E-step: Forward-Backward
        # ==============================================================

        # Forward (log-space)
        log_alpha = np.full((T, A), -np.inf)
        log_alpha[0] = log_pi + log_emit[:, obs[0]]

        for t in range(1, T):
            # log_alpha[t, j] = log(sum_i alpha[t-1,i] * trans[i,j]) + emit[j, obs[t]]
            # In log-space: logsumexp over i of (log_alpha[t-1, i] + log_trans[i, j])
            for j in range(A):
                log_alpha[t, j] = (
                    _logsumexp(log_alpha[t - 1] + log_trans[:, j]) + log_emit[j, obs[t]]
                )

        # Log-likelihood
        ll = float(_logsumexp(log_alpha[T - 1]))
        result.log_likelihoods.append(ll)

        # Backward (log-space)
        log_beta = np.full((T, A), -np.inf)
        log_beta[T - 1] = 0.0  # log(1)

        for t in range(T - 2, -1, -1):
            for i in range(A):
                log_beta[t, i] = _logsumexp(
                    log_trans[i, :] + log_emit[:, obs[t + 1]] + log_beta[t + 1]
                )

        # Posterior state probabilities  γ[t, i] = P(state_t = i | observations)
        log_gamma = log_alpha + log_beta
        log_gamma = log_gamma - _logsumexp(log_gamma, axis=1)[:, None]

        # Posterior transition probabilities  ξ[t, i, j]
        # Only needed as accumulated counts for the M-step
        log_xi_sum = np.full((A, A), -np.inf)
        for t in range(T - 1):
            for i in range(A):
                for j in range(A):
                    val = (
                        log_alpha[t, i]
                        + log_trans[i, j]
                        + log_emit[j, obs[t + 1]]
                        + log_beta[t + 1, j]
                    )
                    log_xi_sum[i, j] = np.logaddexp(log_xi_sum[i, j], val)
        # Normalise
        log_xi_sum = log_xi_sum - _logsumexp(log_xi_sum)

        # ==============================================================
        # M-step: re-estimate parameters
        # ==============================================================

        # We keep transitions FIXED (they encode the language model).
        # We only update emissions and initial-state distribution.

        # --- Update emission probabilities ---
        gamma = np.exp(log_gamma)  # (T, A)  — probabilities
        new_emit = np.full((A, A), 1e-10)  # Laplace-ish floor
        for t in range(T):
            new_emit[:, obs[t]] += gamma[t]
        # Normalise rows
        new_emit /= new_emit.sum(axis=1, keepdims=True)
        log_emit = np.log(new_emit + 1e-30)

        # --- Update initial distribution ---
        log_pi = log_gamma[0].copy()

        # ==============================================================
        # Decode current best plaintext (MAP per position)
        # ==============================================================
        best_states = np.argmax(log_gamma, axis=1)
        plaintext_chars: list[str] = []
        for s in best_states:
            if include_space and s == 0:
                plaintext_chars.append(" ")
            else:
                plaintext_chars.append(chr(ord("a") + s - (1 if include_space else 0)))
        current_plaintext = "".join(plaintext_chars)

        result.iterations = em_iter

        if callback and em_iter % config.callback_interval == 0:
            callback(em_iter, current_plaintext, ll)

        # --- convergence check ---
        if abs(ll - prev_ll) < config.tol:
            break
        prev_ll = ll

    # Final key extraction: from emission matrix
    # For each observed symbol, find which hidden state has highest emission
    # The "key" maps observed → plaintext
    emit_probs = np.exp(log_emit)  # (hidden, observed)
    key_map: dict[int, int] = {}
    for obs_idx in range(A):
        key_map[obs_idx] = int(np.argmax(emit_probs[:, obs_idx]))

    # Build key string (maps cipher letter → plain letter)
    key_chars: list[str] = []
    offset = 1 if include_space else 0
    for c in range(26):
        obs_i = c + offset
        hidden_i = key_map.get(obs_i, obs_i)
        if include_space and hidden_i == 0:
            key_chars.append(" ")
        else:
            key_chars.append(chr(ord("a") + hidden_i - offset))
    result.best_key = "".join(key_chars)
    result.best_plaintext = current_plaintext
    result.best_score = ll

    return result
