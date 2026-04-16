"""Convergence analysis and diagnostics for MCMC cipher cracking.

Provides:
- Score trajectory reports (JSON + terminal sparkline)
- Multi-chain consensus via majority vote over key mappings
- Posterior distribution report (letter-mapping confidence matrix)
- Convergence quality metrics
"""

from __future__ import annotations

import json
import string
from collections import Counter
from pathlib import Path

import numpy as np

from cryptex.solvers.mcmc import MCMCResult


def _write_json_report(
    payload: dict[str, object],
    save_path: str | Path | None,
    default_name: str,
) -> Path:
    path = Path(save_path) if save_path is not None else Path(default_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


# Score Trajectory


def plot_score_trajectory(
    result: MCMCResult,
    reference_score: float | None = None,
    save_path: str | Path | None = None,
    show: bool = False,
) -> Path | None:
    """Write score trajectory diagnostics as a JSON report.

    Parameters
    ----------
    result : MCMCResult
        Result with score_trajectory populated.
    reference_score : float, optional
        Known-good English score to include as a comparison value.
    save_path : str or Path, optional
        Where to save the JSON report. Defaults to 'convergence_plot.json'.
    show : bool
        Retained for API compatibility; ignored.

    Returns
    -------
    Path or None
        Path to the saved JSON report.
    """
    if not result.score_trajectory:
        return None

    iters = [t[0] for t in result.score_trajectory]
    scores = [t[1] for t in result.score_trajectory]
    chain_boundaries: list[int] = []
    if result.chain_scores:
        chain_iters = len(result.score_trajectory) // max(len(result.chain_scores), 1)
        for ci in range(1, len(result.chain_scores)):
            boundary = ci * chain_iters
            if boundary < len(iters):
                chain_boundaries.append(int(iters[min(boundary, len(iters) - 1)]))

    payload: dict[str, object] = {
        "iterations": [int(i) for i in iters],
        "scores": [float(s) for s in scores],
        "reference_score": (
            float(reference_score) if reference_score is not None else None
        ),
        "acceptance_rates": [float(r) for r in result.acceptance_rates],
        "chain_boundaries": chain_boundaries,
        "summary": {
            "min_score": float(min(scores)),
            "max_score": float(max(scores)),
            "final_score": float(scores[-1]),
        },
    }
    return _write_json_report(payload, save_path, "convergence_plot.json")


def sparkline_trajectory(result: MCMCResult, width: int = 60) -> str:
    """Return a terminal-friendly sparkline of the score trajectory."""
    if not result.score_trajectory:
        return ""
    scores = [t[1] for t in result.score_trajectory]
    lo, hi = min(scores), max(scores)
    rng = hi - lo if hi > lo else 1.0
    blocks = "▁▂▃▄▅▆▇█"
    # Downsample to width
    step = max(1, len(scores) // width)
    sampled = scores[::step][:width]
    return "".join(
        blocks[min(int((s - lo) / rng * (len(blocks) - 1)), len(blocks) - 1)]
        for s in sampled
    )


# Multi-Chain Consensus


def chain_consensus(result: MCMCResult) -> dict[str, object]:
    """Analyse inter-chain agreement on key mappings.

    Returns
    -------
    dict with keys:
        - 'agreement_rate': fraction of letter positions where majority of chains agree
        - 'consensus_key': the majority-vote key
        - 'per_letter_confidence': dict mapping each letter to its agreement fraction
        - 'num_chains': number of chains analysed
    """
    keys = result.chain_keys
    if not keys:
        return {
            "agreement_rate": 0.0,
            "consensus_key": "",
            "per_letter_confidence": {},
            "num_chains": 0,
        }

    n_chains = len(keys)
    consensus_chars: list[str] = []
    per_letter_conf: dict[str, float] = {}

    for pos in range(26):
        votes: list[str] = []
        for k in keys:
            if pos < len(k):
                votes.append(k[pos])
        if not votes:
            consensus_chars.append("?")
            continue
        counter = Counter(votes)
        winner, count = counter.most_common(1)[0]
        consensus_chars.append(winner)
        letter = string.ascii_lowercase[pos]
        per_letter_conf[letter] = count / n_chains

    consensus_key = "".join(consensus_chars)
    agreement = sum(per_letter_conf.values()) / 26.0 if per_letter_conf else 0.0

    return {
        "agreement_rate": agreement,
        "consensus_key": consensus_key,
        "per_letter_confidence": per_letter_conf,
        "num_chains": n_chains,
    }


# Posterior Distribution (Key Confidence Heatmap)


def key_confidence_heatmap(
    result: MCMCResult,
    save_path: str | Path | None = None,
) -> Path | None:
    """Write a 26×26 mapping-confidence matrix as JSON.

    Rows = ciphertext letters (a-z), Columns = plaintext letters (a-z).
    Cell intensity = fraction of chains that mapped cipher→plain.
    """
    keys = result.chain_keys
    if not keys:
        return None

    n_chains = len(keys)
    matrix = np.zeros((26, 26), dtype=np.float64)

    for key_str in keys:
        for cipher_idx in range(min(26, len(key_str))):
            plain_char = key_str[cipher_idx]
            plain_idx = ord(plain_char) - ord("a")
            if 0 <= plain_idx < 26:
                matrix[cipher_idx, plain_idx] += 1

    matrix /= max(n_chains, 1)

    letters = list(string.ascii_lowercase)
    payload: dict[str, object] = {
        "cipher_letters": letters,
        "plain_letters": letters,
        "confidence_matrix": matrix.tolist(),
        "num_chains": int(n_chains),
    }
    return _write_json_report(payload, save_path, "key_heatmap.json")


# Frequency Alignment Visualization


ENGLISH_FREQ = np.array(
    [
        0.08167,
        0.01492,
        0.02782,
        0.04253,
        0.12702,
        0.02228,
        0.02015,
        0.06094,
        0.06966,
        0.00153,
        0.00772,
        0.04025,
        0.02406,
        0.06749,
        0.07507,
        0.01929,
        0.00095,
        0.05987,
        0.06327,
        0.09056,
        0.02758,
        0.00978,
        0.02360,
        0.00150,
        0.01974,
        0.00074,
    ]
)


def frequency_comparison_plot(
    ciphertext: str,
    decrypted: str,
    save_path: str | Path | None = None,
) -> Path | None:
    """Write character-frequency comparison as JSON."""

    def _freq(text: str) -> np.ndarray:
        counts = np.zeros(26)
        for ch in text:
            if ch in string.ascii_lowercase:
                counts[ord(ch) - ord("a")] += 1
        total = counts.sum()
        return counts / total if total > 0 else counts

    ct_freq = _freq(ciphertext)
    pt_freq = _freq(decrypted)

    letters = list(string.ascii_lowercase)

    payload: dict[str, object] = {
        "letters": letters,
        "english": ENGLISH_FREQ.tolist(),
        "ciphertext": ct_freq.tolist(),
        "decrypted": pt_freq.tolist(),
    }
    return _write_json_report(payload, save_path, "frequency_comparison.json")

