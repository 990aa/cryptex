"""Convergence analysis and diagnostics for MCMC cipher cracking.

Provides:
- Score trajectory plotting (matplotlib + terminal sparkline)
- Multi-chain consensus via majority vote over key mappings
- Posterior distribution visualization (letter-mapping confidence heatmap)
- Convergence quality metrics
"""

from __future__ import annotations

import string
from collections import Counter
from pathlib import Path

import numpy as np

from cipher.mcmc import MCMCResult


# ------------------------------------------------------------------
# Score Trajectory
# ------------------------------------------------------------------


def plot_score_trajectory(
    result: MCMCResult,
    reference_score: float | None = None,
    save_path: str | Path | None = None,
    show: bool = False,
) -> Path | None:
    """Plot the score trajectory over iterations.

    Parameters
    ----------
    result : MCMCResult
        Result with score_trajectory populated.
    reference_score : float, optional
        Known-good English score to draw as a reference line.
    save_path : str or Path, optional
        Where to save the plot image. Defaults to 'convergence_plot.png'.
    show : bool
        Whether to display the plot interactively.

    Returns
    -------
    Path or None
        Path to the saved plot image.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not result.score_trajectory:
        return None

    iters = [t[0] for t in result.score_trajectory]
    scores = [t[1] for t in result.score_trajectory]

    fig, ax1 = plt.subplots(figsize=(12, 5))

    # Score trajectory
    ax1.plot(iters, scores, "b-", linewidth=1.5, label="Best Score")
    if reference_score is not None:
        ax1.axhline(
            y=reference_score,
            color="green",
            linestyle="--",
            linewidth=1,
            label=f"Reference ({reference_score:.1f})",
        )
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Log-probability Score", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax1.legend(loc="lower right")
    ax1.set_title("MCMC Score Convergence")

    # Mark chain boundaries
    if result.chain_scores:
        chain_iters = len(result.score_trajectory) // max(len(result.chain_scores), 1)
        for ci in range(1, len(result.chain_scores)):
            boundary = ci * chain_iters
            if boundary < len(iters):
                ax1.axvline(
                    x=iters[min(boundary, len(iters) - 1)],
                    color="gray",
                    linestyle=":",
                    alpha=0.5,
                )

    # Acceptance rate on secondary axis
    if result.acceptance_rates:
        ax2 = ax1.twinx()
        rate_iters = iters[: len(result.acceptance_rates)]
        ax2.plot(rate_iters, result.acceptance_rates, "r-", alpha=0.4, linewidth=0.8, label="Accept Rate")
        ax2.set_ylabel("Acceptance Rate", color="red")
        ax2.tick_params(axis="y", labelcolor="red")
        ax2.set_ylim(0, 1)
        ax2.legend(loc="upper right")

    plt.tight_layout()

    if save_path is None:
        save_path = Path("convergence_plot.png")
    else:
        save_path = Path(save_path)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


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


# ------------------------------------------------------------------
# Multi-Chain Consensus
# ------------------------------------------------------------------


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
        return {"agreement_rate": 0.0, "consensus_key": "", "per_letter_confidence": {}, "num_chains": 0}

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


# ------------------------------------------------------------------
# Posterior Distribution (Key Confidence Heatmap)
# ------------------------------------------------------------------


def key_confidence_heatmap(
    result: MCMCResult,
    save_path: str | Path | None = None,
) -> Path | None:
    """Generate a 26×26 heatmap showing mapping confidence across chains.

    Rows = ciphertext letters (a-z), Columns = plaintext letters (a-z).
    Cell intensity = fraction of chains that mapped cipher→plain.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="equal", vmin=0, vmax=1)

    letters = list(string.ascii_lowercase)
    ax.set_xticks(range(26))
    ax.set_xticklabels(letters)
    ax.set_yticks(range(26))
    ax.set_yticklabels(letters)
    ax.set_xlabel("Plaintext Letter")
    ax.set_ylabel("Cipher Position")
    ax.set_title("Key Mapping Confidence (across chains)")

    plt.colorbar(im, ax=ax, label="Confidence")
    plt.tight_layout()

    if save_path is None:
        save_path = Path("key_heatmap.png")
    else:
        save_path = Path(save_path)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


# ------------------------------------------------------------------
# Frequency Alignment Visualization
# ------------------------------------------------------------------


ENGLISH_FREQ = np.array([
    0.08167, 0.01492, 0.02782, 0.04253, 0.12702, 0.02228, 0.02015,
    0.06094, 0.06966, 0.00153, 0.00772, 0.04025, 0.02406, 0.06749,
    0.07507, 0.01929, 0.00095, 0.05987, 0.06327, 0.09056, 0.02758,
    0.00978, 0.02360, 0.00150, 0.01974, 0.00074,
])


def frequency_comparison_plot(
    ciphertext: str,
    decrypted: str,
    save_path: str | Path | None = None,
) -> Path | None:
    """Plot character frequency comparison: ciphertext vs decrypted vs English."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _freq(text: str) -> np.ndarray:
        counts = np.zeros(26)
        for ch in text:
            if ch in string.ascii_lowercase:
                counts[ord(ch) - ord("a")] += 1
        total = counts.sum()
        return counts / total if total > 0 else counts

    ct_freq = _freq(ciphertext)
    pt_freq = _freq(decrypted)

    x = np.arange(26)
    width = 0.25
    letters = list(string.ascii_lowercase)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - width, ENGLISH_FREQ, width, label="English", alpha=0.7, color="green")
    ax.bar(x, ct_freq, width, label="Ciphertext", alpha=0.7, color="red")
    ax.bar(x + width, pt_freq, width, label="Decrypted", alpha=0.7, color="blue")

    ax.set_xlabel("Letter")
    ax.set_ylabel("Frequency")
    ax.set_title("Character Frequency Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(letters)
    ax.legend()
    plt.tight_layout()

    if save_path is None:
        save_path = Path("frequency_comparison.png")
    else:
        save_path = Path(save_path)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path
