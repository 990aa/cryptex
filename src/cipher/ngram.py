"""N-gram language model with interpolated smoothing.

Builds quadgram / trigram / bigram / unigram log-probability tables from a
training corpus.  Scoring is done entirely with NumPy arrays for speed —
the hot path does only integer indexing and addition.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

DATA_DIR = Path(__file__).resolve().parent / "data"
MODEL_FILE = DATA_DIR / "ngram_model.pkl"
MODEL_FILE_NOSPACE = DATA_DIR / "ngram_model_nospace.pkl"

# A = 27 when spaces are included (index 0 = space, 1-26 = a-z)
SPACE_CHAR = " "


def char_to_idx(ch: str, include_space: bool = True) -> int:
    """Map a character to its integer index."""
    if ch == " " and include_space:
        return 0
    return ord(ch) - ord("a") + (1 if include_space else 0)


def text_to_indices(text: str, include_space: bool = True) -> np.ndarray:
    """Convert a string to a NumPy int8 array of indices (vectorised)."""
    raw = np.frombuffer(text.encode("ascii"), dtype=np.uint8)
    arr = np.empty(len(raw), dtype=np.int8)
    if include_space:
        # space (32) → 0, letters → 1..26
        mask_space = raw == 32
        arr[:] = (raw - ord("a") + 1).astype(np.int8)
        arr[mask_space] = 0
    else:
        arr[:] = (raw - ord("a")).astype(np.int8)
    return arr


class NgramModel:
    """Character-level n-gram language model with interpolated smoothing."""

    def __init__(
        self,
        include_space: bool = True,
        lambdas: tuple[float, float, float, float] = (0.05, 0.10, 0.25, 0.60),
    ):
        self.include_space = include_space
        self.A = 27 if include_space else 26  # alphabet size
        self.lambdas = lambdas  # unigram, bigram, trigram, quadgram weights

        # Raw count tables
        self.uni = np.zeros(self.A, dtype=np.float64)
        self.bi = np.zeros((self.A, self.A), dtype=np.float64)
        self.tri = np.zeros((self.A, self.A, self.A), dtype=np.float64)
        self.quad = np.zeros((self.A, self.A, self.A, self.A), dtype=np.float64)

        # Log-probability tables (filled after train)
        self.log_uni: np.ndarray | None = None
        self.log_bi: np.ndarray | None = None
        self.log_tri: np.ndarray | None = None
        self.log_quad: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, corpus: str) -> None:
        """Count n-gram frequencies and compute log-probability tables."""
        idx = text_to_indices(corpus, self.include_space)
        n = len(idx)

        # Unigrams
        for i in range(n):
            self.uni[idx[i]] += 1

        # Bigrams
        for i in range(n - 1):
            self.bi[idx[i], idx[i + 1]] += 1

        # Trigrams
        for i in range(n - 2):
            self.tri[idx[i], idx[i + 1], idx[i + 2]] += 1

        # Quadgrams
        for i in range(n - 3):
            self.quad[idx[i], idx[i + 1], idx[i + 2], idx[i + 3]] += 1

        self._compute_log_probs()

    def _compute_log_probs(self) -> None:
        """Convert raw counts → smoothed log-probabilities."""
        A = self.A
        alpha = 1.0  # Laplace smoothing constant

        # Unigram
        total_uni = self.uni.sum() + alpha * A
        p_uni = (self.uni + alpha) / total_uni
        self.log_uni = np.log(p_uni)

        # Bigram – P(c2 | c1)
        denom_bi = self.uni[:, None] + alpha * A  # (A, 1)
        p_bi = (self.bi + alpha) / denom_bi
        self.log_bi = np.log(p_bi)

        # Trigram – P(c3 | c1, c2)
        denom_tri = self.bi[:, :, None] + alpha * A  # (A, A, 1)
        p_tri = (self.tri + alpha) / denom_tri
        self.log_tri = np.log(p_tri)

        # Quadgram – P(c4 | c1, c2, c3)
        denom_quad = self.tri[:, :, :, None] + alpha * A
        p_quad = (self.quad + alpha) / denom_quad
        self.log_quad = np.log(p_quad)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(self, text: str) -> float:
        """Return the total interpolated log-probability of *text*."""
        idx = text_to_indices(text, self.include_space)
        return self.score_indices(idx)

    def score_indices(self, idx: np.ndarray) -> float:
        """Score an already-indexed array (faster for hot loops)."""
        n = len(idx)
        if n == 0:
            return -1e18
        l1, l2, l3, l4 = self.lambdas
        log_uni = self.log_uni
        log_bi = self.log_bi
        log_tri = self.log_tri
        log_quad = self.log_quad
        assert log_uni is not None
        assert log_bi is not None
        assert log_tri is not None
        assert log_quad is not None

        total = 0.0
        for i in range(n):
            a = idx[i]
            p = l1 * np.exp(log_uni[a])
            if i >= 1:
                b = idx[i - 1]
                p += l2 * np.exp(log_bi[b, a])
            if i >= 2:
                c = idx[i - 2]
                b = idx[i - 1]
                p += l3 * np.exp(log_tri[c, b, a])
            if i >= 3:
                d = idx[i - 3]
                c = idx[i - 2]
                b = idx[i - 1]
                p += l4 * np.exp(log_quad[d, c, b, a])
            total += np.log(p) if p > 0 else -50.0
        return float(total)

    def score_quadgrams_fast(self, idx: np.ndarray) -> float:
        """Fast quadgram-only scoring (no interpolation). Used in MCMC hot loop.

        Fully vectorised with NumPy fancy-indexing — no Python loop."""
        n = len(idx)
        if n < 4:
            return self.score_indices(idx)
        log_quad = self.log_quad
        assert log_quad is not None
        return float(log_quad[idx[:-3], idx[1:-2], idx[2:-1], idx[3:]].sum())

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path | None = None) -> None:
        path = path or MODEL_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "include_space": self.include_space,
                    "lambdas": self.lambdas,
                    "log_uni": self.log_uni,
                    "log_bi": self.log_bi,
                    "log_tri": self.log_tri,
                    "log_quad": self.log_quad,
                    "A": self.A,
                },
                f,
            )

    @classmethod
    def load(cls, path: Path | None = None) -> "NgramModel":
        path = path or MODEL_FILE
        with open(path, "rb") as f:
            data = pickle.load(f)  # noqa: S301
        m = cls(include_space=data["include_space"], lambdas=data["lambdas"])
        m.log_uni = data["log_uni"]
        m.log_bi = data["log_bi"]
        m.log_tri = data["log_tri"]
        m.log_quad = data["log_quad"]
        m.A = data["A"]
        return m


def get_model(force_retrain: bool = False, include_space: bool = True) -> NgramModel:
    """Return a trained model, building it if necessary."""
    model_path = MODEL_FILE if include_space else MODEL_FILE_NOSPACE
    if model_path.exists() and not force_retrain:
        return NgramModel.load(model_path)

    from cipher.corpus import download_corpus

    corpus = download_corpus()
    if not include_space:
        corpus = corpus.replace(" ", "")

    model = NgramModel(include_space=include_space)

    from rich.progress import Progress

    label = "with spaces" if include_space else "no spaces"
    with Progress() as progress:
        task = progress.add_task(f"[green]Training n-gram model ({label})…", total=1)
        model.train(corpus)
        progress.advance(task)

    model.save(model_path)
    return model
