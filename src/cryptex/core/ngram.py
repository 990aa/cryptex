"""High-performance n-gram language model.

Implements either Laplace interpolation (legacy) or interpolated Kneser-Ney
for orders up to 5.  The module keeps the original APIs but extends them for
dirty Unicode input and faster scoring paths used by search solvers.
"""

from __future__ import annotations

import pickle
import unicodedata
import importlib
import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

_numba_spec = importlib.util.find_spec("numba")
if _numba_spec is not None:
    numba = importlib.import_module("numba")
    njit = numba.njit
    prange = numba.prange
    NUMBA_AVAILABLE = True
else:  # pragma: no cover - optional dependency
    NUMBA_AVAILABLE = False

    def njit(*_args, **_kwargs):  # type: ignore[misc]
        def _decorator(fn):
            return fn

        return _decorator

    def prange(*args, **kwargs):  # type: ignore[misc]
        return range(*args, **kwargs)


if TYPE_CHECKING:
    pass

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODEL_FILE = DATA_DIR / "ngram_model_v2.pkl"
MODEL_FILE_NOSPACE = DATA_DIR / "ngram_model_nospace_v2.pkl"
MODEL_VERSION = 2

# A = 27 when spaces are included (index 0 = space, 1-26 = a-z)
SPACE_CHAR = " "


@njit(cache=True, fastmath=True, parallel=True)
def _score_quadgrams_mapped_numba(
    ciphertext_idx: np.ndarray,
    inv_map: np.ndarray,
    log_quad_flat: np.ndarray,
    A: int,
) -> float:
    """Numba hot path: score ciphertext using mapping without materializing plaintext."""
    n = ciphertext_idx.shape[0]
    if n < 4:
        return 0.0

    total = 0.0
    for i in range(n - 3):
        a = int(inv_map[int(ciphertext_idx[i])])
        b = int(inv_map[int(ciphertext_idx[i + 1])])
        c = int(inv_map[int(ciphertext_idx[i + 2])])
        d = int(inv_map[int(ciphertext_idx[i + 3])])
        flat_idx = (((a * A) + b) * A + c) * A + d
        total += log_quad_flat[flat_idx]
    return total


def _ascii_letter_from_char(ch: str) -> str | None:
    """Best-effort map from Unicode character to a-z."""
    if not ch:
        return None
    if "a" <= ch <= "z":
        return ch
    if "A" <= ch <= "Z":
        return ch.lower()

    folded = unicodedata.normalize("NFKD", ch)
    for c in folded:
        if "a" <= c <= "z":
            return c
        if "A" <= c <= "Z":
            return c.lower()
    return None


def char_to_idx(ch: str, include_space: bool = True) -> int:
    """Map a character to its integer index."""
    if ch == " " and include_space:
        return 0
    mapped = _ascii_letter_from_char(ch)
    if mapped is None:
        raise ValueError(f"Character {ch!r} cannot be mapped into model alphabet")
    return ord(mapped) - ord("a") + (1 if include_space else 0)


def text_to_indices(text: str, include_space: bool = True) -> np.ndarray:
    """Convert text to an index array, tolerating Unicode and symbols.

    Non-letter symbols are ignored, except whitespace when ``include_space`` is True.
    """
    if not text:
        return np.empty(0, dtype=np.int8)

    if text.isascii():
        raw = np.frombuffer(text.encode("ascii"), dtype=np.uint8)
        raw = raw.copy()
        mask_upper = (raw >= ord("A")) & (raw <= ord("Z"))
        raw[mask_upper] += 32  # in-place to lowercase

        if include_space:
            mask_valid = ((raw >= ord("a")) & (raw <= ord("z"))) | (raw == 32)
            raw = raw[mask_valid]
            arr = np.empty(len(raw), dtype=np.int8)
            mask_space = raw == 32
            arr[:] = (raw - ord("a") + 1).astype(np.int8)
            arr[mask_space] = 0
            return arr

        mask_valid = (raw >= ord("a")) & (raw <= ord("z"))
        raw = raw[mask_valid]
        return (raw - ord("a")).astype(np.int8)

    out: list[int] = []
    for ch in text:
        mapped = _ascii_letter_from_char(ch)
        if mapped is not None:
            out.append(ord(mapped) - ord("a") + (1 if include_space else 0))
            continue
        if include_space and ch.isspace():
            out.append(0)

    if not out:
        return np.empty(0, dtype=np.int8)
    return np.asarray(out, dtype=np.int8)


class NgramModel:
    """Character-level n-gram language model."""

    VERSION = MODEL_VERSION

    def __init__(
        self,
        include_space: bool = True,
        lambdas: tuple[float, float, float, float] = (0.05, 0.10, 0.25, 0.60),
        max_order: int = 5,
        smoothing: str = "kneser_ney",
        discount: float = 0.75,
        prefer_numba: bool = True,
    ):
        self.include_space = include_space
        self.A = 27 if include_space else 26  # alphabet size
        self.lambdas = lambdas  # unigram, bigram, trigram, quadgram weights
        self.max_order = max(1, min(5, int(max_order)))
        self.smoothing = smoothing
        self.discount = discount
        self.use_numba = bool(prefer_numba and NUMBA_AVAILABLE)

        # Raw count tables
        self.uni = np.zeros(self.A, dtype=np.float32)
        self.bi = np.zeros((self.A, self.A), dtype=np.float32)
        self.tri = np.zeros((self.A, self.A, self.A), dtype=np.float32)
        self.quad = np.zeros((self.A, self.A, self.A, self.A), dtype=np.float32)
        self.penta: np.ndarray | None = None

        # Log-probability tables (filled after train)
        self.log_uni: np.ndarray | None = None
        self.log_bi: np.ndarray | None = None
        self.log_tri: np.ndarray | None = None
        self.log_quad: np.ndarray | None = None
        self.log_penta: np.ndarray | None = None
        self._log_quad_flat: np.ndarray | None = None

    # Training

    def train(self, corpus: str) -> None:
        """Count n-gram frequencies and compute log-probability tables."""
        idx = text_to_indices(corpus, self.include_space)
        self._count_ngrams(idx)
        self._compute_log_probs()

    def _count_ngrams(self, idx: np.ndarray) -> None:
        """Vectorised n-gram counting using NumPy indexed accumulation."""
        n = len(idx)
        A = self.A

        self.uni = np.bincount(idx, minlength=A).astype(np.float32, copy=False)
        self.bi = np.zeros((A, A), dtype=np.float32)
        self.tri = np.zeros((A, A, A), dtype=np.float32)
        self.quad = np.zeros((A, A, A, A), dtype=np.float32)
        self.penta = None

        if n >= 2:
            np.add.at(self.bi, (idx[:-1], idx[1:]), 1)

        if n >= 3:
            np.add.at(self.tri, (idx[:-2], idx[1:-1], idx[2:]), 1)

        if n >= 4:
            np.add.at(self.quad, (idx[:-3], idx[1:-2], idx[2:-1], idx[3:]), 1)

        if self.max_order >= 5 and n >= 5:
            self.penta = np.zeros((A, A, A, A, A), dtype=np.float32)
            np.add.at(
                self.penta,
                (idx[:-4], idx[1:-3], idx[2:-2], idx[3:-1], idx[4:]),
                1,
            )

    def _compute_log_probs(self) -> None:
        """Convert raw counts → smoothed log-probabilities."""
        if self.smoothing == "laplace":
            self._compute_log_probs_laplace()
        else:
            self._compute_log_probs_kneser_ney()

        self._prepare_hot_tables()

    def _prepare_hot_tables(self) -> None:
        """Build contiguous lookup tables for scorer hot paths."""
        if self.log_quad is not None:
            self._log_quad_flat = np.ascontiguousarray(self.log_quad.ravel())

    def _compute_log_probs_laplace(self) -> None:
        """Legacy Laplace smoothing for compatibility."""
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

        if self.max_order >= 5 and self.penta is not None:
            denom_penta = self.quad[:, :, :, :, None] + alpha * A
            p_penta = (self.penta + alpha) / denom_penta
            self.log_penta = np.log(p_penta)
        else:
            self.log_penta = None

    def _compute_log_probs_kneser_ney(self) -> None:
        """Interpolated Kneser-Ney smoothing up to order 5."""
        A = self.A
        d = float(self.discount)
        eps = 1e-12

        uni = self.uni.astype(np.float64, copy=False)
        bi = self.bi.astype(np.float64, copy=False)
        tri = self.tri.astype(np.float64, copy=False)
        quad = self.quad.astype(np.float64, copy=False)

        # Unigram continuation probability from bigram types.
        continuation = (bi > 0).sum(axis=0).astype(np.float64)
        cont_total = continuation.sum()
        if cont_total <= 0:
            p1 = np.full(A, 1.0 / A, dtype=np.float64)
        else:
            p1 = continuation / cont_total
        p1 = np.clip(p1, eps, None)
        p1 /= p1.sum()
        self.log_uni = np.log(p1)

        # Bigram KN: P(w | h1)
        discounted_bi = np.maximum(bi - d, 0.0)
        denom_bi = uni[:, None]
        p2 = np.zeros((A, A), dtype=np.float64)
        np.divide(discounted_bi, denom_bi, out=p2, where=denom_bi > 0)
        n1_bi = (bi > 0).sum(axis=1).astype(np.float64)
        lambda_bi = np.zeros(A, dtype=np.float64)
        np.divide(d * n1_bi, uni, out=lambda_bi, where=uni > 0)
        p2 += lambda_bi[:, None] * p1[None, :]

        unseen_bi = uni == 0
        if np.any(unseen_bi):
            p2[unseen_bi, :] = p1

        p2 = np.clip(p2, eps, None)
        p2 /= p2.sum(axis=1, keepdims=True)
        self.log_bi = np.log(p2)

        # Trigram KN: P(w | h1,h2)
        discounted_tri = np.maximum(tri - d, 0.0)
        denom_tri = bi[:, :, None]
        p3 = np.zeros((A, A, A), dtype=np.float64)
        np.divide(discounted_tri, denom_tri, out=p3, where=denom_tri > 0)

        n1_tri = (tri > 0).sum(axis=2).astype(np.float64)
        lambda_tri = np.zeros((A, A), dtype=np.float64)
        np.divide(d * n1_tri, bi, out=lambda_tri, where=bi > 0)
        p3 += lambda_tri[:, :, None] * p2[None, :, :]

        unseen_tri = bi == 0
        if np.any(unseen_tri):
            a_idx, b_idx = np.where(unseen_tri)
            p3[a_idx, b_idx, :] = p2[b_idx, :]

        p3 = np.clip(p3, eps, None)
        p3 /= p3.sum(axis=2, keepdims=True)
        self.log_tri = np.log(p3)

        # Quadgram KN: P(w | h1,h2,h3)
        discounted_quad = np.maximum(quad - d, 0.0)
        denom_quad = tri[:, :, :, None]
        p4 = np.zeros((A, A, A, A), dtype=np.float64)
        np.divide(discounted_quad, denom_quad, out=p4, where=denom_quad > 0)

        n1_quad = (quad > 0).sum(axis=3).astype(np.float64)
        lambda_quad = np.zeros((A, A, A), dtype=np.float64)
        np.divide(d * n1_quad, tri, out=lambda_quad, where=tri > 0)
        p4 += lambda_quad[:, :, :, None] * p3[None, :, :, :]

        unseen_quad = tri == 0
        if np.any(unseen_quad):
            a_idx, b_idx, c_idx = np.where(unseen_quad)
            p4[a_idx, b_idx, c_idx, :] = p3[b_idx, c_idx, :]

        p4 = np.clip(p4, eps, None)
        p4 /= p4.sum(axis=3, keepdims=True)
        self.log_quad = np.log(p4)

        # Pentagram KN: P(w | h1,h2,h3,h4)
        self.log_penta = None
        if self.max_order >= 5 and self.penta is not None:
            penta = self.penta.astype(np.float64, copy=False)
            discounted_penta = np.maximum(penta - d, 0.0)
            denom_penta = quad[:, :, :, :, None]
            p5 = np.zeros((A, A, A, A, A), dtype=np.float64)
            np.divide(discounted_penta, denom_penta, out=p5, where=denom_penta > 0)

            n1_penta = (penta > 0).sum(axis=4).astype(np.float64)
            lambda_penta = np.zeros((A, A, A, A), dtype=np.float64)
            np.divide(d * n1_penta, quad, out=lambda_penta, where=quad > 0)
            p5 += lambda_penta[:, :, :, :, None] * p4[None, :, :, :, :]

            unseen_penta = quad == 0
            if np.any(unseen_penta):
                a_idx, b_idx, c_idx, d_idx = np.where(unseen_penta)
                p5[a_idx, b_idx, c_idx, d_idx, :] = p4[b_idx, c_idx, d_idx, :]

            p5 = np.clip(p5, eps, None)
            p5 /= p5.sum(axis=4, keepdims=True)
            self.log_penta = np.log(p5)

    # Scoring

    def score(self, text: str) -> float:
        """Return the total interpolated log-probability of *text*."""
        idx = text_to_indices(text, self.include_space)
        return self.score_indices(idx)

    def score_indices(self, idx: np.ndarray) -> float:
        """Score an already-indexed array (faster for hot loops)."""
        if self.smoothing == "kneser_ney":
            return self._score_indices_kneser_ney(idx)
        return self._score_indices_laplace(idx)

    def _score_indices_kneser_ney(self, idx: np.ndarray) -> float:
        """Log-likelihood under the highest available KN conditional order."""
        n = len(idx)
        if n == 0:
            return -1e18

        log_uni = self.log_uni
        log_bi = self.log_bi
        log_tri = self.log_tri
        log_quad = self.log_quad
        log_penta = self.log_penta
        assert log_uni is not None
        assert log_bi is not None
        assert log_tri is not None
        assert log_quad is not None

        total = float(log_uni[idx[0]])
        if n >= 2:
            total += float(log_bi[idx[0], idx[1]])
        if n >= 3:
            total += float(log_tri[idx[0], idx[1], idx[2]])
        if n >= 4:
            total += float(log_quad[idx[0], idx[1], idx[2], idx[3]])

        if n <= 4:
            return total

        if log_penta is not None:
            total += float(
                log_penta[idx[:-4], idx[1:-3], idx[2:-2], idx[3:-1], idx[4:]].sum()
            )
            return total

        total += float(log_quad[idx[1:-3], idx[2:-2], idx[3:-1], idx[4:]].sum())
        return total

    def _score_indices_laplace(self, idx: np.ndarray) -> float:
        """Legacy interpolated scoring used before KN migration."""
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

    def score_bigrams(self, idx: np.ndarray) -> float:
        """Fast bigram chain score for short-text robustness."""
        n = len(idx)
        if n < 2:
            return self.score_indices(idx)
        log_bi = self.log_bi
        assert log_bi is not None
        return float(log_bi[idx[:-1], idx[1:]].sum())

    def score_trigrams(self, idx: np.ndarray) -> float:
        """Fast trigram chain score for medium-length text."""
        n = len(idx)
        if n < 3:
            return self.score_bigrams(idx)
        log_tri = self.log_tri
        assert log_tri is not None
        return float(log_tri[idx[:-2], idx[1:-1], idx[2:]].sum())

    def score_adaptive(self, idx: np.ndarray) -> float:
        """Length-adaptive language score to stabilize short ciphertext cracking."""
        n = len(idx)
        if n < 8:
            return self.score_bigrams(idx)
        if n < 25:
            return 0.4 * self.score_bigrams(idx) + 0.6 * self.score_trigrams(idx)
        if n < 60:
            return self.score_trigrams(idx)
        return self.score_quadgrams_fast(idx)

    def score_digraphs(self, idx: np.ndarray) -> float:
        """Digraph-pair score used by the Playfair solver."""
        if len(idx) < 2:
            return 0.0
        log_bi = self.log_bi
        assert log_bi is not None
        a = idx[0::2]
        b = idx[1::2]
        n = min(len(a), len(b))
        if n == 0:
            return 0.0
        return float(log_bi[a[:n], b[:n]].sum())

    def score_pentagrams_fast(self, idx: np.ndarray) -> float:
        """Fast pentagram-only scoring when a 5-gram table is available."""
        n = len(idx)
        if n < 5 or self.log_penta is None:
            return self.score_quadgrams_fast(idx)
        return float(
            self.log_penta[
                idx[:-4],
                idx[1:-3],
                idx[2:-2],
                idx[3:-1],
                idx[4:],
            ].sum()
        )

    def score_quadgrams_with_mapping(
        self, ciphertext_idx: np.ndarray, inv_map: np.ndarray
    ) -> float:
        """Score plaintext implied by ``inv_map[cipher_idx] -> plain_idx``.

        This avoids allocating a full plaintext index array every proposal.
        """
        if len(ciphertext_idx) < 4:
            plain_idx = inv_map[ciphertext_idx]
            return self.score_indices(plain_idx)

        log_quad_flat = self._log_quad_flat
        assert log_quad_flat is not None

        if self.use_numba:
            return float(
                _score_quadgrams_mapped_numba(
                    ciphertext_idx,
                    inv_map,
                    log_quad_flat,
                    self.A,
                )
            )

        plain_idx = inv_map[ciphertext_idx]
        return self.score_quadgrams_fast(plain_idx)

    # Persistence

    def save(self, path: Path | None = None) -> None:
        path = path or MODEL_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "version": MODEL_VERSION,
                    "VERSION": MODEL_VERSION,
                    "include_space": self.include_space,
                    "lambdas": self.lambdas,
                    "max_order": self.max_order,
                    "smoothing": self.smoothing,
                    "discount": self.discount,
                    "log_uni": self.log_uni,
                    "log_bi": self.log_bi,
                    "log_tri": self.log_tri,
                    "log_quad": self.log_quad,
                    "log_penta": self.log_penta,
                    "A": self.A,
                },
                f,
            )

    @classmethod
    def load(cls, path: Path | None = None) -> "NgramModel":
        path = path or MODEL_FILE
        with open(path, "rb") as f:
            data = pickle.load(f)  # noqa: S301
        if not isinstance(data, dict):
            raise ValueError("Invalid n-gram model file format")
        version = int(data.get("version", data.get("VERSION", -1)))
        if version != MODEL_VERSION:
            raise ValueError(f"Stale model version {version}; expected {MODEL_VERSION}")
        m = cls(
            include_space=data.get("include_space", True),
            lambdas=data.get("lambdas", (0.05, 0.10, 0.25, 0.60)),
            max_order=data.get("max_order", 4),
            smoothing=data.get("smoothing", "laplace"),
            discount=data.get("discount", 0.75),
        )
        m.log_uni = data["log_uni"]
        m.log_bi = data["log_bi"]
        m.log_tri = data["log_tri"]
        m.log_quad = data["log_quad"]
        m.log_penta = data.get("log_penta")
        m.A = data.get("A", 27 if m.include_space else 26)
        m._prepare_hot_tables()
        return m


def _load_model(path: Path) -> NgramModel | None:
    """Load a model if it exists and matches the current version."""
    if not path.exists():
        return None
    try:
        return NgramModel.load(path)
    except Exception:
        return None


def get_model(force_retrain: bool = False, include_space: bool = True) -> NgramModel:
    """Return a trained model, building it if necessary."""
    model_path = MODEL_FILE if include_space else MODEL_FILE_NOSPACE
    if not force_retrain:
        cached = _load_model(model_path)
        if cached is not None:
            return cached

    from cryptex.corpus import download_corpus

    corpus = download_corpus()
    if not include_space:
        corpus = corpus.replace(" ", "")

    model = NgramModel(
        include_space=include_space,
        max_order=5,
        smoothing="kneser_ney",
        discount=0.75,
    )

    from rich.progress import Progress

    label = "with spaces" if include_space else "no spaces"
    with Progress() as progress:
        task = progress.add_task(
            f"[green]Training {model.max_order}-gram {model.smoothing} model ({label})…",
            total=1,
        )
        model.train(corpus)
        progress.advance(task)

    model.save(model_path)
    return model
