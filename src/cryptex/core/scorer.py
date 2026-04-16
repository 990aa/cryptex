"""Unified scoring interface used by multiple solvers."""

from __future__ import annotations

from enum import Enum

import numpy as np

from cryptex.core.ngram import NgramModel, text_to_indices


class ScoreMode(str, Enum):
    ADAPTIVE = "adaptive"
    QUADGRAM = "quadgram"
    DIGRAPH = "digraph"
    FULL = "full"


class UnifiedScorer:
    """Thin adapter over NgramModel scoring paths."""

    def __init__(self, model: NgramModel, mode: ScoreMode = ScoreMode.ADAPTIVE):
        self.model = model
        self.mode = mode

    def score_text(self, text: str) -> float:
        idx = text_to_indices(text, include_space=self.model.include_space)
        return self.score_indices(idx)

    def score_indices(self, idx: np.ndarray) -> float:
        if self.mode == ScoreMode.ADAPTIVE:
            return self.model.score_adaptive(idx)
        if self.mode == ScoreMode.QUADGRAM:
            return self.model.score_quadgrams_fast(idx)
        if self.mode == ScoreMode.DIGRAPH:
            return self.model.score_digraphs(idx)
        return self.model.score_indices(idx)

    def score_mapping(self, ciphertext_idx: np.ndarray, inv_map: np.ndarray) -> float:
        if self.mode in (ScoreMode.ADAPTIVE, ScoreMode.QUADGRAM):
            return self.model.score_quadgrams_with_mapping(ciphertext_idx, inv_map)
        plain_idx = inv_map[ciphertext_idx]
        return self.score_indices(plain_idx)

