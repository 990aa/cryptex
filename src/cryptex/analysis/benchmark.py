"""Expanded benchmark orchestration for cryptex."""

from __future__ import annotations

from dataclasses import dataclass, field

from cryptex.analysis.evaluation import BenchmarkEntry, run_benchmark
from cryptex.core.ngram import NgramModel


@dataclass
class BenchmarkSuiteResult:
    lengths: list[int] = field(default_factory=list)
    per_length: dict[int, list[BenchmarkEntry]] = field(default_factory=dict)


def run_benchmark_suite(
    model: NgramModel,
    corpus: str,
    lengths: list[int] | None = None,
    trials: int = 3,
    success_threshold: float = 0.20,
) -> BenchmarkSuiteResult:
    """Run benchmark across multiple ciphertext lengths."""
    if lengths is None:
        lengths = [80, 120, 180, 260]

    result = BenchmarkSuiteResult(lengths=list(lengths))
    for length in lengths:
        result.per_length[length] = run_benchmark(
            model,
            corpus,
            text_length=length,
            trials=trials,
            success_threshold=success_threshold,
        )
    return result

