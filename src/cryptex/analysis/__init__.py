"""Evaluation and diagnostics utilities."""

from cryptex.analysis.adversarial import StressTestCase, StressTestResult, generate_stress_tests, run_stress_tests
from cryptex.analysis.convergence import chain_consensus, frequency_comparison_plot, key_confidence_heatmap, plot_score_trajectory, sparkline_trajectory
from cryptex.analysis.evaluation import BenchmarkEntry, PhaseTransitionResult, key_accuracy, plot_benchmark, plot_phase_transition, run_benchmark, run_phase_transition, symbol_error_rate

__all__ = [
    "BenchmarkEntry",
    "PhaseTransitionResult",
    "StressTestCase",
    "StressTestResult",
    "chain_consensus",
    "frequency_comparison_plot",
    "generate_stress_tests",
    "key_accuracy",
    "key_confidence_heatmap",
    "plot_benchmark",
    "plot_phase_transition",
    "plot_score_trajectory",
    "run_benchmark",
    "run_phase_transition",
    "run_stress_tests",
    "sparkline_trajectory",
    "symbol_error_rate",
]
