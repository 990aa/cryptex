"""Tests for convergence analysis (convergence.py)."""

from __future__ import annotations


import pytest


@pytest.fixture()
def mock_mcmc_result():
    """Create a mock MCMCResult with realistic data."""
    from cryptex.solvers.mcmc import MCMCResult

    r = MCMCResult()
    r.best_key = "zyxwvutsrqponmlkjihgfedcba"
    r.best_plaintext = "the quick brown fox"
    r.best_score = -500.0
    r.chain_scores = [-600.0, -550.0, -500.0, -520.0]
    r.chain_keys = [
        "zyxwvutsrqponmlkjihgfedcba",
        "zyxwvutsrqponmlkjihgfedcba",
        "zyxwvutsrqponmlkjihgfedcba",
        "abcdefghijklmnopqrstuvwxyz",
    ]
    r.score_trajectory = [(i * 100, -1000 + i * 50) for i in range(20)]
    r.acceptance_rates = [0.3, 0.25, 0.2, 0.15]
    return r


class TestSparklineTrajectory:
    def test_returns_string(self, mock_mcmc_result) -> None:
        from cryptex.analysis.convergence import sparkline_trajectory

        s = sparkline_trajectory(mock_mcmc_result)
        assert isinstance(s, str)
        assert len(s) > 0

    def test_width_parameter(self, mock_mcmc_result) -> None:
        from cryptex.analysis.convergence import sparkline_trajectory

        short = sparkline_trajectory(mock_mcmc_result, width=10)
        long = sparkline_trajectory(mock_mcmc_result, width=50)
        assert len(short) <= len(long) + 5  # Approximate

    def test_empty_trajectory(self) -> None:
        from cryptex.analysis.convergence import sparkline_trajectory
        from cryptex.solvers.mcmc import MCMCResult

        r = MCMCResult()
        s = sparkline_trajectory(r)
        assert isinstance(s, str)


class TestChainConsensus:
    def test_returns_dict(self, mock_mcmc_result) -> None:
        from cryptex.analysis.convergence import chain_consensus

        result = chain_consensus(mock_mcmc_result)
        assert "agreement_rate" in result
        assert "consensus_key" in result
        assert "per_letter_confidence" in result
        assert "num_chains" in result

    def test_agreement_rate_range(self, mock_mcmc_result) -> None:
        from cryptex.analysis.convergence import chain_consensus

        result = chain_consensus(mock_mcmc_result)
        rate = result["agreement_rate"]
        assert isinstance(rate, (int, float))
        assert 0.0 <= rate <= 1.0

    def test_perfect_agreement(self) -> None:
        from cryptex.analysis.convergence import chain_consensus
        from cryptex.solvers.mcmc import MCMCResult

        r = MCMCResult()
        key = "abcdefghijklmnopqrstuvwxyz"
        r.chain_keys = [key, key, key]
        r.chain_scores = [-100, -100, -100]
        result = chain_consensus(r)
        assert result["agreement_rate"] == 1.0

    def test_no_chains(self) -> None:
        from cryptex.analysis.convergence import chain_consensus
        from cryptex.solvers.mcmc import MCMCResult

        r = MCMCResult()
        result = chain_consensus(r)
        assert result["num_chains"] == 0


class TestPlotScoreTrajectory:
    def test_saves_file(self, mock_mcmc_result, tmp_path) -> None:
        from cryptex.analysis.convergence import plot_score_trajectory

        save_path = tmp_path / "test_trajectory.png"
        result = plot_score_trajectory(mock_mcmc_result, save_path=str(save_path))
        assert result is not None
        assert save_path.exists()

    def test_with_reference_score(self, mock_mcmc_result, tmp_path) -> None:
        from cryptex.analysis.convergence import plot_score_trajectory

        save_path = tmp_path / "test_ref.png"
        plot_score_trajectory(
            mock_mcmc_result, reference_score=-400.0, save_path=str(save_path)
        )
        assert save_path.exists()


class TestKeyConfidenceHeatmap:
    def test_saves_file(self, mock_mcmc_result, tmp_path) -> None:
        from cryptex.analysis.convergence import key_confidence_heatmap

        save_path = tmp_path / "test_heatmap.png"
        result = key_confidence_heatmap(mock_mcmc_result, save_path=str(save_path))
        assert result is not None
        assert save_path.exists()

    def test_no_chains(self, tmp_path) -> None:
        from cryptex.analysis.convergence import key_confidence_heatmap
        from cryptex.solvers.mcmc import MCMCResult

        r = MCMCResult()
        result = key_confidence_heatmap(r, save_path=str(tmp_path / "empty.png"))
        # Should handle gracefully
        assert result is None or (tmp_path / "empty.png").exists()


class TestFrequencyComparisonPlot:
    def test_saves_file(self, tmp_path) -> None:
        from cryptex.analysis.convergence import frequency_comparison_plot

        save_path = tmp_path / "test_freq.png"
        result = frequency_comparison_plot(
            "wkh txlfn eurzq ira", "the quick brown fox", save_path=str(save_path)
        )
        assert result is not None
        assert save_path.exists()
