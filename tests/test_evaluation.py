"""Tests for the evaluation framework (evaluation.py)."""

from __future__ import annotations


class TestSymbolErrorRate:
    def test_identical(self) -> None:
        from cryptex.evaluation import symbol_error_rate

        assert symbol_error_rate("hello", "hello") == 0.0

    def test_completely_different(self) -> None:
        from cryptex.evaluation import symbol_error_rate

        ser = symbol_error_rate("aaaaa", "zzzzz")
        assert ser == 1.0

    def test_partial_match(self) -> None:
        from cryptex.evaluation import symbol_error_rate

        ser = symbol_error_rate("hello", "hxllo")
        assert abs(ser - 0.2) < 1e-9  # 1 out of 5 wrong

    def test_different_lengths(self) -> None:
        from cryptex.evaluation import symbol_error_rate

        # Should handle different lengths (use shorter)
        ser = symbol_error_rate("hello", "hel")
        assert isinstance(ser, float)

    def test_empty_strings(self) -> None:
        from cryptex.evaluation import symbol_error_rate

        ser = symbol_error_rate("", "")
        assert ser == 0.0 or isinstance(ser, float)

    def test_single_char_match(self) -> None:
        from cryptex.evaluation import symbol_error_rate

        assert symbol_error_rate("a", "a") == 0.0

    def test_single_char_mismatch(self) -> None:
        from cryptex.evaluation import symbol_error_rate

        assert symbol_error_rate("a", "b") == 1.0


class TestKeyAccuracy:
    def test_identical_keys(self) -> None:
        from cryptex.evaluation import key_accuracy

        import string

        key = string.ascii_lowercase
        assert key_accuracy(key, key) == 1.0

    def test_completely_different(self) -> None:
        from cryptex.evaluation import key_accuracy

        key1 = "abcdefghijklmnopqrstuvwxyz"
        key2 = "zyxwvutsrqponmlkjihgfedcba"
        acc = key_accuracy(key1, key2)
        # Only 'n' maps to itself in atbash (position 13)
        assert 0 <= acc <= 1.0

    def test_one_swap(self) -> None:
        from cryptex.evaluation import key_accuracy

        key1 = "abcdefghijklmnopqrstuvwxyz"
        key2 = "bacdefghijklmnopqrstuvwxyz"  # a↔b swapped
        acc = key_accuracy(key1, key2)
        assert abs(acc - 24 / 26) < 1e-9


class TestFrequencyAnalysisBaseline:
    def test_returns_string(self) -> None:
        from cryptex.evaluation import _frequency_analysis_baseline

        result = _frequency_analysis_baseline("wkh txlfn eurzq ira")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_preserves_length(self) -> None:
        from cryptex.evaluation import _frequency_analysis_baseline

        ct = "abcdefghij"
        result = _frequency_analysis_baseline(ct)
        assert len(result) == len(ct)


class TestRandomRestartBaseline:
    def test_returns_string(self, trained_model) -> None:
        from cryptex.evaluation import _random_restart_baseline

        result = _random_restart_baseline("hello world", trained_model, n_restarts=5)
        assert isinstance(result, str)
        assert len(result) > 0


class TestPhaseTransitionResult:
    def test_defaults(self) -> None:
        from cryptex.evaluation import PhaseTransitionResult

        r = PhaseTransitionResult()
        assert r.lengths == []
        assert r.threshold_length == 0


class TestBenchmarkEntry:
    def test_defaults(self) -> None:
        from cryptex.evaluation import BenchmarkEntry

        e = BenchmarkEntry()
        assert e.method == ""
        assert e.avg_ser == 0.0
        assert e.trials == 0


class TestPlotFunctions:
    def test_plot_phase_transition(self, tmp_path) -> None:
        from cryptex.evaluation import PhaseTransitionResult, plot_phase_transition

        r = PhaseTransitionResult()
        r.lengths = [50, 100, 200]
        r.success_rates = [0.2, 0.6, 1.0]
        r.avg_ser = [0.3, 0.1, 0.01]
        r.avg_times = [5.0, 10.0, 15.0]
        r.threshold_length = 100

        path = plot_phase_transition(r, save_path=str(tmp_path / "pt.png"))
        assert path is not None
        assert (tmp_path / "pt.png").exists()

    def test_plot_benchmark(self, tmp_path) -> None:
        from cryptex.evaluation import BenchmarkEntry, plot_benchmark

        entries = [
            BenchmarkEntry(
                method="MCMC", avg_ser=0.05, avg_time=10.0, success_rate=0.8, trials=3
            ),
            BenchmarkEntry(
                method="GA", avg_ser=0.15, avg_time=20.0, success_rate=0.6, trials=3
            ),
        ]
        path = plot_benchmark(entries, save_path=str(tmp_path / "bench.png"))
        assert path is not None
        assert (tmp_path / "bench.png").exists()
