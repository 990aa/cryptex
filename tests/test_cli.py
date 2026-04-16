"""Tests for CLI (cli.py) — argument parsing and subcommand dispatch."""

from __future__ import annotations

import argparse

import pytest

from cryptex.cli import build_parser, main


class TestBuildParser:
    def test_returns_parser(self) -> None:
        parser = build_parser()
        assert isinstance(parser, argparse.ArgumentParser)

    def test_train_subcommand(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["train"])
        assert args.command == "train"
        assert args.force is False

    def test_train_force_flag(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["train", "--force"])
        assert args.force is True

    def test_demo_defaults(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["demo"])
        assert args.command == "demo"
        assert args.cipher == "substitution"
        assert args.method == "mcmc"

    def test_demo_cipher_choices(self) -> None:
        parser = build_parser()
        for c in (
            "substitution",
            "vigenere",
            "transposition",
            "playfair",
            "noisy-substitution",
        ):
            args = parser.parse_args(["demo", "--cipher", c])
            assert args.cipher == c

    def test_demo_method_choices(self) -> None:
        parser = build_parser()
        for m in ("mcmc", "hmm", "genetic"):
            args = parser.parse_args(["demo", "--method", m])
            assert args.method == m

    def test_crack_with_text(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["crack", "--text", "abc"])
        assert args.command == "crack"
        assert args.text == "abc"

    def test_crack_with_file(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["crack", "--file", "input.txt"])
        assert args.file == "input.txt"

    def test_crack_auto_flag(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["crack", "--auto", "--text", "abc"])
        assert args.auto is True
        assert args.text == "abc"

    def test_crack_cipher_and_method(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["crack", "-c", "vigenere", "-m", "hmm", "-t", "xyz"])
        assert args.cipher == "vigenere"
        assert args.method == "hmm"

    def test_crack_language_flag(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["crack", "--text", "abc", "--language", "french"])
        assert args.language == "french"

    def test_crack_output_format(self) -> None:
        parser = build_parser()
        args = parser.parse_args([
            "crack",
            "--text",
            "abc",
            "--output-format",
            "json",
        ])
        assert args.output_format == "json"

    def test_crack_timeout_and_top_k(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            ["crack", "--text", "abc", "--timeout", "2.5", "--top-k", "3"]
        )
        assert args.timeout == 2.5
        assert args.top_k == 3

    def test_crack_known_pairs(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "crack",
                "--text",
                "abc",
                "--known-pair",
                "it:xy",
                "--known-pair",
                "is:za",
            ]
        )
        assert args.known_pair == ["it:xy", "is:za"]

    def test_analyse_defaults(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["analyse"])
        assert args.command == "analyse"
        assert args.text is None
        assert args.output == "."

    def test_analyse_with_options(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["analyse", "--text", "cipher", "--output", "/tmp"])
        assert args.text == "cipher"
        assert args.output == "/tmp"

    def test_detect_with_text(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["detect", "--text", "xyz"])
        assert args.command == "detect"
        assert args.text == "xyz"

    def test_detect_with_file(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["detect", "--file", "ct.txt"])
        assert args.file == "ct.txt"

    def test_benchmark_defaults(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["benchmark"])
        assert args.command == "benchmark"
        assert args.length == 300
        assert args.trials == 3
        assert args.output == "."

    def test_benchmark_with_options(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            ["benchmark", "--length", "500", "--trials", "5", "-o", "out"]
        )
        assert args.length == 500
        assert args.trials == 5
        assert args.output == "out"

    def test_phase_transition_defaults(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["phase-transition"])
        assert args.command == "phase-transition"
        assert args.trials == 3

    def test_stress_test(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["stress-test"])
        assert args.command == "stress-test"

    def test_historical(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["historical"])
        assert args.command == "historical"

    def test_language_train(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["language", "train", "--lang", "german"])
        assert args.command == "language"
        assert args.action == "train"
        assert args.lang == "german"

    def test_language_detect(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["language", "detect", "--text", "bonjour"])
        assert args.action == "detect"
        assert args.text == "bonjour"

    def test_language_list(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["language", "list"])
        assert args.action == "list"

    def test_language_defaults(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["language", "train"])
        assert args.lang == "french"  # default
        assert args.force is False

    def test_no_subcommand(self) -> None:
        parser = build_parser()
        args = parser.parse_args([])
        assert args.command is None


class TestMainDispatch:
    """Test that main() dispatches without crashing for trivial cases."""

    def test_no_command_prints_help(self, capsys) -> None:
        """main() with no args should print help (not raise)."""
        main([])

    def test_invalid_command_raises(self) -> None:
        """Unknown subcommand should cause SystemExit from argparse."""
        with pytest.raises(SystemExit):
            main(["nonexistent_command"])

