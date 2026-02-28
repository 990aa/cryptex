"""Command-line interface for the Unsupervised Cipher Cracker.

Usage examples (all run through ``uv run``):
    uv run cipher train                          # download corpus & train model
    uv run cipher demo --cipher substitution     # encrypt & crack live demo
    uv run cipher demo --method genetic          # demo with genetic algorithm
    uv run cipher crack --cipher substitution --method mcmc --text "encrypted text"
    uv run cipher crack --cipher vigenere --text "vjg equkr dtqyp hqz"
    uv run cipher analyse                        # convergence analysis with plots
    uv run cipher detect --text "cipher text"    # detect cipher type
    uv run cipher benchmark --trials 3           # benchmark MCMC vs GA vs baselines
    uv run cipher phase-transition --trials 3    # success rate vs ciphertext length
    uv run cipher stress-test                    # adversarial stress tests
    uv run cipher historical                     # historical cipher challenges
    uv run cipher language train --lang french   # train French language model
    uv run cipher language detect --text "bonjour le monde"
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# Ensure stdout can handle Unicode (Rich box-drawing chars) on Windows
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[call-non-callable]
        except Exception:
            pass
    if hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[call-non-callable]
        except Exception:
            pass

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

console = Console(force_terminal=True)


# ------------------------------------------------------------------
# Sub-commands
# ------------------------------------------------------------------


def cmd_train(args: argparse.Namespace) -> None:
    """Download corpus and train the n-gram model."""
    from cipher.ngram import get_model

    console.print(
        "[bold cyan]Training language models (this may take a few minutes the first time)…"
    )
    t0 = time.time()
    model = get_model(force_retrain=args.force, include_space=True)
    model_ns = get_model(force_retrain=args.force, include_space=False)
    elapsed = time.time() - t0
    console.print(f"[bold green]✓ Models ready  ({elapsed:.1f}s)")
    console.print(f"  Space model: alphabet={model.A}, weights={model.lambdas}")
    console.print(
        f"  No-space model: alphabet={model_ns.A}, weights={model_ns.lambdas}"
    )


def cmd_demo(args: argparse.Namespace) -> None:
    """Encrypt a sample text, then crack it live."""
    from cipher.ciphers import get_engine
    from cipher.ngram import get_model

    console.print("[bold cyan]Preparing demo…")
    model = get_model()

    cipher_name = args.cipher
    engine = get_engine(cipher_name)

    # Sample plaintext
    sample = (
        "it is a truth universally acknowledged that a single man in possession of a good "
        "fortune must be in want of a wife however little known the feelings or views of "
        "such a man may be on his first entering a neighbourhood this truth is so well fixed "
        "in the minds of the surrounding families that he is considered as the rightful "
        "property of some one or other of their daughters"
    )

    # Playfair needs more text for statistical signal
    if cipher_name in ("playfair", "play"):
        sample = (
            "it is a truth universally acknowledged that a single man in possession of a "
            "good fortune must be in want of a wife however little known the feelings or "
            "views of such a man may be on his first entering a neighbourhood this truth "
            "is so well fixed in the minds of the surrounding families that he is considered "
            "as the rightful property of some one or other of their daughters my dear "
            "mister bennet said his lady to him one day have you heard that netherfield park "
            "is let at last mister bennet replied that he had not but it is returned she "
            "for missus long has just been here and she told me all about it mister bennet "
            "made no answer do not you want to know who has taken it cried his wife impatiently"
        )

    key = engine.random_key()
    ciphertext = engine.encrypt(sample, key)

    console.print()
    console.print(
        Panel(
            Text(ciphertext[:500], style="red"),
            title="[bold red]Ciphertext",
            border_style="red",
        )
    )
    console.print(f"[dim]True key: {key}")
    console.print()

    _crack(ciphertext, cipher_name, args.method, model)


def cmd_crack(args: argparse.Namespace) -> None:
    """Crack user-supplied ciphertext."""
    from cipher.ngram import get_model

    model = get_model()

    # Read ciphertext from --text or stdin
    if args.text:
        ciphertext = args.text
    elif args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            ciphertext = f.read()
    else:
        console.print("[dim]Reading ciphertext from stdin (Ctrl-D / Ctrl-Z to end)…")
        ciphertext = sys.stdin.read()

    ciphertext = ciphertext.strip().lower()
    if not ciphertext:
        console.print("[red]Error: empty ciphertext.")
        sys.exit(1)

    _crack(ciphertext, args.cipher, args.method, model)


# ------------------------------------------------------------------
# Unified crack dispatcher
# ------------------------------------------------------------------


def _crack(ciphertext: str, cipher_name: str, method: str, model) -> None:  # noqa: ANN001

    cipher_name = cipher_name.lower().strip()
    method = method.lower().strip()
    t0 = time.time()

    # ----- Simple Substitution -----
    if cipher_name in ("substitution", "simple", "sub"):
        if method == "hmm":
            _crack_substitution_hmm(ciphertext, model, t0)
        elif method == "genetic":
            _crack_substitution_genetic(ciphertext, model, t0)
        else:
            _crack_substitution_mcmc(ciphertext, model, t0)

    # ----- Vigenère -----
    elif cipher_name in ("vigenere", "vigenère", "vig"):
        _crack_vigenere(ciphertext, model, t0)

    # ----- Columnar Transposition -----
    elif cipher_name in ("transposition", "columnar", "trans"):
        _crack_transposition(ciphertext, model, t0)

    # ----- Playfair -----
    elif cipher_name in ("playfair", "play"):
        _crack_playfair(ciphertext, model, t0)

    # ----- Noisy Substitution -----
    elif cipher_name.startswith("noisy"):
        console.print("[dim]Noisy cipher detected — HMM handles noise naturally.")
        if method == "hmm":
            _crack_substitution_hmm(ciphertext, model, t0)
        else:
            # Default to MCMC even for noisy (still works, just less robust)
            _crack_substitution_mcmc(ciphertext, model, t0)

    else:
        console.print(f"[red]Unknown cipher type: {cipher_name}")
        sys.exit(1)


# ------------------------------------------------------------------
# Substitution — MCMC
# ------------------------------------------------------------------


def _crack_substitution_mcmc(ciphertext: str, model, t0: float) -> None:  # noqa: ANN001
    from cipher.display import MCMCDisplay, print_result_box
    from cipher.mcmc import MCMCConfig, run_mcmc

    config = MCMCConfig()
    display = MCMCDisplay(config.num_restarts, config.iterations, ciphertext)

    with Live(display.render(), console=console, refresh_per_second=8) as live:

        def cb(chain_idx, iteration, cur_key, cur_pt, cur_score, best_score, temp):
            display.callback(
                chain_idx, iteration, cur_key, cur_pt, cur_score, best_score, temp
            )
            live.update(display.render())

        result = run_mcmc(ciphertext, model, config, callback=cb)

    elapsed = time.time() - t0
    print_result_box(
        "Simple Substitution",
        "MCMC",
        result.best_key,
        result.best_plaintext,
        result.best_score,
        elapsed,
    )


# ------------------------------------------------------------------
# Substitution — HMM
# ------------------------------------------------------------------


def _crack_substitution_hmm(ciphertext: str, model, t0: float) -> None:  # noqa: ANN001
    from cipher.display import HMMDisplay, print_result_box
    from cipher.hmm import HMMConfig, run_hmm

    config = HMMConfig()
    display = HMMDisplay()

    with Live(display.render(), console=console, refresh_per_second=4) as live:

        def cb(iteration, plaintext, ll):
            display.callback(iteration, plaintext, ll)
            live.update(display.render())

        result = run_hmm(ciphertext, model, config, callback=cb)

    elapsed = time.time() - t0
    print_result_box(
        "Simple Substitution",
        "HMM/EM",
        result.best_key,
        result.best_plaintext,
        result.best_score,
        elapsed,
    )


# ------------------------------------------------------------------
# Vigenère
# ------------------------------------------------------------------


def _crack_vigenere(ciphertext: str, model, t0: float) -> None:  # noqa: ANN001
    from cipher.display import GenericDisplay, print_result_box
    from cipher.vigenere_cracker import VigenereConfig, crack_vigenere

    config = VigenereConfig()
    display = GenericDisplay("Vigenère Cracker")

    with Live(display.render(), console=console, refresh_per_second=4) as live:

        def cb(key, pt, score):
            display.update(f"Key={key}  score={score:,.1f}", pt, score)
            live.update(display.render())

        result = crack_vigenere(ciphertext, model, config, callback=cb)

    elapsed = time.time() - t0
    print_result_box(
        "Vigenère",
        "IoC + Frequency Analysis",
        result.best_key,
        result.best_plaintext,
        result.best_score,
        elapsed,
    )


# ------------------------------------------------------------------
# Columnar Transposition
# ------------------------------------------------------------------


def _crack_transposition(ciphertext: str, model, t0: float) -> None:  # noqa: ANN001
    from cipher.display import GenericDisplay, print_result_box
    from cipher.transposition_cracker import TranspositionConfig, crack_transposition

    config = TranspositionConfig()
    display = GenericDisplay("Transposition Cracker")

    with Live(display.render(), console=console, refresh_per_second=4) as live:

        def cb(chain, it, pt, score, temp):
            display.update(
                f"Chain {chain} iter {it}  T={temp:.4f}  score={score:,.1f}", pt, score
            )
            live.update(display.render())

        result = crack_transposition(ciphertext, model, config, callback=cb)

    elapsed = time.time() - t0
    print_result_box(
        "Columnar Transposition",
        "MCMC",
        str(result.best_key),
        result.best_plaintext,
        result.best_score,
        elapsed,
    )


# ------------------------------------------------------------------
# Playfair
# ------------------------------------------------------------------


def _crack_playfair(ciphertext: str, model, t0: float) -> None:  # noqa: ANN001
    from cipher.display import GenericDisplay, print_result_box
    from cipher.ngram import get_model
    from cipher.playfair_cracker import PlayfairConfig, crack_playfair

    # Playfair output has no spaces — use the no-space model for better discrimination
    model_ns = get_model(include_space=False)

    config = PlayfairConfig()
    display = GenericDisplay("Playfair Cracker")

    with Live(display.render(), console=console, refresh_per_second=4) as live:

        def cb(chain, it, pt, score, temp):
            display.update(
                f"Chain {chain} iter {it}  T={temp:.4f}  score={score:,.1f}", pt, score
            )
            live.update(display.render())

        result = crack_playfair(ciphertext, model_ns, config, callback=cb)

    elapsed = time.time() - t0
    print_result_box(
        "Playfair",
        "MCMC",
        result.best_key,
        result.best_plaintext,
        result.best_score,
        elapsed,
    )


# ------------------------------------------------------------------
# Substitution — Genetic Algorithm
# ------------------------------------------------------------------


def _crack_substitution_genetic(ciphertext: str, model, t0: float) -> None:  # noqa: ANN001
    from cipher.display import GenericDisplay, print_result_box
    from cipher.genetic import GeneticConfig, run_genetic

    config = GeneticConfig()
    display = GenericDisplay("Genetic Algorithm")

    with Live(display.render(), console=console, refresh_per_second=4) as live:

        def cb(gen: int, key: str, pt: str, score: float) -> None:
            display.update(f"Gen {gen}  score={score:,.1f}", pt, score)
            live.update(display.render())

        result = run_genetic(ciphertext, model, config, callback=cb)

    elapsed = time.time() - t0
    print_result_box(
        "Simple Substitution",
        "Genetic Algorithm",
        result.best_key,
        result.best_plaintext,
        result.best_score,
        elapsed,
    )


# ------------------------------------------------------------------
# Phase 2: Convergence analysis
# ------------------------------------------------------------------


def cmd_analyse(args: argparse.Namespace) -> None:
    """Run MCMC with full convergence analysis and diagnostic plots."""
    from cipher.ciphers import SimpleSubstitution
    from cipher.convergence import (
        chain_consensus,
        frequency_comparison_plot,
        key_confidence_heatmap,
        plot_score_trajectory,
        sparkline_trajectory,
    )
    from cipher.display import print_result_box
    from cipher.mcmc import MCMCConfig, run_mcmc
    from cipher.ngram import get_model

    console.print("[bold cyan]Running MCMC with convergence analysis…")
    model = get_model()

    if args.text:
        ciphertext = args.text.strip().lower()
        original = None
    else:
        sample = (
            "it is a truth universally acknowledged that a single man in possession of a good "
            "fortune must be in want of a wife however little known the feelings or views of "
            "such a man may be on his first entering a neighbourhood this truth is so well fixed "
            "in the minds of the surrounding families that he is considered as the rightful "
            "property of some one or other of their daughters"
        )
        key = SimpleSubstitution.random_key()
        ciphertext = SimpleSubstitution.encrypt(sample, key)
        original = sample
        console.print(f"[dim]True key: {key}")

    config = MCMCConfig(iterations=50_000, num_restarts=8, track_trajectory=True)

    from cipher.display import MCMCDisplay

    display = MCMCDisplay(config.num_restarts, config.iterations, ciphertext)
    t0 = time.time()

    with Live(display.render(), console=console, refresh_per_second=8) as live:

        def cb(chain_idx, iteration, cur_key, cur_pt, cur_score, best_score, temp):
            display.callback(
                chain_idx, iteration, cur_key, cur_pt, cur_score, best_score, temp
            )
            live.update(display.render())

        result = run_mcmc(ciphertext, model, config, callback=cb)

    elapsed = time.time() - t0

    console.print()
    console.print(f"[bold green]Completed in {elapsed:.1f}s")
    console.print(f"  Best score: {result.best_score:,.2f}")
    console.print(f"  Chains used: {len(result.chain_scores)}")
    console.print(f"  Score trajectory: {sparkline_trajectory(result)}")

    consensus = chain_consensus(result)
    console.print(f"  Chain agreement: {consensus['agreement_rate']:.1%}")

    out_dir = args.output or "."
    console.print(f"\n[bold cyan]Generating diagnostic plots in {out_dir}/…")

    ref_score = None
    if original:
        from cipher.ngram import text_to_indices

        ref_score = model.score_quadgrams_fast(
            text_to_indices(original, model.include_space)
        )

    p1 = plot_score_trajectory(
        result, reference_score=ref_score, save_path=f"{out_dir}/convergence_plot.png"
    )
    if p1:
        console.print(f"  [green]Saved: {p1}")

    p2 = key_confidence_heatmap(result, save_path=f"{out_dir}/key_heatmap.png")
    if p2:
        console.print(f"  [green]Saved: {p2}")

    p3 = frequency_comparison_plot(
        ciphertext,
        result.best_plaintext,
        save_path=f"{out_dir}/frequency_comparison.png",
    )
    if p3:
        console.print(f"  [green]Saved: {p3}")

    if original:
        from cipher.evaluation import symbol_error_rate

        ser = symbol_error_rate(original, result.best_plaintext)
        console.print(f"\n  [bold]Symbol Error Rate: {ser:.1%}")

    print_result_box(
        "Simple Substitution",
        "MCMC (with analysis)",
        result.best_key,
        result.best_plaintext,
        result.best_score,
        elapsed,
        extra_info={
            "Chains": len(result.chain_scores),
            "Agreement": f"{consensus['agreement_rate']:.1%}",
        },
    )


# ------------------------------------------------------------------
# Phase 2: Cipher type detection
# ------------------------------------------------------------------


def cmd_detect(args: argparse.Namespace) -> None:
    """Detect the cipher type of unknown ciphertext."""
    from rich.table import Table

    from cipher.detector import detect_cipher_type

    if args.text:
        ciphertext = args.text.strip().lower()
    elif args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            ciphertext = f.read().strip().lower()
    else:
        console.print("[dim]Reading ciphertext from stdin…")
        ciphertext = sys.stdin.read().strip().lower()

    if not ciphertext:
        console.print("[red]Error: empty ciphertext.")
        sys.exit(1)

    result = detect_cipher_type(ciphertext)

    table = Table(title="Cipher Type Detection", show_header=True)
    table.add_column("Type", style="bold")
    table.add_column("Score", justify="right")
    assert result.all_scores is not None
    for cipher_type, score in sorted(result.all_scores.items(), key=lambda x: -x[1]):
        style = "bold green" if cipher_type == result.predicted_type else ""
        table.add_row(cipher_type, f"{score:.1%}", style=style)

    console.print()
    console.print(table)
    console.print(
        f"\n[bold]Predicted: [green]{result.predicted_type}[/green] "
        f"({result.confidence:.1%} confidence)"
    )
    console.print(f"[dim]Reasoning: {result.reasoning}")

    f = result.features
    assert f is not None
    console.print("\n[bold cyan]Statistical Features:")
    console.print(f"  IoC: {f.ioc:.4f}  (English ~0.0667)")
    console.print(f"  Entropy: {f.entropy:.3f} bits")
    console.print(f"  Chi-squared: {f.chi_squared:.4f}")
    console.print(f"  Length: {f.length}")


# ------------------------------------------------------------------
# Phase 2: Benchmark
# ------------------------------------------------------------------


def cmd_benchmark(args: argparse.Namespace) -> None:
    """Benchmark MCMC vs GA vs frequency analysis vs random restarts."""
    from rich.table import Table

    from cipher.corpus import download_corpus
    from cipher.evaluation import plot_benchmark, run_benchmark
    from cipher.ngram import get_model

    model = get_model()
    console.print("[bold cyan]Running benchmark…")
    console.print(f"  Text length: {args.length}, Trials: {args.trials}")
    console.print(
        "  Methods: MCMC, Genetic Algorithm, Frequency Analysis, Random Restarts\n"
    )

    corpus = download_corpus()

    def cb(method: str, trial: int, ser: float, elapsed: float) -> None:
        console.print(
            f"  [dim]{method} trial {trial + 1}: SER={ser:.1%}, time={elapsed:.1f}s"
        )

    results = run_benchmark(
        model, corpus, text_length=args.length, trials=args.trials, callback=cb
    )

    table = Table(title="Benchmark Results", show_header=True)
    table.add_column("Method", style="bold")
    table.add_column("Avg SER", justify="right")
    table.add_column("Success Rate", justify="right")
    table.add_column("Avg Time (s)", justify="right")

    for entry in results:
        table.add_row(
            entry.method,
            f"{entry.avg_ser:.1%}",
            f"{entry.success_rate:.0%}",
            f"{entry.avg_time:.1f}",
        )

    console.print()
    console.print(table)

    out = args.output or "."
    path = plot_benchmark(results, save_path=f"{out}/benchmark.png")
    if path:
        console.print(f"\n[green]Saved benchmark plot: {path}")


# ------------------------------------------------------------------
# Phase 2: Phase transition
# ------------------------------------------------------------------


def cmd_phase_transition(args: argparse.Namespace) -> None:
    """Run phase transition analysis — success rate vs ciphertext length."""
    from rich.table import Table

    from cipher.corpus import download_corpus
    from cipher.evaluation import plot_phase_transition, run_phase_transition
    from cipher.ngram import get_model

    model = get_model()
    corpus = download_corpus()

    lengths = [50, 75, 100, 125, 150, 200, 300, 500]
    console.print("[bold cyan]Running phase transition analysis…")
    console.print(f"  Lengths: {lengths}")
    console.print(f"  Trials per length: {args.trials}\n")

    def cb(length: int, trial: int, ser: float, elapsed: float) -> None:
        status = "[green]OK" if ser < 0.05 else "[red]FAIL"
        console.print(
            f"  Length={length}, trial {trial + 1}: SER={ser:.1%} {status} [dim]({elapsed:.1f}s)"
        )

    result = run_phase_transition(
        model, corpus, lengths=lengths, trials_per_length=args.trials, callback=cb
    )

    console.print(
        f"\n[bold]Phase Transition Threshold: ~{result.threshold_length} characters"
    )

    table = Table(title="Phase Transition Results", show_header=True)
    table.add_column("Length", justify="right")
    table.add_column("Success Rate", justify="right")
    table.add_column("Avg SER", justify="right")
    table.add_column("Avg Time (s)", justify="right")

    for i, length in enumerate(result.lengths):
        table.add_row(
            str(length),
            f"{result.success_rates[i]:.0%}",
            f"{result.avg_ser[i]:.1%}",
            f"{result.avg_times[i]:.1f}",
        )
    console.print()
    console.print(table)

    out = args.output or "."
    path = plot_phase_transition(result, save_path=f"{out}/phase_transition.png")
    if path:
        console.print(f"\n[green]Saved phase transition plot: {path}")


# ------------------------------------------------------------------
# Phase 2: Adversarial stress tests
# ------------------------------------------------------------------


def cmd_stress_test(_args: argparse.Namespace) -> None:
    """Run adversarial stress tests."""
    from rich.table import Table

    from cipher.adversarial import generate_stress_tests, run_stress_tests
    from cipher.ngram import get_model

    model = get_model()
    cases = generate_stress_tests()
    console.print(f"[bold cyan]Running {len(cases)} adversarial stress tests…\n")

    def cb(name: str, result) -> None:  # noqa: ANN001
        status = "[green]PASS" if result.success else "[red]FAIL"
        console.print(
            f"  {status}  {name}: SER={result.ser:.1%}, "
            f"score={result.score:,.1f}, time={result.time_seconds:.1f}s"
        )

    results = run_stress_tests(model, cases, callback=cb)

    passed = sum(1 for r in results if r.success)
    console.print(f"\n[bold]Results: {passed}/{len(results)} passed")

    table = Table(title="Stress Test Results", show_header=True)
    table.add_column("Test Case", style="bold")
    table.add_column("SER", justify="right")
    table.add_column("Score", justify="right")
    table.add_column("Time (s)", justify="right")
    table.add_column("Status")

    for r in results:
        status = "[green]PASS" if r.success else "[red]FAIL"
        table.add_row(
            r.case_name,
            f"{r.ser:.1%}",
            f"{r.score:,.1f}",
            f"{r.time_seconds:.1f}",
            status,
        )

    console.print()
    console.print(table)


# ------------------------------------------------------------------
# Phase 2: Historical ciphers
# ------------------------------------------------------------------


def cmd_historical(_args: argparse.Namespace) -> None:
    """Run historical cipher challenges."""
    from cipher.historical import get_historical_ciphers, run_historical_challenge

    ciphers = get_historical_ciphers()
    console.print(f"[bold cyan]Running {len(ciphers)} historical cipher challenges…\n")

    for hc in ciphers:
        console.print(f"[bold]{hc.name}")
        console.print(f"  [dim]{hc.description[:120]}…")
        console.print(f"  Type: {hc.cipher_type}, Difficulty: {hc.difficulty}")

        result = run_historical_challenge(hc)

        if "error" in result:
            console.print(f"  [red]Error: {result['error']}")
        else:
            console.print(
                f"  Time: {result['time_seconds']:.1f}s, "
                f"Score: {result.get('score', 'N/A')}"
            )
            if result.get("ser") is not None:
                status = "[green]PASS" if result["success"] else "[red]FAIL"
                console.print(f"  SER: {result['ser']:.1%} {status}")
            console.print(f"  Decrypted: {result.get('decrypted', '')[:80]}…")
        console.print()


# ------------------------------------------------------------------
# Phase 2: Multi-language support
# ------------------------------------------------------------------


def cmd_language(args: argparse.Namespace) -> None:
    """Train language models or detect language."""
    if args.action == "train":
        from cipher.languages import get_language_model

        console.print(f"[bold cyan]Training {args.lang} language model…")
        model = get_language_model(args.lang, force_retrain=args.force)
        console.print(f"[bold green]Done! Alphabet size: {model.A}")

    elif args.action == "detect":
        from cipher.languages import detect_language

        text = args.text or sys.stdin.read()
        result, scores = detect_language(text)
        console.print(f"[bold]Detected language: [green]{result}")
        for lang, score in sorted(scores.items(), key=lambda x: x[1]):
            console.print(f"  {lang}: chi² = {score:.4f}")

    elif args.action == "list":
        from cipher.languages import get_available_languages

        for lang in get_available_languages():
            console.print(f"  {lang}")


# ------------------------------------------------------------------
# Argument parser
# ------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cipher",
        description="Unsupervised Cipher Cracker — MCMC, HMM, Genetic Algorithm & more",
    )
    sub = parser.add_subparsers(dest="command")

    # -- train --
    p_train = sub.add_parser("train", help="Download corpus & train the language model")
    p_train.add_argument(
        "--force", action="store_true", help="Force re-download and re-train"
    )

    # -- demo --
    p_demo = sub.add_parser("demo", help="Encrypt a sample text, then crack it live")
    p_demo.add_argument(
        "--cipher",
        "-c",
        default="substitution",
        choices=[
            "substitution",
            "vigenere",
            "transposition",
            "playfair",
            "noisy-substitution",
        ],
        help="Cipher type (default: substitution)",
    )
    p_demo.add_argument(
        "--method",
        "-m",
        default="mcmc",
        choices=["mcmc", "hmm", "genetic"],
        help="Solving method (default: mcmc)",
    )

    # -- crack --
    p_crack = sub.add_parser("crack", help="Crack user-supplied ciphertext")
    p_crack.add_argument(
        "--cipher",
        "-c",
        default="substitution",
        choices=[
            "substitution",
            "vigenere",
            "transposition",
            "playfair",
            "noisy-substitution",
        ],
        help="Cipher type",
    )
    p_crack.add_argument(
        "--method",
        "-m",
        default="mcmc",
        choices=["mcmc", "hmm", "genetic"],
        help="Solving method (default: mcmc)",
    )
    p_crack.add_argument("--text", "-t", type=str, help="Ciphertext string")
    p_crack.add_argument("--file", "-f", type=str, help="Read ciphertext from file")

    # -- analyse --
    p_analyse = sub.add_parser(
        "analyse", help="Run MCMC with convergence analysis & diagnostic plots"
    )
    p_analyse.add_argument(
        "--text", "-t", type=str, help="Ciphertext (default: auto-generated demo)"
    )
    p_analyse.add_argument(
        "--output", "-o", type=str, default=".", help="Output directory for plots"
    )

    # -- detect --
    p_detect = sub.add_parser("detect", help="Detect cipher type from ciphertext")
    p_detect.add_argument("--text", "-t", type=str, help="Ciphertext string")
    p_detect.add_argument("--file", "-f", type=str, help="Read ciphertext from file")

    # -- benchmark --
    p_bench = sub.add_parser("benchmark", help="Benchmark MCMC vs GA vs baselines")
    p_bench.add_argument(
        "--length", type=int, default=300, help="Test ciphertext length (default: 300)"
    )
    p_bench.add_argument(
        "--trials", type=int, default=3, help="Trials per method (default: 3)"
    )
    p_bench.add_argument(
        "--output", "-o", type=str, default=".", help="Output directory for plots"
    )

    # -- phase-transition --
    p_phase = sub.add_parser(
        "phase-transition", help="Analyse phase transition in ciphertext length"
    )
    p_phase.add_argument(
        "--trials", type=int, default=3, help="Trials per length (default: 3)"
    )
    p_phase.add_argument(
        "--output", "-o", type=str, default=".", help="Output directory for plots"
    )

    # -- stress-test --
    sub.add_parser("stress-test", help="Run adversarial stress tests")

    # -- historical --
    sub.add_parser("historical", help="Run historical cipher challenges")

    # -- language --
    p_lang = sub.add_parser("language", help="Multi-language support")
    p_lang.add_argument(
        "action",
        choices=["train", "detect", "list"],
        help="Action to perform",
    )
    p_lang.add_argument(
        "--lang", type=str, default="french", help="Language (french, german, spanish)"
    )
    p_lang.add_argument("--text", "-t", type=str, help="Text for language detection")
    p_lang.add_argument("--force", action="store_true", help="Force re-train")

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "train":
        cmd_train(args)
    elif args.command == "demo":
        cmd_demo(args)
    elif args.command == "crack":
        cmd_crack(args)
    elif args.command == "analyse":
        cmd_analyse(args)
    elif args.command == "detect":
        cmd_detect(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)
    elif args.command == "phase-transition":
        cmd_phase_transition(args)
    elif args.command == "stress-test":
        cmd_stress_test(args)
    elif args.command == "historical":
        cmd_historical(args)
    elif args.command == "language":
        cmd_language(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
