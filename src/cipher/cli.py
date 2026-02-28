"""Command-line interface for the Unsupervised Cipher Cracker.

Usage examples (all run through ``uv run``):
    uv run cipher crack --cipher substitution --method mcmc --text "encrypted text here"
    uv run cipher crack --cipher vigenere --text "vjg equkr dtqyp hqz"
    uv run cipher demo --cipher substitution
    uv run cipher train                          # force-(re)train the language model
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import time

# Ensure stdout can handle Unicode (Rich box-drawing chars) on Windows
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    if hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(encoding="utf-8")
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
    display = MCMCDisplay(config.num_restarts, config.iterations)

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
# Argument parser
# ------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cipher",
        description="Unsupervised Cipher Cracker — MCMC & HMM",
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
        choices=["mcmc", "hmm"],
        help="Solving method (default: mcmc).  HMM only for substitution/noisy.",
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
        choices=["mcmc", "hmm"],
        help="Solving method (default: mcmc)",
    )
    p_crack.add_argument("--text", "-t", type=str, help="Ciphertext string")
    p_crack.add_argument("--file", "-f", type=str, help="Read ciphertext from file")

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
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
