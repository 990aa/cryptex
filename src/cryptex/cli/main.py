"""Command-line interface for the Unsupervised Cipher Cracker.

Usage examples (all run through ``uv run``):
    uv run cryptex train                          # download corpus & train model
    uv run cryptex demo --cipher substitution     # encrypt & crack live demo
    uv run cryptex demo --method genetic          # demo with genetic algorithm
    uv run cryptex crack --cipher substitution --method mcmc --text "encrypted text"
    uv run cryptex crack --auto --file intercepted.txt
    uv run cryptex crack --cipher vigenere --text "vjg equkr dtqyp hqz"
    uv run cryptex analyse                        # convergence analysis with plots
    uv run cryptex detect --text "cipher text"    # detect cipher type
    uv run cryptex benchmark --trials 3           # benchmark MCMC vs GA vs baselines
    uv run cryptex phase-transition --trials 3    # success rate vs ciphertext length
    uv run cryptex stress-test                    # adversarial stress tests
    uv run cryptex historical                     # historical cipher challenges
    uv run cryptex language train --lang french   # train French language model
    uv run cryptex language detect --text "bonjour le monde"
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Protocol

# Ensure stdout can handle Unicode (Rich box-drawing chars) on Windows
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    stdout_reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(stdout_reconfigure):
        try:
            stdout_reconfigure(encoding="utf-8")
        except Exception:
            pass
    stderr_reconfigure = getattr(sys.stderr, "reconfigure", None)
    if callable(stderr_reconfigure):
        try:
            stderr_reconfigure(encoding="utf-8")
        except Exception:
            pass

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

console = Console()
interactive_console = Console(stderr=True)


class _ScoredResult(Protocol):
    best_score: float
    best_plaintext: str


@dataclass
class CrackOutput:
    cipher: str
    method: str
    key: str
    plaintext: str
    score: float
    elapsed: float
    timed_out: bool = False
    candidates: list[dict[str, object]] = field(default_factory=list)


def _parse_known_pairs(values: list[str] | None) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for raw in values or []:
        if ":" not in raw:
            raise ValueError(
                "Known pair must use '<plaintext_fragment>:<ciphertext_fragment>' format"
            )
        pt, ct = raw.split(":", 1)
        if not pt or not ct:
            raise ValueError("Known pair fragments must both be non-empty")
        pairs.append((pt, ct))
    return pairs


def _run_with_timeout(
    runner: Callable[[threading.Event | None], _ScoredResult],
    timeout: float | None,
) -> tuple[_ScoredResult, bool]:
    if timeout is None:
        return runner(None), False

    stop_event = threading.Event()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(runner, stop_event)
        try:
            return future.result(timeout=timeout), False
        except concurrent.futures.TimeoutError:
            stop_event.set()
            try:
                return future.result(timeout=10), True
            except concurrent.futures.TimeoutError as exc:
                raise TimeoutError(
                    "Solver did not stop cleanly before timeout grace window"
                ) from exc


def _result_candidates(result: _ScoredResult, top_k: int) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    if hasattr(result, "top_candidates"):
        top_candidates = getattr(result, "top_candidates")
        if isinstance(top_candidates, list):
            for key, plaintext, score in top_candidates[: max(1, top_k)]:
                candidates.append(
                    {
                        "key": str(key),
                        "plaintext": str(plaintext),
                        "score": float(score),
                    }
                )

    if not candidates:
        candidates.append(
            {
                "key": str(getattr(result, "best_key", "")),
                "plaintext": str(getattr(result, "best_plaintext", "")),
                "score": float(getattr(result, "best_score", float("-inf"))),
            }
        )
    return candidates[: max(1, top_k)]


def _emit_output(output: CrackOutput, output_format: str, top_k: int = 1) -> None:
    from cryptex.cli.display import print_result_box

    if output_format == "json":
        payload = {
            "cipher": output.cipher,
            "method": output.method,
            "key": output.key,
            "score": output.score,
            "elapsed_seconds": output.elapsed,
            "timed_out": output.timed_out,
            "plaintext": output.plaintext,
            "candidates": output.candidates[: max(1, top_k)],
        }
        print(json.dumps(payload, ensure_ascii=True, indent=2))
        return

    if output_format == "plain":
        if top_k <= 1 or len(output.candidates) <= 1:
            print(output.plaintext)
            return
        for idx, cand in enumerate(output.candidates[: max(1, top_k)], start=1):
            print(f"[{idx}] score={cand['score']:.2f} key={cand['key']}")
            print(cand["plaintext"])
            if idx < min(len(output.candidates), top_k):
                print()
        return

    extra_info = {"Timed Out": "yes" if output.timed_out else "no"}
    print_result_box(
        output.cipher,
        output.method,
        output.key,
        output.plaintext,
        output.score,
        output.elapsed,
        extra_info=extra_info,
    )

    if top_k > 1 and len(output.candidates) > 1:
        console.print("[bold cyan]Top candidates:[/bold cyan]")
        for idx, cand in enumerate(output.candidates[: max(1, top_k)], start=1):
            console.print(
                f"  {idx}. score={float(cand['score']):,.2f} key={cand['key']}"
            )


# Sub-commands


def cmd_train(args: argparse.Namespace) -> None:
    """Download corpus and train the n-gram model."""
    from cryptex.ngram import get_model

    console.print(
        "[bold cyan]Training language models (this may take a few minutes the first time)..."
    )
    t0 = time.time()
    model = get_model(force_retrain=args.force, include_space=True)
    model_ns = get_model(force_retrain=args.force, include_space=False)
    elapsed = time.time() - t0
    console.print(f"[bold green]OK Models ready  ({elapsed:.1f}s)")
    console.print(f"  Space model: alphabet={model.A}, weights={model.lambdas}")
    console.print(
        f"  No-space model: alphabet={model_ns.A}, weights={model_ns.lambdas}"
    )


def cmd_demo(args: argparse.Namespace) -> None:
    """Encrypt a sample text, then crack it live."""
    from cryptex.ciphers import get_engine
    from cryptex.ngram import get_model

    console.print("[bold cyan]Preparing demo...")
    model = get_model()

    cipher_name = args.cipher
    engine = get_engine(cipher_name)

    sample = (
        "it is a truth universally acknowledged that a single man in possession of a good "
        "fortune must be in want of a wife however little known the feelings or views of "
        "such a man may be on his first entering a neighbourhood this truth is so well fixed "
        "in the minds of the surrounding families that he is considered as the rightful "
        "property of some one or other of their daughters"
    )

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
    from cryptex.core.io import (
        discover_effective_alphabet,
        ghost_map_text,
        likely_homophonic_cipher,
        restore_ghost_text,
    )
    from cryptex.core.ngram import get_model
    from cryptex.languages import get_language_model

    if args.language == "english":
        model = get_model(include_space=True)
    else:
        model = get_language_model(args.language, include_space=True)

    if args.text:
        ciphertext = args.text
    elif args.file:
        try:
            with open(args.file, "r", encoding="utf-8") as f:
                ciphertext = f.read()
        except OSError as exc:
            console.print(f"[red]Error reading file '{args.file}': {exc}")
            sys.exit(1)
    else:
        console.print("[dim]Reading ciphertext from stdin (Ctrl-D / Ctrl-Z to end)...")
        ciphertext = sys.stdin.read()

    ciphertext = ciphertext.strip()
    if not ciphertext:
        console.print("[red]Error: empty ciphertext.")
        sys.exit(1)

    try:
        known_pairs = _parse_known_pairs(args.known_pair)
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)

    if args.auto:
        _crack_auto(
            ciphertext,
            model,
            output_format=args.output_format,
            timeout=args.timeout,
            top_k=args.top_k,
            language=args.language,
        )
        return

    cipher_name = args.cipher.lower().strip()
    postprocess_plaintext: Callable[[str], str] | None = None

    if cipher_name in ("substitution", "simple", "sub") or cipher_name.startswith(
        "noisy"
    ):
        mapping = ghost_map_text(ciphertext)
        ciphertext_core = mapping.core_text
        if not ciphertext_core:
            console.print("[red]Error: no decipherable alphabetic content found.")
            sys.exit(1)

        alphabet = discover_effective_alphabet(ciphertext)
        if likely_homophonic_cipher(ciphertext):
            console.print(
                "[yellow]Warning:[/yellow] Effective alphabet is large "
                f"({len(alphabet)} symbols). This may be homophonic substitution."
            )

        def _postprocess_plaintext(pt: str, ghost=mapping) -> str:
            return restore_ghost_text(pt, ghost)

        postprocess_plaintext = _postprocess_plaintext
        ciphertext = ciphertext_core.lower()
    else:
        ciphertext = ciphertext.lower()

    if (
        args.output_format == "terminal"
        and args.timeout is None
        and args.top_k <= 1
        and not known_pairs
    ):
        _crack(
            ciphertext,
            args.cipher,
            args.method,
            model,
            postprocess_plaintext=postprocess_plaintext,
            language=args.language,
        )
        return

    output = _crack_noninteractive(
        ciphertext,
        args.cipher,
        args.method,
        model,
        timeout=args.timeout,
        top_k=args.top_k,
        postprocess_plaintext=postprocess_plaintext,
        known_pairs=known_pairs,
        language=args.language,
    )
    _emit_output(output, args.output_format, top_k=args.top_k)


# Auto orchestration


def _crack_auto(ciphertext_raw: str, model) -> None:  # noqa: ANN001
    """Automatically detect cipher type and run one or more candidate solvers."""
    from cryptex.detector import detect_cipher_type
    from cryptex.display import print_result_box
    from cryptex.io import ghost_map_text, likely_homophonic_cipher, restore_ghost_text
    from cryptex.mcmc import MCMCConfig

    detected = detect_cipher_type(ciphertext_raw.lower())
    console.print(
        f"[bold cyan]Auto-detect:[/bold cyan] {detected.predicted_type} "
        f"({detected.confidence:.1%} confidence)"
    )

    if likely_homophonic_cipher(ciphertext_raw):
        console.print(
            "[yellow]Warning:[/yellow] symbol inventory suggests a possible "
            "homophonic cipher; falling back to available classical solvers."
        )

    if detected.confidence >= 0.80:
        cipher_name = detected.predicted_type
        if cipher_name == "substitution":
            ghost = ghost_map_text(ciphertext_raw)
            core = ghost.core_text.lower()
            if not core:
                console.print("[red]Error: no decipherable alphabetic content found.")
                return
            _crack(
                core,
                "substitution",
                "mcmc",
                model,
                postprocess_plaintext=lambda pt, g=ghost: restore_ghost_text(pt, g),
            )
            return

        _crack(ciphertext_raw.lower(), cipher_name, "mcmc", model)
        return

    console.print(
        "[bold cyan]Low confidence:[/bold cyan] running competitive solvers and selecting the best score."
    )

    candidates: list[tuple[str, _ScoredResult, str, str, float]] = []

    ghost = ghost_map_text(ciphertext_raw)
    if ghost.core_text:
        t0 = time.time()
        sub_result = _run_substitution_mcmc(
            ghost.core_text.lower(),
            model,
            config=MCMCConfig(
                iterations=25_000,
                num_restarts=4,
                track_trajectory=False,
                callback_interval=5000,
            ),
            callback=None,
        )
        sub_result.best_plaintext = restore_ghost_text(sub_result.best_plaintext, ghost)
        candidates.append(
            (
                "Simple Substitution",
                sub_result,
                "MCMC (Adaptive + Tempering)",
                sub_result.best_key,
                time.time() - t0,
            )
        )

    if ghost.core_text:
        t0 = time.time()
        hmm_result = _run_substitution_hmm(ghost.core_text.lower(), model)
        hmm_result.best_plaintext = restore_ghost_text(hmm_result.best_plaintext, ghost)
        candidates.append(
            (
                "Simple Substitution",
                hmm_result,
                "HMM/EM",
                hmm_result.best_key,
                time.time() - t0,
            )
        )

    t0 = time.time()
    vig_result = _run_vigenere(ciphertext_raw.lower(), model)
    candidates.append(
        (
            "Vigenere",
            vig_result,
            "IoC + Frequency Analysis",
            vig_result.best_key,
            time.time() - t0,
        )
    )

    if not candidates:
        console.print("[red]No viable candidates produced a result.")
        return

    winner_name, winner_result, winner_method, winner_key, elapsed = max(
        candidates,
        key=lambda entry: float(entry[1].best_score),
    )

    console.print("\n[bold green]Auto-selection complete.[/bold green]")
    for label, candidate, method, _key, runtime in candidates:
        console.print(
            f"  {label:<22} {method:<28} score={candidate.best_score:,.2f}  time={runtime:.1f}s"
        )

    print_result_box(
        winner_name,
        winner_method,
        str(winner_key),
        winner_result.best_plaintext,
        winner_result.best_score,
        elapsed,
    )


# Unified crack dispatcher


def _crack(
    ciphertext: str,
    cipher_name: str,
    method: str,
    model,
    postprocess_plaintext: Callable[[str], str] | None = None,
    language: str = "english",
) -> None:  # noqa: ANN001
    cipher_name = cipher_name.lower().strip()
    method = method.lower().strip()
    t0 = time.time()

    if cipher_name in ("substitution", "simple", "sub"):
        if method == "hmm":
            _crack_substitution_hmm(
                ciphertext,
                model,
                t0,
                postprocess_plaintext=postprocess_plaintext,
            )
        elif method == "genetic":
            _crack_substitution_genetic(
                ciphertext,
                model,
                t0,
                postprocess_plaintext=postprocess_plaintext,
            )
        else:
            _crack_substitution_mcmc(
                ciphertext,
                model,
                t0,
                postprocess_plaintext=postprocess_plaintext,
            )

    elif cipher_name in ("vigenere", "vig"):
        _crack_vigenere(ciphertext, model, t0)

    elif cipher_name in ("transposition", "columnar", "trans"):
        _crack_transposition(ciphertext, model, t0)

    elif cipher_name in ("playfair", "play"):
        _crack_playfair(ciphertext, model, t0, language=language)

    elif cipher_name in ("affine",):
        _crack_affine(ciphertext, model, t0)

    elif cipher_name in ("railfence", "rail-fence", "rail"):
        _crack_railfence(ciphertext, model, t0)

    elif cipher_name.startswith("noisy"):
        console.print("[dim]Noisy cipher detected - HMM handles noise naturally.")
        if method == "hmm":
            _crack_substitution_hmm(
                ciphertext,
                model,
                t0,
                postprocess_plaintext=postprocess_plaintext,
            )
        else:
            _crack_substitution_mcmc(
                ciphertext,
                model,
                t0,
                postprocess_plaintext=postprocess_plaintext,
            )

    else:
        console.print(f"[red]Unknown cipher type: {cipher_name}")
        sys.exit(1)


# Solver runners (non-interactive)


def _run_substitution_mcmc(  # noqa: ANN001
    ciphertext: str,
    model,
    config=None,
    callback=None,
    stop_event: threading.Event | None = None,
    fixed_mapping: dict[str, str] | None = None,
):
    from cryptex.mcmc import MCMCConfig, run_mcmc

    if config is None:
        config = MCMCConfig()
    return run_mcmc(
        ciphertext,
        model,
        config,
        callback=callback,
        stop_event=stop_event,
        fixed_mapping=fixed_mapping,
    )


def _run_substitution_hmm(  # noqa: ANN001
    ciphertext: str,
    model,
    config=None,
    callback=None,
    stop_event: threading.Event | None = None,
    noise_rate: float = 0.05,
):
    from cryptex.hmm import HMMConfig, run_hmm

    if config is None:
        config = HMMConfig()
    return run_hmm(
        ciphertext,
        model,
        config,
        callback=callback,
        noise_rate=noise_rate,
        stop_event=stop_event,
    )


def _run_substitution_genetic(  # noqa: ANN001
    ciphertext: str,
    model,
    config=None,
    callback=None,
    stop_event: threading.Event | None = None,
):
    from cryptex.genetic import GeneticConfig, run_genetic

    if config is None:
        config = GeneticConfig()
    return run_genetic(
        ciphertext,
        model,
        config,
        callback=callback,
        stop_event=stop_event,
    )


def _run_vigenere(
    ciphertext: str,
    model,
    config=None,
    callback=None,
    stop_event: threading.Event | None = None,
):  # noqa: ANN001
    from cryptex.vigenere_cracker import VigenereConfig, crack_vigenere

    if config is None:
        config = VigenereConfig()
    return crack_vigenere(
        ciphertext,
        model,
        config,
        callback=callback,
        stop_event=stop_event,
    )


def _run_transposition(  # noqa: ANN001
    ciphertext: str,
    model,
    config=None,
    callback=None,
    stop_event: threading.Event | None = None,
):
    from cryptex.transposition_cracker import TranspositionConfig, crack_transposition

    if config is None:
        config = TranspositionConfig()
    return crack_transposition(
        ciphertext,
        model,
        config,
        callback=callback,
        stop_event=stop_event,
    )


def _run_playfair(
    ciphertext: str,
    _model,
    config=None,
    callback=None,
    stop_event: threading.Event | None = None,
    language: str = "english",
):  # noqa: ANN001
    from cryptex.languages import get_language_model
    from cryptex.ngram import get_model
    from cryptex.playfair_cracker import PlayfairConfig, crack_playfair

    if language == "english":
        model_ns = get_model(include_space=False)
    else:
        model_ns = get_language_model(language, include_space=False)
    if config is None:
        config = PlayfairConfig()
    return crack_playfair(
        ciphertext,
        model_ns,
        config,
        callback=callback,
        stop_event=stop_event,
    )


def _run_affine(
    ciphertext: str,
    model,
    config=None,
    callback=None,
    stop_event: threading.Event | None = None,
):  # noqa: ANN001
    from cryptex.solvers.affine import AffineConfig, crack_affine

    if config is None:
        config = AffineConfig()
    return crack_affine(
        ciphertext,
        model,
        config,
        callback=callback,
        stop_event=stop_event,
    )


def _run_railfence(
    ciphertext: str,
    model,
    config=None,
    callback=None,
    stop_event: threading.Event | None = None,
):  # noqa: ANN001
    from cryptex.solvers.railfence import RailFenceConfig, crack_railfence

    if config is None:
        config = RailFenceConfig()
    return crack_railfence(
        ciphertext,
        model,
        config,
        callback=callback,
        stop_event=stop_event,
    )


# Substitution - MCMC


def _crack_substitution_mcmc(
    ciphertext: str,
    model,
    t0: float,
    postprocess_plaintext: Callable[[str], str] | None = None,
) -> None:  # noqa: ANN001
    from cryptex.display import MCMCDisplay, print_result_box
    from cryptex.mcmc import MCMCConfig

    config = MCMCConfig()
    display = MCMCDisplay(config.num_restarts, config.iterations, ciphertext)

    with Live(display.render(), console=interactive_console, refresh_per_second=8) as live:

        def cb(chain_idx, iteration, cur_key, cur_pt, cur_score, best_score, temp):
            display.callback(
                chain_idx, iteration, cur_key, cur_pt, cur_score, best_score, temp
            )
            live.update(display.render())

        result = _run_substitution_mcmc(ciphertext, model, config=config, callback=cb)

    if postprocess_plaintext is not None:
        result.best_plaintext = postprocess_plaintext(result.best_plaintext)

    elapsed = time.time() - t0
    print_result_box(
        "Simple Substitution",
        "MCMC (Adaptive + Tempering)",
        result.best_key,
        result.best_plaintext,
        result.best_score,
        elapsed,
    )


# Substitution - HMM


def _crack_substitution_hmm(
    ciphertext: str,
    model,
    t0: float,
    postprocess_plaintext: Callable[[str], str] | None = None,
) -> None:  # noqa: ANN001
    from cryptex.display import HMMDisplay, print_result_box
    from cryptex.hmm import HMMConfig

    config = HMMConfig()
    display = HMMDisplay()

    with Live(display.render(), console=interactive_console, refresh_per_second=4) as live:

        def cb(iteration, plaintext, ll):
            display.callback(iteration, plaintext, ll)
            live.update(display.render())

        result = _run_substitution_hmm(ciphertext, model, config=config, callback=cb)

    if postprocess_plaintext is not None:
        result.best_plaintext = postprocess_plaintext(result.best_plaintext)

    elapsed = time.time() - t0
    print_result_box(
        "Simple Substitution",
        "HMM/EM",
        result.best_key,
        result.best_plaintext,
        result.best_score,
        elapsed,
    )


# Vigenere


def _crack_vigenere(ciphertext: str, model, t0: float) -> None:  # noqa: ANN001
    from cryptex.display import GenericDisplay, print_result_box
    from cryptex.vigenere_cracker import VigenereConfig

    config = VigenereConfig()
    display = GenericDisplay("Vigenere Cracker")

    with Live(display.render(), console=interactive_console, refresh_per_second=4) as live:

        def cb(key, pt, score):
            display.update(f"Key={key}  score={score:,.1f}", pt, score)
            live.update(display.render())

        result = _run_vigenere(ciphertext, model, config=config, callback=cb)

    elapsed = time.time() - t0
    print_result_box(
        "Vigenere",
        "IoC + Frequency Analysis",
        result.best_key,
        result.best_plaintext,
        result.best_score,
        elapsed,
    )


# Columnar Transposition


def _crack_transposition(ciphertext: str, model, t0: float) -> None:  # noqa: ANN001
    from cryptex.display import GenericDisplay, print_result_box
    from cryptex.transposition_cracker import TranspositionConfig

    config = TranspositionConfig()
    display = GenericDisplay("Transposition Cracker")

    with Live(display.render(), console=interactive_console, refresh_per_second=4) as live:

        def cb(chain, it, pt, score, temp):
            display.update(
                f"Chain {chain} iter {it}  T={temp:.4f}  score={score:,.1f}", pt, score
            )
            live.update(display.render())

        result = _run_transposition(ciphertext, model, config=config, callback=cb)

    elapsed = time.time() - t0
    print_result_box(
        "Columnar Transposition",
        "MCMC",
        str(result.best_key),
        result.best_plaintext,
        result.best_score,
        elapsed,
    )


# Playfair


def _crack_playfair(ciphertext: str, model, t0: float) -> None:  # noqa: ANN001
    from cryptex.display import GenericDisplay, print_result_box
    from cryptex.playfair_cracker import PlayfairConfig

    config = PlayfairConfig()
    display = GenericDisplay("Playfair Cracker")

    with Live(display.render(), console=interactive_console, refresh_per_second=4) as live:

        def cb(chain, it, pt, score, temp):
            display.update(
                f"Chain {chain} iter {it}  T={temp:.4f}  score={score:,.1f}", pt, score
            )
            live.update(display.render())

        result = _run_playfair(ciphertext, model, config=config, callback=cb)

    elapsed = time.time() - t0
    print_result_box(
        "Playfair",
        "MCMC",
        result.best_key,
        result.best_plaintext,
        result.best_score,
        elapsed,
    )


def _crack_affine(ciphertext: str, model, t0: float) -> None:  # noqa: ANN001
    from cryptex.display import GenericDisplay, print_result_box
    from cryptex.solvers.affine import AffineConfig

    config = AffineConfig()
    display = GenericDisplay("Affine Solver")

    with Live(display.render(), console=interactive_console, refresh_per_second=4) as live:

        def cb(a, b, pt, score):
            display.update(f"a={a}, b={b}  score={score:,.1f}", pt, score)
            live.update(display.render())

        result = _run_affine(ciphertext, model, config=config, callback=cb)

    elapsed = time.time() - t0
    print_result_box(
        "Affine",
        "Brute Force + Adaptive Scoring",
        result.best_key,
        result.best_plaintext,
        result.best_score,
        elapsed,
    )


def _crack_railfence(ciphertext: str, model, t0: float) -> None:  # noqa: ANN001
    from cryptex.display import GenericDisplay, print_result_box
    from cryptex.solvers.railfence import RailFenceConfig

    config = RailFenceConfig()
    display = GenericDisplay("Rail Fence Solver")

    with Live(display.render(), console=interactive_console, refresh_per_second=4) as live:

        def cb(rails, pt, score):
            display.update(f"rails={rails}  score={score:,.1f}", pt, score)
            live.update(display.render())

        result = _run_railfence(ciphertext, model, config=config, callback=cb)

    elapsed = time.time() - t0
    print_result_box(
        "Rail Fence",
        "Rail Sweep + Adaptive Scoring",
        result.best_key,
        result.best_plaintext,
        result.best_score,
        elapsed,
    )


# Substitution - Genetic Algorithm


def _crack_substitution_genetic(
    ciphertext: str,
    model,
    t0: float,
    postprocess_plaintext: Callable[[str], str] | None = None,
) -> None:  # noqa: ANN001
    from cryptex.display import GenericDisplay, print_result_box
    from cryptex.genetic import GeneticConfig

    config = GeneticConfig()
    display = GenericDisplay("Genetic Algorithm")

    with Live(display.render(), console=interactive_console, refresh_per_second=4) as live:

        def cb(gen: int, key: str, pt: str, score: float) -> None:
            display.update(f"Gen {gen}  score={score:,.1f}", pt, score)
            live.update(display.render())

        result = _run_substitution_genetic(
            ciphertext, model, config=config, callback=cb
        )

    if postprocess_plaintext is not None:
        result.best_plaintext = postprocess_plaintext(result.best_plaintext)

    elapsed = time.time() - t0
    print_result_box(
        "Simple Substitution",
        "Genetic Algorithm",
        result.best_key,
        result.best_plaintext,
        result.best_score,
        elapsed,
    )


# Phase 2: Convergence analysis


def cmd_analyse(args: argparse.Namespace) -> None:
    """Run MCMC with full convergence analysis and diagnostic JSON reports."""
    from cryptex.ciphers import SimpleSubstitution
    from cryptex.convergence import (
        chain_consensus,
        frequency_comparison_plot,
        key_confidence_heatmap,
        plot_score_trajectory,
        sparkline_trajectory,
    )
    from cryptex.display import print_result_box
    from cryptex.mcmc import MCMCConfig, run_mcmc
    from cryptex.ngram import get_model

    console.print("[bold cyan]Running MCMC with convergence analysis...")
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

    from cryptex.display import MCMCDisplay

    display = MCMCDisplay(config.num_restarts, config.iterations, ciphertext)
    t0 = time.time()

    with Live(display.render(), console=interactive_console, refresh_per_second=8) as live:

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
    console.print(f"\n[bold cyan]Generating diagnostic reports in {out_dir}/...")

    ref_score = None
    if original:
        from cryptex.ngram import text_to_indices

        ref_score = model.score_quadgrams_fast(
            text_to_indices(original, model.include_space)
        )

    p1 = plot_score_trajectory(
        result, reference_score=ref_score, save_path=f"{out_dir}/convergence_plot.json"
    )
    if p1:
        console.print(f"  [green]Saved: {p1}")

    p2 = key_confidence_heatmap(result, save_path=f"{out_dir}/key_heatmap.json")
    if p2:
        console.print(f"  [green]Saved: {p2}")

    p3 = frequency_comparison_plot(
        ciphertext,
        result.best_plaintext,
        save_path=f"{out_dir}/frequency_comparison.json",
    )
    if p3:
        console.print(f"  [green]Saved: {p3}")

    if original:
        from cryptex.evaluation import symbol_error_rate

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


# Phase 2: Cipher type detection


def cmd_detect(args: argparse.Namespace) -> None:
    """Detect the cipher type of unknown ciphertext."""
    from rich.table import Table

    from cryptex.detector import detect_cipher_type

    if args.text:
        ciphertext = args.text.strip().lower()
    elif args.file:
        try:
            with open(args.file, "r", encoding="utf-8") as f:
                ciphertext = f.read().strip().lower()
        except OSError as exc:
            console.print(f"[red]Error reading file '{args.file}': {exc}")
            sys.exit(1)
    else:
        console.print("[dim]Reading ciphertext from stdin...")
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
    console.print(f"  Playfair score: {f.playfair_score:.3f}")
    console.print(f"  Length: {f.length}")


# Phase 2: Benchmark


def cmd_benchmark(args: argparse.Namespace) -> None:
    """Benchmark MCMC vs GA vs frequency analysis vs random restarts."""
    from rich.table import Table

    from cryptex.corpus import download_corpus
    from cryptex.evaluation import plot_benchmark, run_benchmark
    from cryptex.ngram import get_model

    model = get_model()
    console.print("[bold cyan]Running benchmark...")
    console.print(f"  Text length: {args.length}, Trials: {args.trials}")
    console.print(f"  Success threshold (SER): <= {args.success_threshold:.1%}")
    console.print(
        "  Methods: MCMC, Genetic Algorithm, Frequency Analysis, Random Restarts\n"
    )

    corpus = download_corpus()

    def cb(method: str, trial: int, ser: float, elapsed: float) -> None:
        console.print(
            f"  [dim]{method} trial {trial + 1}: SER={ser:.1%}, time={elapsed:.1f}s"
        )

    results = run_benchmark(
        model,
        corpus,
        text_length=args.length,
        trials=args.trials,
        success_threshold=args.success_threshold,
        callback=cb,
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
    path = plot_benchmark(
        results,
        save_path=f"{out}/benchmark.json",
        success_threshold=args.success_threshold,
    )
    if path:
        console.print(f"\n[green]Saved benchmark report: {path}")


# Phase 2: Phase transition


def cmd_phase_transition(args: argparse.Namespace) -> None:
    """Run phase transition analysis - success rate vs ciphertext length."""
    from rich.table import Table

    from cryptex.corpus import download_corpus
    from cryptex.evaluation import plot_phase_transition, run_phase_transition
    from cryptex.ngram import get_model

    model = get_model()
    corpus = download_corpus()

    lengths = [50, 75, 100, 125, 150, 200, 300, 500]
    console.print("[bold cyan]Running phase transition analysis...")
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
    path = plot_phase_transition(result, save_path=f"{out}/phase_transition.json")
    if path:
        console.print(f"\n[green]Saved phase transition report: {path}")


# Phase 2: Adversarial stress tests


def cmd_stress_test(_args: argparse.Namespace) -> None:
    """Run adversarial stress tests."""
    from rich.table import Table

    from cryptex.adversarial import generate_stress_tests, run_stress_tests
    from cryptex.ngram import get_model

    model = get_model()
    cases = generate_stress_tests()
    console.print(f"[bold cyan]Running {len(cases)} adversarial stress tests...\n")

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


# Phase 2: Historical ciphers


def cmd_historical(_args: argparse.Namespace) -> None:
    """Run historical cipher challenges."""
    from cryptex.historical import get_historical_ciphers, run_historical_challenge

    ciphers = get_historical_ciphers()
    console.print(
        f"[bold cyan]Running {len(ciphers)} historical cipher challenges...\n"
    )

    for hc in ciphers:
        console.print(f"[bold]{hc.name}")
        console.print(f"  [dim]{hc.description[:120]}...")
        console.print(f"  Type: {hc.cipher_type}, Difficulty: {hc.difficulty}")

        result = run_historical_challenge(hc)

        if "error" in result:
            console.print(f"  [red]Error: {result.get('error')}")
        else:
            time_seconds = result.get("time_seconds")
            score = result.get("score")
            time_text = (
                f"{time_seconds:.1f}s"
                if isinstance(time_seconds, (int, float))
                else "N/A"
            )
            console.print(
                f"  Time: {time_text}, Score: {score if score is not None else 'N/A'}"
            )
            ser = result.get("ser")
            if isinstance(ser, (int, float)):
                status = "[green]PASS" if bool(result.get("success")) else "[red]FAIL"
                console.print(f"  SER: {ser:.1%} {status}")
            decrypted = result.get("decrypted")
            preview = decrypted[:80] if isinstance(decrypted, str) else ""
            console.print(f"  Decrypted: {preview}...")
        console.print()


# Phase 2: Multi-language support


def cmd_language(args: argparse.Namespace) -> None:
    """Train language models or detect language."""
    if args.action == "train":
        from cryptex.languages import get_language_model

        console.print(f"[bold cyan]Training {args.lang} language model...")
        model = get_language_model(args.lang, force_retrain=args.force)
        console.print(f"[bold green]Done! Alphabet size: {model.A}")

    elif args.action == "detect":
        from cryptex.languages import detect_language

        text = args.text or sys.stdin.read()
        result, scores = detect_language(text)
        console.print(f"[bold]Detected language: [green]{result}")
        for lang, score in sorted(scores.items(), key=lambda x: x[1]):
            console.print(f"  {lang}: chi2 = {score:.4f}")

    elif args.action == "list":
        from cryptex.languages import get_available_languages

        for lang in get_available_languages():
            console.print(f"  {lang}")


def cmd_kpa(args: argparse.Namespace) -> None:
    """Run known-plaintext attack analysis on aligned windows."""
    from cryptex.solvers.kpa import known_plaintext_attack

    ciphertext = (args.ciphertext or "").strip().lower()
    known = (args.known or "").strip().lower()
    if not ciphertext or not known:
        console.print("[red]Error: both --ciphertext and --known are required.")
        sys.exit(1)

    result = known_plaintext_attack(ciphertext, known)
    console.print("[bold cyan]Known-plaintext attack results")
    console.print(f"  Substitution windows: {len(result.substitutions)}")
    console.print(f"  Caesar shifts: {result.caesar_shifts}")
    console.print(f"  Affine keys: {result.affine_keys}")
    console.print(f"  Vigenere keys: {result.vigenere_key_guesses}")


# Argument parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cryptex",
        description="Cryptex - classical cryptanalysis toolkit with adaptive solvers",
    )
    sub = parser.add_subparsers(dest="command")

    p_train = sub.add_parser(
        "train", help="Download corpus and train the language model"
    )
    p_train.add_argument(
        "--force", action="store_true", help="Force re-download and re-train"
    )

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
            "affine",
            "railfence",
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
            "affine",
            "railfence",
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
    p_crack.add_argument(
        "--auto",
        action="store_true",
        help="Auto-detect cipher and run competitive solver orchestration",
    )
    p_crack.add_argument("--text", "-t", type=str, help="Ciphertext string")
    p_crack.add_argument("--file", "-f", type=str, help="Read ciphertext from file")

    p_analyse = sub.add_parser(
        "analyse", help="Run MCMC with convergence analysis and diagnostic reports"
    )
    p_analyse.add_argument(
        "--text", "-t", type=str, help="Ciphertext (default: auto-generated demo)"
    )
    p_analyse.add_argument(
        "--output", "-o", type=str, default=".", help="Output directory for reports"
    )

    p_detect = sub.add_parser("detect", help="Detect cipher type from ciphertext")
    p_detect.add_argument("--text", "-t", type=str, help="Ciphertext string")
    p_detect.add_argument("--file", "-f", type=str, help="Read ciphertext from file")

    p_bench = sub.add_parser("benchmark", help="Benchmark MCMC vs GA vs baselines")
    p_bench.add_argument(
        "--length", type=int, default=300, help="Test ciphertext length (default: 300)"
    )
    p_bench.add_argument(
        "--trials", type=int, default=3, help="Trials per method (default: 3)"
    )
    p_bench.add_argument(
        "--success-threshold",
        type=float,
        default=0.20,
        help="Success threshold on SER (default: 0.20)",
    )
    p_bench.add_argument(
        "--output", "-o", type=str, default=".", help="Output directory for plots"
    )

    p_phase = sub.add_parser(
        "phase-transition", help="Analyse phase transition in ciphertext length"
    )
    p_phase.add_argument(
        "--trials", type=int, default=3, help="Trials per length (default: 3)"
    )
    p_phase.add_argument(
        "--output", "-o", type=str, default=".", help="Output directory for plots"
    )

    sub.add_parser("stress-test", help="Run adversarial stress tests")
    sub.add_parser("historical", help="Run historical cipher challenges")

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

    p_kpa = sub.add_parser("kpa", help="Known-plaintext attack helper")
    p_kpa.add_argument("--ciphertext", type=str, required=True, help="Ciphertext")
    p_kpa.add_argument("--known", type=str, required=True, help="Known plaintext")

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
    elif args.command == "kpa":
        cmd_kpa(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
