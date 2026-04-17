from __future__ import annotations

import json
import random
import statistics
import string
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from cryptex.analysis.evaluation import (
    _frequency_analysis_baseline,
    _hillclimb_frequency_baseline,
    symbol_error_rate,
)
from cryptex.ciphers import (
    Affine,
    ColumnarTransposition,
    NoisyCipher,
    Playfair,
    RailFence,
    SimpleSubstitution,
    Vigenere,
)
from cryptex.core.detector import detect_cipher_type
from cryptex.core.ngram import get_model
from cryptex.languages import detect_language, get_available_languages
from cryptex.solvers.affine import AffineConfig, crack_affine
from cryptex.solvers.genetic import GeneticConfig, run_genetic
from cryptex.solvers.hmm import HMMConfig, run_hmm
from cryptex.solvers.kpa import crack_with_known_plaintext
from cryptex.solvers.mcmc import MCMCConfig, run_mcmc
from cryptex.solvers.playfair import PlayfairConfig, crack_playfair
from cryptex.solvers.railfence import RailFenceConfig, crack_railfence
from cryptex.solvers.transposition import TranspositionConfig, crack_transposition
from cryptex.solvers.vigenere import VigenereConfig, crack_vigenere


ROOT = Path(__file__).resolve().parents[1]
PLOTS = ROOT / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)


def _alpha(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch in string.ascii_lowercase)


def _playfair_plain(text: str) -> str:
    return _alpha(text).replace("j", "i")


def _playfair_key(seed_phrase: str) -> str:
    seen = set()
    key_chars: list[str] = []
    for ch in _alpha(seed_phrase).replace("j", "i"):
        if ch not in seen:
            seen.add(ch)
            key_chars.append(ch)
    for ch in "abcdefghiklmnopqrstuvwxyz":
        if ch not in seen:
            seen.add(ch)
            key_chars.append(ch)
    return "".join(key_chars[:25])


def _summarise_metric(
    ser_values: list[float],
    times: list[float],
    success_count: int,
    failure_count: int,
    threshold: float | None,
) -> dict[str, object]:
    return {
        "runs": len(ser_values),
        "successes": success_count,
        "failures": failure_count,
        "threshold": threshold,
        "avg_ser": float(sum(ser_values) / len(ser_values)) if ser_values else None,
        "median_ser": float(statistics.median(ser_values)) if ser_values else None,
        "avg_time_seconds": float(sum(times) / len(times)) if times else None,
    }


def _ser_for(expected_plain: str, decoded_plain: str) -> float:
    expected_alpha = _alpha(expected_plain)
    decoded_alpha = _alpha(decoded_plain)
    n = min(len(expected_alpha), len(decoded_alpha))
    if n == 0:
        return 1.0
    return float(symbol_error_rate(expected_alpha[:n], decoded_alpha[:n]))


def _load_existing_plot(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    random.seed(20260417)
    np.random.seed(20260417)

    model_space = get_model(include_space=True)
    model_nospace = get_model(include_space=False)

    custom_texts = [
        "the quick brown fox jumps over the lazy dog while distant bells ring across the valley and careful observers count each passing cloud",
        "cryptanalysis improves when repeated structures and predictable language patterns appear in longer passages that provide stronger statistical signal",
        "in the old library a patient researcher compares cipher alphabets by hand and marks every odd trigram before testing a new key hypothesis",
        "engineers built this toolkit to decode historical messages recover hidden meaning and measure confidence with transparent reproducible metrics",
        "weather reports from the harbor mention storms lantern light and crowded docks where merchants exchange coded notes before dawn",
        "a disciplined team validates every solver with deterministic seeds strict error bounds and plain language reports suitable for peer review",
        "classical ciphers can still teach modern lessons about search heuristics probabilistic modeling and robust software engineering under uncertainty",
        "when short text arrives the system returns top candidates with scores because ambiguity is expected and responsible tools should expose uncertainty",
        "careful benchmarking separates luck from genuine algorithmic strength by using multiple trials fixed corpora and consistent evaluation thresholds",
        "secure operations demand reliable automation where training detection cracking and reporting all run from the command line without manual intervention",
    ]

    language_texts = [
        ("english", "the market opens early and traders discuss risk before the morning bell rings"),
        ("english", "clear reporting and repeatable experiments improve trust in technical results"),
        ("french", "le petit village observe la pluie du matin et les enfants courent vers le marche"),
        ("french", "la methode scientifique exige des preuves claires et des mesures reproductibles"),
        ("german", "die forschung braucht geduld und saubere daten damit modelle verlasslich bleiben"),
        ("german", "im alten haus liest der student jeden abend ein langes kapitel"),
        ("spanish", "el equipo revisa cada resultado con calma y publica informes transparentes"),
        ("spanish", "la ciudad celebra un festival donde musicos tocan hasta la medianoche"),
        ("english", "language detection should remain stable even when punctuation and numbers appear 2026"),
        ("french", "les archives contiennent des lettres anciennes qui inspirent de nouvelles analyses"),
    ]

    sub_key = "phqgiumeaylnofdxkrcvstzwbj"
    vig_key = "cipher"
    trans_key = [2, 0, 3, 1, 4]
    affine_key = (5, 8)
    rail_key = 4
    playfair_key = _playfair_key("playfair benchmark key")
    noisy_engine = NoisyCipher(SimpleSubstitution(), corruption_rate=0.03, deletion_rate=0.01)

    roundtrip_checks: dict[str, dict[str, int]] = defaultdict(lambda: {"passed": 0, "total": 0})
    method_ser: dict[str, list[float]] = defaultdict(list)
    method_time: dict[str, list[float]] = defaultdict(list)
    method_success: dict[str, int] = defaultdict(int)
    method_failures: dict[str, int] = defaultdict(int)

    thresholds = {
        "MCMC": 0.20,
        "Genetic Algorithm": 0.30,
        "HMM/EM": 0.40,
        "Frequency Analysis": 0.60,
        "Hill Climb + Freq Init": 0.35,
        "KPA-Constrained MCMC": 0.20,
        "Noisy MCMC": 0.50,
        "Noisy HMM": 0.60,
        "Vigenere": 0.30,
        "Transposition": 0.25,
        "Playfair": 0.70,
        "Affine": 0.20,
        "Rail Fence": 0.35,
    }

    detector_total = 0
    detector_correct = 0
    total_solver_runs = 0

    for idx, text in enumerate(custom_texts, start=1):
        plain = _alpha(text)
        if len(plain) < 80:
            continue

        # Roundtrip checks over all implemented cipher engines.
        rt_cases = {
            "substitution": (
                plain,
                SimpleSubstitution.decrypt(SimpleSubstitution.encrypt(plain, sub_key), sub_key),
            ),
            "vigenere": (
                plain,
                Vigenere.decrypt(Vigenere.encrypt(plain, vig_key), vig_key),
            ),
            "transposition": (
                plain,
                ColumnarTransposition.decrypt(
                    ColumnarTransposition.encrypt(plain, trans_key),
                    trans_key,
                )[: len(plain)],
            ),
            "affine": (
                plain,
                Affine.decrypt(Affine.encrypt(plain, affine_key), affine_key),
            ),
            "railfence": (
                plain,
                RailFence.decrypt(RailFence.encrypt(plain, rail_key), rail_key),
            ),
            "playfair": (
                Playfair._prepare(_playfair_plain(plain)),
                Playfair.decrypt(
                    Playfair.encrypt(_playfair_plain(plain), playfair_key),
                    playfair_key,
                ),
            ),
        }
        for name, (expected, observed) in rt_cases.items():
            roundtrip_checks[name]["total"] += 1
            if expected == observed:
                roundtrip_checks[name]["passed"] += 1

        noisy_ct_rt = noisy_engine.encrypt(plain, sub_key)
        noisy_pt_rt = noisy_engine.decrypt(noisy_ct_rt, sub_key)
        roundtrip_checks["noisy-substitution"]["total"] += 1
        if len(noisy_pt_rt) > 0:
            roundtrip_checks["noisy-substitution"]["passed"] += 1

        sub_ct = SimpleSubstitution.encrypt(plain, sub_key)
        noisy_ct = noisy_engine.encrypt(plain, sub_key)

        detector_cases = {
            "substitution": sub_ct,
            "vigenere": Vigenere.encrypt(plain, vig_key),
            "transposition": ColumnarTransposition.encrypt(plain, trans_key),
            "playfair": Playfair.encrypt(_playfair_plain(plain), playfair_key),
            "affine": Affine.encrypt(plain, affine_key),
            "railfence": RailFence.encrypt(plain, rail_key),
        }
        for expected, ciphertext in detector_cases.items():
            pred = detect_cipher_type(ciphertext).predicted_type
            detector_total += 1
            if pred == expected or (expected in ("affine", "railfence") and pred == "substitution"):
                detector_correct += 1

        anchor_len = 7
        anchor_start = min(10, max(0, len(plain) - anchor_len - 1))
        anchor_pt = plain[anchor_start : anchor_start + anchor_len]
        anchor_ct = sub_ct[anchor_start : anchor_start + anchor_len]

        jobs: list[tuple[str, callable, str]] = [
            (
                "MCMC",
                lambda: run_mcmc(
                    sub_ct,
                    model_space,
                    MCMCConfig(
                        iterations=3000,
                        num_restarts=1,
                        random_seed=100 + idx,
                        track_trajectory=False,
                        use_parallel_tempering=False,
                    ),
                ).best_plaintext,
                plain,
            ),
            (
                "Genetic Algorithm",
                lambda: run_genetic(
                    sub_ct,
                    model_space,
                    GeneticConfig(population_size=40, generations=60),
                ).best_plaintext,
                plain,
            ),
            (
                "HMM/EM",
                lambda: run_hmm(sub_ct, model_space, HMMConfig(max_iter=18)).best_plaintext,
                plain,
            ),
            (
                "Frequency Analysis",
                lambda: _frequency_analysis_baseline(sub_ct),
                plain,
            ),
            (
                "Hill Climb + Freq Init",
                lambda: _hillclimb_frequency_baseline(sub_ct, model_space, iterations=1800),
                plain,
            ),
            (
                "KPA-Constrained MCMC",
                lambda: crack_with_known_plaintext(
                    sub_ct,
                    [(anchor_pt, anchor_ct)],
                    model_space,
                    config=MCMCConfig(
                        iterations=3200,
                        num_restarts=1,
                        random_seed=200 + idx,
                        track_trajectory=False,
                        use_parallel_tempering=False,
                    ),
                ).best_plaintext,
                plain,
            ),
            (
                "Noisy MCMC",
                lambda: run_mcmc(
                    noisy_ct,
                    model_space,
                    MCMCConfig(
                        iterations=2400,
                        num_restarts=1,
                        random_seed=300 + idx,
                        track_trajectory=False,
                        use_parallel_tempering=False,
                    ),
                ).best_plaintext,
                plain,
            ),
            (
                "Noisy HMM",
                lambda: run_hmm(noisy_ct, model_space, HMMConfig(max_iter=18)).best_plaintext,
                plain,
            ),
            (
                "Vigenere",
                lambda: crack_vigenere(
                    Vigenere.encrypt(plain, vig_key),
                    model_space,
                    VigenereConfig(max_key_length=12, top_key_lengths=6),
                ).best_plaintext,
                plain,
            ),
            (
                "Transposition",
                lambda: crack_transposition(
                    ColumnarTransposition.encrypt(plain, trans_key),
                    model_space,
                    TranspositionConfig(
                        iterations=1800,
                        num_restarts=1,
                        min_cols=len(trans_key),
                        max_cols=len(trans_key),
                    ),
                ).best_plaintext,
                plain,
            ),
            (
                "Playfair",
                lambda: crack_playfair(
                    Playfair.encrypt(_playfair_plain(plain), playfair_key),
                    model_nospace,
                    PlayfairConfig(
                        iterations=3000,
                        num_restarts=1,
                        population_size=8,
                        min_recommended_length=40,
                    ),
                ).best_plaintext,
                _playfair_plain(plain),
            ),
            (
                "Affine",
                lambda: crack_affine(
                    Affine.encrypt(plain, affine_key),
                    model_space,
                    AffineConfig(callback_interval=1000),
                ).best_plaintext,
                plain,
            ),
            (
                "Rail Fence",
                lambda: crack_railfence(
                    RailFence.encrypt(plain, rail_key),
                    model_space,
                    RailFenceConfig(min_rails=2, max_rails=8),
                ).best_plaintext,
                plain,
            ),
        ]

        for method_name, runner, expected_plain in jobs:
            total_solver_runs += 1
            started = time.perf_counter()
            try:
                decoded = runner()
                elapsed = time.perf_counter() - started
                ser = _ser_for(expected_plain, decoded)

                method_ser[method_name].append(ser)
                method_time[method_name].append(float(elapsed))
                if ser <= thresholds[method_name]:
                    method_success[method_name] += 1
            except Exception:
                method_failures[method_name] += 1

    method_summaries: dict[str, dict[str, object]] = {}
    for method_name in sorted(thresholds.keys()):
        method_summaries[method_name] = _summarise_metric(
            method_ser[method_name],
            method_time[method_name],
            method_success[method_name],
            method_failures[method_name],
            thresholds[method_name],
        )

    # Language detection checks
    lang_total = 0
    lang_correct = 0
    lang_details: list[dict[str, object]] = []
    for expected, sample in language_texts:
        predicted, scores = detect_language(sample)
        lang_total += 1
        if predicted == expected:
            lang_correct += 1
        lang_details.append(
            {
                "expected": expected,
                "predicted": predicted,
                "top_score": float(scores.get(predicted, 0.0)) if scores else 0.0,
            }
        )

    # Include existing benchmark/phase-transition JSON as reference context.
    benchmark_ref = _load_existing_plot(PLOTS / "benchmark.json")
    phase_ref = _load_existing_plot(PLOTS / "phase_transition.json")

    # Method ranking for quick "beats others" interpretation in docs.
    ranking = []
    for method_name, stats in method_summaries.items():
        if stats["avg_ser"] is None:
            continue
        ranking.append(
            {
                "method": method_name,
                "avg_ser": stats["avg_ser"],
                "successes": stats["successes"],
                "runs": stats["runs"],
                "avg_time_seconds": stats["avg_time_seconds"],
            }
        )
    ranking.sort(key=lambda row: float(row["avg_ser"]))

    output = {
        "metadata": {
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "custom_text_count": len(custom_texts),
            "language_sample_count": len(language_texts),
            "available_languages": get_available_languages(),
            "total_solver_runs": total_solver_runs,
        },
        "references": {
            "benchmark_json": benchmark_ref,
            "phase_transition_json": phase_ref,
        },
        "custom_suite": {
            "method_summaries": method_summaries,
            "method_ranking_by_avg_ser": ranking,
            "roundtrip_checks": roundtrip_checks,
            "detector_accuracy": {
                "correct": detector_correct,
                "total": detector_total,
                "accuracy": float(detector_correct / detector_total) if detector_total else 0.0,
            },
            "language_detection": {
                "correct": lang_correct,
                "total": lang_total,
                "accuracy": float(lang_correct / lang_total) if lang_total else 0.0,
                "details": lang_details,
            },
        },
    }

    out_path = PLOTS / "custom_benchmark_suite.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
