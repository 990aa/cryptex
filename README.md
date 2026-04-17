# Cryptex

Cryptex is a classical cryptanalysis toolkit for educational and research workflows.

It includes:
- Cipher engines for substitution, Vigenere, columnar transposition, Playfair, affine, and rail fence.
- Cracking solvers for all major supported cipher families, including constrained known-plaintext mode.
- Adaptive n-gram scoring with Kneser-Ney language models.
- Auto-detection and competitive solver orchestration.
- Evaluation utilities for convergence, benchmarking, phase-transition behavior, stress tests, and historical challenge packs.

## Requirements

- Python 3.14+
- uv package manager: https://docs.astral.sh/uv/getting-started/installation/

## Installation

```bash
uv sync --dev
```

## Quick Start

```bash
# Train language models (first run downloads corpus)
uv run cryptex train

# Live substitution demo
uv run cryptex demo --cipher substitution --method mcmc

# Crack unknown text with automatic routing
uv run cryptex crack --auto --text "xqzv 9988 @@ lxfopv ef rnhr -- ???"

# Run test suite
uv run pytest -q
```

## What Is Implemented

### Cipher engines

- `substitution`: monoalphabetic substitution.
- `vigenere`: polyalphabetic Vigenere cipher.
- `transposition`: columnar transposition cipher.
- `playfair`: digraph Playfair cipher.
- `affine`: affine monoalphabetic transformation.
- `railfence`: rail fence transposition.
- `noisy-substitution`: substitution wrapped with corruption/deletion noise.

### Solvers and analyzers

- `MCMC`: adaptive simulated annealing with optional tempering and crossover proposals.
- `HMM/EM`: noisy-substitution tolerant probabilistic decoding.
- `Genetic`: population-based substitution search for comparison.
- `Vigenere cracker`: IoC + Kasiski + n-gram scored Caesar subproblems.
- `Transposition cracker`: MCMC search over column permutations.
- `Playfair cracker`: population-based hill climbing with move operators.
- `Affine cracker`: exhaustive search over valid `(a, b)` keys.
- `Rail fence cracker`: rail-count search with n-gram scoring.
- `KPA`: known-plaintext clue extraction and constrained substitution cracking.
- `Detector`: feature-based cipher-family prediction with confidence and reasoning.

### Language and scoring core

- Kneser-Ney n-gram language model (space and no-space variants).
- Multi-language support hooks and frequency-based language detection (`english`, `french`, `german`, `spanish`).
- Robust ghost-mapping I/O pipeline for preserving punctuation/case around substitution cracking.

## Command Overview

| Command | Purpose |
| --- | --- |
| `uv run cryptex train [--force]` | Download corpus and train/load language models |
| `uv run cryptex demo ...` | Encrypt sample text and crack live |
| `uv run cryptex crack ...` | Crack custom ciphertext from text/file/stdin |
| `uv run cryptex detect ...` | Predict cipher family and print feature diagnostics |
| `uv run cryptex analyse ...` | MCMC convergence diagnostics + JSON reports |
| `uv run cryptex benchmark ...` | Compare MCMC, GA, frequency baseline, hill-climb baseline |
| `uv run cryptex phase-transition ...` | Success rate vs ciphertext length |
| `uv run cryptex stress-test` | Run adversarial robustness suite |
| `uv run cryptex historical` | Run curated historical challenges |
| `uv run cryptex language ...` | Train/list/detect language models |
| `uv run cryptex kpa ...` | Known-plaintext clue extraction |

Run help at any time:

```bash
uv run cryptex --help
uv run cryptex crack --help
```

## Full Runbook

### 1) Train models

```bash
uv run cryptex train
```

### 2) Run every demo cipher/method path

```bash
uv run cryptex demo --cipher substitution --method mcmc
uv run cryptex demo --cipher substitution --method hmm
uv run cryptex demo --cipher substitution --method genetic

uv run cryptex demo --cipher vigenere
uv run cryptex demo --cipher transposition
uv run cryptex demo --cipher playfair
uv run cryptex demo --cipher affine
uv run cryptex demo --cipher railfence
uv run cryptex demo --cipher noisy-substitution --method mcmc
```

### 3) Crack command input styles

Direct text:

```bash
uv run cryptex crack --cipher substitution --method mcmc --text "wkh txlfn eurzq ira mxpsv ryhu wkh odcb grj"
```

File:

```bash
uv run cryptex crack --cipher vigenere --file .\samples\vig.txt
```

stdin:

```bash
Get-Content .\samples\mystery.txt | uv run cryptex crack --cipher substitution
```

Auto detect + competitive cracking:

```bash
uv run cryptex crack --auto --text "xqzv 9988 @@ lxfopv ef rnhr -- ???"
```

JSON/plain outputs, timeout, top-k candidates:

```bash
uv run cryptex crack --cipher substitution --text "..." --output-format json --top-k 3 --timeout 30
uv run cryptex crack --cipher substitution --text "..." --output-format plain --top-k 5
```

Known-plaintext constrained cracking:

```bash
uv run cryptex crack --cipher substitution --text "..." --known-pair "truth:abcde" --known-pair "single:qwert"
```

KPA clue extraction helper:

```bash
uv run cryptex kpa --ciphertext "..." --known "knownword"
```

For substitution/noisy-substitution paths, ghost mapping preserves punctuation/layout and restores case in final output.

### 4) Detection

```bash
uv run cryptex detect --text "lxfopvefrnhr"
uv run cryptex detect --file .\unknown.txt
```

### 5) Analysis and benchmark reports

```bash
uv run cryptex analyse --output .\plots
uv run cryptex benchmark --length 300 --trials 3 --success-threshold 0.20 --output .\plots
uv run cryptex phase-transition --trials 3 --output .\plots
```

Generated JSON artifacts:
- `plots/convergence_plot.json`
- `plots/key_heatmap.json`
- `plots/frequency_comparison.json`
- `plots/benchmark.json`
- `plots/phase_transition.json`

### 6) Stress, historical, and language workflows

```bash
uv run cryptex stress-test
uv run cryptex historical

uv run cryptex language list
uv run cryptex language train --lang french
uv run cryptex language train --lang german
uv run cryptex language train --lang spanish
uv run cryptex language detect --text "bonjour le monde"
```

Supported language labels: `english`, `french`, `german`, `spanish`.

## Benchmark and Validation Artifacts

Use the project benchmark harness to run a full 10-text custom sweep across implemented algorithms:

```bash
uv run python .\tools\run_custom_benchmarks.py
```

Primary output:
- `plots/custom_benchmark_suite.json`

Detailed benchmark discussion and measured results are maintained in `BENCHMARKS.md`.

### Measured Snapshot

From `plots/benchmark.json` (built-in substitution benchmark):

| Method | Avg SER | Success Rate | Avg Time (s) |
| --- | ---: | ---: | ---: |
| MCMC | 0.0000 | 1.0000 | 10.4524 |
| Genetic Algorithm | 0.0000 | 1.0000 | 8.5106 |
| Frequency Analysis | 0.6231 | 0.0000 | 0.0002 |
| Hill Climb + Freq Init | 0.2828 | 0.6667 | 0.1514 |

From `plots/custom_benchmark_suite.json` (10 custom texts, all implemented algorithms):
- Total solver runs: 130
- Roundtrip encode/decode checks: 10/10 pass for substitution, vigenere, transposition, playfair, affine, rail fence, and noisy-substitution (best-effort decode path)
- Cipher detector accuracy: 39/60 (65.0%)
- Language detection accuracy: 9/10 (90.0%)

See `BENCHMARKS.md` for the full per-algorithm ranking and interpretation.

## Development Quality Gates

```bash
uvx ruff check --fix
uvx ruff format
uvx ty check
uv run pytest -q
```

## Known Limits

- Very short ciphertexts can remain ambiguous.
- Playfair is the hardest mode and may need longer text and higher iteration budgets.
- Non-alphabetic-only payloads fail cleanly with `Error: no decipherable alphabetic content found.`
- Missing files fail cleanly with `Error reading file '<path>': ...`.

## Additional Documentation

- `BENCHMARKS.md` for reproducible benchmark commands and measured project results.

