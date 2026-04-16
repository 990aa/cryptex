# cipher

High-performance classical cryptanalysis toolkit (v0.4.0) for educational and research use.

It provides:
- Substitution cracking with adaptive MCMC, parallel tempering, and crossover jumps.
- Alternative substitution solvers: HMM/EM and genetic algorithm.
- Dedicated crackers for Vigenere, columnar transposition, and Playfair.
- Kneser-Ney 5-gram language models (space and no-space variants).
- Robust dirty-text handling (Unicode folding, punctuation preservation, case restoration).
- Auto-detection and competitive solver orchestration.
- Evaluation tooling: benchmark, phase transition analysis, adversarial tests, historical cases.

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
uv run cipher train

# Run a live substitution demo
uv run cipher demo --cipher substitution --method mcmc

# Crack unknown text with automatic routing
uv run cipher crack --auto --text "xqzv 9988 @@ lxfopv ef rnhr -- ???"

# Run full tests
uv run pytest -q
```

## Command Overview

| Command | Purpose |
| --- | --- |
| `uv run cipher train [--force]` | Download corpus and train/load language models |
| `uv run cipher demo ...` | Encrypt sample text and crack live |
| `uv run cipher crack ...` | Crack custom ciphertext from text/file/stdin |
| `uv run cipher detect ...` | Predict cipher family and print feature diagnostics |
| `uv run cipher analyse ...` | MCMC convergence diagnostics + plots |
| `uv run cipher benchmark ...` | Compare MCMC, GA, frequency baseline, random restart |
| `uv run cipher phase-transition ...` | Success rate vs ciphertext length |
| `uv run cipher stress-test` | Run adversarial robustness suite |
| `uv run cipher historical` | Run curated historical challenges |
| `uv run cipher language ...` | Train/list/detect language models |

Run help any time:

```bash
uv run cipher --help
uv run cipher crack --help
```

## Crack Command: All Input Styles

### 1) Direct text

```bash
uv run cipher crack --cipher substitution --method mcmc --text "wkh txlfn eurzq ira mxpsv ryhu wkh odcb grj"
```

### 2) File input

```bash
uv run cipher crack --cipher vigenere --file .\samples\vig.txt
```

### 3) stdin input

```bash
Get-Content .\samples\mystery.txt | uv run cipher crack --cipher substitution
```

### 4) Auto mode (detect + orchestrate)

```bash
uv run cipher crack --auto --text "xqzv 9988 @@ lxfopv ef rnhr -- ???"
```

### 5) Dirty text with punctuation and mixed case

```bash
uv run cipher crack --cipher substitution --text "Wkh txlfn, Eurzq Fox! 2026?"
```

For substitution/noisy-substitution routes, the CLI uses ghost mapping to:
- extract a solver core,
- keep punctuation/layout,
- restore case and symbols in output.

## Cipher and Method Options

### Cipher types

- `substitution`
- `vigenere`
- `transposition`
- `playfair`
- `noisy-substitution`

### Methods (for substitution/noisy-substitution)

- `mcmc` (default)
- `hmm`
- `genetic`

Examples:

```bash
uv run cipher demo --cipher substitution --method genetic
uv run cipher crack --cipher substitution --method hmm --file .\cipher.txt
uv run cipher demo --cipher noisy-substitution --method hmm
```

## Detection and Auto Routing

```bash
uv run cipher detect --text "lxfopvefrnhr"
uv run cipher detect --file .\unknown.txt
```

`detect` reports:
- predicted type,
- confidence,
- reasoning,
- feature values (IoC, entropy, chi-squared, length).

`crack --auto` behavior:
- High confidence (>= 80%): route directly to predicted solver.
- Low confidence: run competitive candidates and choose highest score.

## Analysis and Plots

### Convergence analysis

```bash
uv run cipher analyse --output .\plots
```

Generates:
- `plots/convergence_plot.png`
- `plots/key_heatmap.png`
- `plots/frequency_comparison.png`

### Benchmark

```bash
uv run cipher benchmark --length 120 --trials 3 --success-threshold 0.25 --output .\plots
```

Notes:
- Success is counted when `SER <= success-threshold`.
- The benchmark plot includes explicit success-rate labels for each method.

### Phase transition

```bash
uv run cipher phase-transition --trials 3 --output .\plots
```

## Stress, Historical, and Language Commands

```bash
uv run cipher stress-test
uv run cipher historical

uv run cipher language list
uv run cipher language train --lang french
uv run cipher language detect --text "bonjour le monde"
```

Supported languages: english, french, german, spanish.

## Development Quality Gates

Run the same checks used during cleanup in this iteration:

```bash
uvx ruff check --fix
uvx ruff format
uvx ty check
uv run pytest -q
```

Current verified state in this workspace:
- Ruff: clean
- Ty: clean
- Pytest: 257 passed

## Known Limits and Failure Modes

- Very short ciphertext (< ~30 chars) can decrypt poorly.
- Playfair remains the hardest mode and may need longer text and more iterations.
- Non-alphabetic-only text fails cleanly with: `Error: no decipherable alphabetic content found.`
- Missing files fail cleanly with: `Error reading file '<path>': ...`

See full empirical run log in `docs/trials.md`.

## Additional Docs

- `docs/IMPLEMENTATION.md` - architecture and module internals
- `docs/TESTING.md` - test strategy and reproducible command set
- `docs/trials.md` - custom input matrix with observed outputs and failures
