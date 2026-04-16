# Testing Guide

This guide documents the repeatable commands used to validate the project.

## 1) Prerequisites

```bash
uv sync --dev
```

The first model-dependent run downloads corpus data and builds local model artifacts.

### Standard cracking modes

```bash
uv run cryptex demo --cipher substitution --method mcmc
uv run cryptex demo --cipher substitution --method hmm
uv run cryptex demo --cipher substitution --method genetic
uv run cryptex demo --cipher vigenere
uv run cryptex demo --cipher transposition
uv run cryptex demo --cipher playfair
```

Expected:

- live progress display
- final result panel with method/key/score/time

### Input sources (`--text`, `--file`, `stdin`)

```bash
uv run cryptex crack --cipher substitution --text "wkh txlfn eurzq ira"
uv run cryptex crack --cipher substitution --file .\cipher.txt
Get-Content .\cipher.txt | uv run cryptex crack --cipher substitution
```

Expected:

- no crashes
- final result panel
- file errors reported cleanly if path is invalid

### Auto orchestration and detection

```bash
uv run cryptex detect --text "lxfopvefrnhr"
uv run cryptex crack --auto --text "xqzv 9988 @@ lxfopv ef rnhr -- ???"
```

Expected:

- predicted type + confidence
- auto mode prints candidate comparison and selected winner

### Evaluation plots

```bash
uv run cryptex analyse --output .\plots
uv run cryptex benchmark --length 120 --trials 1 --success-threshold 0.25 --output .\plots
uv run cryptex phase-transition --trials 1 --output .\plots
```

Expected files:

- `plots/convergence_plot.png`
- `plots/key_heatmap.png`
- `plots/frequency_comparison.png`
- `plots/benchmark.png`
- `plots/phase_transition.png`

## 6) Failure-Path Checks

These are expected error cases and should remain stable.

### Missing input file

```bash
uv run cryptex crack --cipher substitution --file does_not_exist.txt
```

Expected:

- exit code 1
- clean error message beginning with `Error reading file ...`

### No decipherable alphabetic payload

```bash
uv run cryptex crack --cipher substitution --text '1234 !!! ### $$$'
```

Expected:

- exit code 1
- `Error: no decipherable alphabetic content found.`

## 7) Performance Notes

- MCMC-heavy tests and stress commands can be slow on low-power hardware.
- Playfair cracking is computationally expensive and may need longer runs for high quality output.
- Benchmark and phase-transition commands are best run with small trial counts during rapid iteration.

