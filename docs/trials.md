# Trial Log (2026-04-16)

This log captures the latest validation run for the cryptex v0.5.0 codebase,
including custom-input experiments, edge cases, and observed failures.

## Environment

- OS: Windows
- Runner: `uv run ...`
- Working directory: repository root

## 2) Benchmark Plot Validation (Success Rate Fix)

Command:

```bash
uv run cryptex benchmark --length 120 --trials 1 --success-threshold 0.25 --output ./plots
```

Observed table:

- MCMC: Avg SER 0.0%, Success Rate 100%, Avg Time 1.3s
- Genetic Algorithm: Avg SER 0.0%, Success Rate 100%, Avg Time 5.8s
- Frequency Analysis: Avg SER 73.5%, Success Rate 0%, Avg Time 0.0s
- Random Restarts: Avg SER 100.0%, Success Rate 0%, Avg Time 0.0s

Observed artifact:

- `plots/benchmark.png` saved

Result:

- Success-rate bars were populated and annotated.
- Threshold (`SER <= 25%`) was applied correctly.

## 3) Auto-Orchestration Trial (Low Confidence Input)

Command:

```bash
uv run cryptex crack --auto --text "xqzv 9988 @@ lxfopv ef rnhr -- ???"
```

Observed output summary:

- Auto-detect: `vigenere (61.1% confidence)`
- Low-confidence competitive run executed 3 candidates:
  - Substitution MCMC score: `-24.77`
  - Substitution HMM score: `-27.44`
  - Vigenere score: `-79.23`
- Winner selected: Substitution MCMC
- Decrypted preview: `rave 9988 @@ brouse to ging -- ???`

Interpretation:

- Routing behavior is correct for ambiguous/noisy payloads.
- Output quality is limited when input is very short and mixed with symbols.

## 4) Custom Input Matrix (Programmatic Trials)

The following trial matrix was produced by a direct solver harness (same runtime environment).

| Case | Input Length | Key Result | SER | Score | Outcome |
| --- | ---: | --- | ---: | ---: | --- |
| substitution_clean | 144 | plaintext largely recovered | 0.0171 | -229.93 | pass |
| substitution_dirty_unicode_punct | 79 (core 62) | preserved layout but poor decipherment | 0.7451 (core) | -167.80 | fail |
| vigenere_custom | 112 | recovered key `citterciphez` (incorrect) | 0.2449 | -478.09 | fail |
| transposition_custom | 155 | detected columns = 5, exact plaintext recovery | 0.0000 | -322.03 | pass |
| playfair_custom | 70 | low-quality decryption on short sample | 0.9851 (alpha) | -293.30 | fail |
| substitution_too_short_expected_failure | 17 | partial output only | 0.3571 | -18.57 | expected fail |
| detector_low_confidence_mixed | n/a | predicted `vigenere` @ 0.6111 confidence | n/a | n/a | ambiguous (expected) |
| homophonic_heuristic_symbol_rich | n/a | heuristic flagged true | n/a | n/a | warning path works |

Notes:

- Substitution and transposition are strong on sufficient length.
- Vigenere and Playfair quality dropped for these short custom samples.
- Dirty text handling preserved format, but short core text still limited recovery.

## 5) Failure Cases (CLI)

### 5.1 Missing file path

Command:

```bash
uv run cryptex crack --cipher substitution --file does_not_exist.txt
```

Observed output:

```text
Error reading file 'does_not_exist.txt': [Errno 2] No such file or directory: 'does_not_exist.txt'
```

Status:

- expected fail
- clean user-facing error (no traceback)

### 5.2 Non-alphabetic payload

Command:

```bash
uv run cryptex crack --cipher substitution --text '1234 !!! ### $$$'
```

Observed output:

```text
Error: no decipherable alphabetic content found.
```

Status:

- expected fail
- robust input guard works

## 6) Generated/Verified Artifacts

- `plots/benchmark.png`
- `plots/convergence_plot.png` (previously generated and still valid)
- `plots/key_heatmap.png` (previously generated and still valid)
- `plots/frequency_comparison.png` (previously generated and still valid)
- `plots/phase_transition.png` (previously generated and still valid)

## 7) Summary

- Benchmark success-rate plotting issue: resolved and validated.
- Lint/format/type/test checks: all passing.
- CLI failure paths for file input and non-alphabetic text: cleanly handled.
- Custom-input trials show strong performance on substitution/transposition with enough signal, and known weaknesses on short/noisy Playfair and Vigenere samples.


