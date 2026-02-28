# Testing Guide

> Complete step-by-step instructions to test, demo, and stress-test every feature of the **Unsupervised Cipher Cracker** (`cipher`) package.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Running the Automated Test Suite](#running-the-automated-test-suite)
3. [Test Suite Breakdown](#test-suite-breakdown)
4. [Manual Demo — Full Feature Walkthrough](#manual-demo--full-feature-walkthrough)
   - [Step 1: Train the Language Model](#step-1-train-the-language-model)
   - [Step 2: Substitution Cipher (MCMC)](#step-2-substitution-cipher-mcmc)
   - [Step 3: Substitution Cipher (HMM/EM)](#step-3-substitution-cipher-hmm-em)
   - [Step 4: Substitution Cipher (Genetic Algorithm)](#step-4-substitution-cipher-genetic-algorithm)
   - [Step 5: Vigenère Cipher](#step-5-vigenère-cipher)
   - [Step 6: Columnar Transposition Cipher](#step-6-columnar-transposition-cipher)
   - [Step 7: Playfair Cipher](#step-7-playfair-cipher)
   - [Step 8: Noisy Substitution (HMM)](#step-8-noisy-substitution-hmm)
   - [Step 9: Cipher Type Detection](#step-9-cipher-type-detection)
   - [Step 10: Convergence Analysis & Diagnostic Plots](#step-10-convergence-analysis--diagnostic-plots)
   - [Step 11: Benchmark MCMC vs GA vs Baselines](#step-11-benchmark-mcmc-vs-ga-vs-baselines)
   - [Step 12: Phase Transition Analysis](#step-12-phase-transition-analysis)
   - [Step 13: Adversarial Stress Tests](#step-13-adversarial-stress-tests)
   - [Step 14: Historical Cipher Challenges](#step-14-historical-cipher-challenges)
   - [Step 15: Multi-Language Support](#step-15-multi-language-support)
   - [Step 16: Crack Custom Ciphertext](#step-16-crack-custom-ciphertext)
5. [Edge Case Testing](#edge-case-testing)
6. [Stress Testing](#stress-testing)
7. [Performance Benchmarking](#performance-benchmarking)

---

## Prerequisites

1. **Python 3.14+** installed
2. **uv** package manager installed
3. Clone the repository:
   ```bash
   git clone https://github.com/990aa/uncipher.git
   cd uncipher
   ```
4. Install dependencies (including dev):
   ```bash
   uv sync --dev
   ```

> **Note:** The first run of any command that requires the language model will automatically download the corpus from Project Gutenberg (~8 books) and train the n-gram model. This takes 1–3 minutes. Subsequent runs use the cached model.

---

## Running the Automated Test Suite

### Run All Tests

```bash
uv run pytest tests/ -v --timeout=300
```

**Expected output:**
```
tests/test_adversarial.py::test_stress_test_case_defaults PASSED
tests/test_adversarial.py::test_stress_test_result_defaults PASSED
...
tests/test_stress.py::test_stress_sequential PASSED
tests/test_stress.py::test_evaluation_integration PASSED
========================= 251 passed in ~220s =========================
```

All 251 tests should pass. The full suite takes approximately 3–4 minutes due to the stress tests that actually run MCMC/HMM/GA solvers.

### Run Specific Test Modules

```bash
# Unit tests only (fast, ~10s)
uv run pytest tests/test_ciphers.py tests/test_ngram.py tests/test_cli.py -v

# Solver tests (medium, ~60s)
uv run pytest tests/test_mcmc.py tests/test_hmm.py tests/test_genetic.py -v

# Cracker tests (medium, ~30s)
uv run pytest tests/test_crackers.py -v

# Analysis/diagnostics tests (fast, ~5s)
uv run pytest tests/test_detector.py tests/test_convergence.py tests/test_evaluation.py -v

# Stress tests only (slow, ~180s)
uv run pytest tests/test_stress.py -v --timeout=300
```

### Run Tests by Marker/Keyword

```bash
# Run only tests matching a keyword
uv run pytest tests/ -v -k "substitution"
uv run pytest tests/ -v -k "vigenere"
uv run pytest tests/ -v -k "playfair"
uv run pytest tests/ -v -k "hmm"
uv run pytest tests/ -v -k "genetic"
uv run pytest tests/ -v -k "edge"
```

---

## Test Suite Breakdown

| Test File | Tests | Coverage Area |
|---|---|---|
| `test_corpus.py` | 8 | Corpus download, caching, preprocessing, Gutenberg stripping |
| `test_ngram.py` | 16 | Character encoding, n-gram counting, scoring, model save/load |
| `test_ciphers.py` | 25+ | All 5 cipher engines: encrypt/decrypt roundtrips, edge cases |
| `test_mcmc.py` | 14 | MCMC config, key operations, solver execution, callbacks |
| `test_hmm.py` | 8 | HMM config, log-space math, Baum-Welch EM execution |
| `test_genetic.py` | 12 | GA config, crossover, mutation, population evolution |
| `test_crackers.py` | 11 | Vigenère IoC/Kasiski, transposition, Playfair crackers |
| `test_detector.py` | 16 | All 6 statistical features, cipher type classification |
| `test_convergence.py` | 11 | Sparklines, chain consensus, diagnostic plots |
| `test_evaluation.py` | 15 | SER, key accuracy, baselines, benchmark/phase transition |
| `test_languages.py` | 15 | Language frequencies, transliteration, detection, models |
| `test_adversarial.py` | 12 | Stress test generators, adversarial keys, stress runner |
| `test_historical.py` | 12 | Historical cipher catalogue, challenge runner |
| `test_display.py` | 18 | Rich displays (MCMC, HMM, Generic), result boxes |
| `test_cli.py` | 24 | Argument parsing for all 10 subcommands, dispatch |
| `test_stress.py` | 11 | End-to-end solver tests, multi-cipher stress, SER validation |

**Total: 251 tests**

---

## Manual Demo — Full Feature Walkthrough

Each step below is a command you can run. Expected output descriptions follow each command.

### Step 1: Train the Language Model

```bash
uv run cipher train
```

**Expected output:**
```
Training language models (this may take a few minutes the first time)…
✓ Models ready  (85.3s)
  Space model: alphabet=27, weights=(0.05, 0.1, 0.25, 0.6)
  No-space model: alphabet=26, weights=(0.05, 0.1, 0.25, 0.6)
```

The first run downloads 8 Gutenberg books, strips headers, cleans text, and trains unigram through quadgram models. Subsequent runs load from cache in <1s.

**Force retrain:**
```bash
uv run cipher train --force
```

### Step 2: Substitution Cipher (MCMC)

```bash
uv run cipher demo --cipher substitution --method mcmc
```

**Expected output:**

1. A red panel showing the encrypted ciphertext
2. The true key (26-character permutation) printed in dim text
3. A **live-updating Rich dashboard** showing:
   - Current chain number and iteration (e.g., "Chain 3/5 · Iteration 28500/30000")
   - Real-time decrypted text (starts as gibberish, converges to English)
   - Unicode sparkline score trajectory (▁▂▃▄▅▆▇█)
   - Temperature, current score, best score, acceptance rate
   - Compact key mapping table (cipher → plaintext)
4. Final result box with:
   - Cipher type: Simple Substitution
   - Method: MCMC
   - Recovered key
   - Decrypted plaintext (should be readable English)
   - Score and time elapsed (~25–75s total)

### Step 3: Substitution Cipher (HMM/EM)

```bash
uv run cipher demo --cipher substitution --method hmm
```

**Expected output:**

1. Red panel with ciphertext
2. Live dashboard showing HMM iterations (1–100) with log-likelihood convergence
3. Convergence sparkline showing log-likelihood improving over iterations
4. Current decryption attempt updating each iteration
5. Final result box — HMM may produce partial decryption (it's more naturally suited for noisy channels)

### Step 4: Substitution Cipher (Genetic Algorithm)

```bash
uv run cipher demo --cipher substitution --method genetic
```

**Expected output:**

1. Red panel with ciphertext
2. Live display showing:
   - Generation count (up to 500)
   - Best fitness score improving over generations
   - Current best decryption
3. Final result box with evolved key and plaintext

### Step 5: Vigenère Cipher

```bash
uv run cipher demo --cipher vigenere
```

**Expected output:**

1. Red panel with Vigenère-encrypted text (no spaces in ciphertext)
2. Status messages:
   - Key length detection via IoC analysis
   - Best key length found (e.g., "Key length: 6")
   - Key recovery per position
3. Final result box with recovered keyword and decrypted text

### Step 6: Columnar Transposition Cipher

```bash
uv run cipher demo --cipher transposition
```

**Expected output:**

1. Red panel with transposed ciphertext
2. Live display showing MCMC search over column permutations
3. Tries multiple key lengths (2–10 columns)
4. Final result box with recovered column order and plaintext
5. Note: Plaintext may have trailing 'x' characters (padding)

### Step 7: Playfair Cipher

```bash
uv run cipher demo --cipher playfair
```

**Expected output:**

1. Red panel with Playfair-encrypted text (no spaces, no 'j')
2. Live display showing MCMC over 5×5 matrix mutations
3. Score improving as matrix converges
4. Final result box with recovered Playfair matrix key and decrypted text
5. Decryption will have 'i' instead of 'j' and 'x' padding between repeated letters

### Step 8: Noisy Substitution (HMM)

```bash
uv run cipher demo --cipher noisy-substitution --method hmm
```

**Expected output:**

1. Red panel with noisy ciphertext (some characters randomly corrupted or deleted)
2. HMM iteration display — the HMM naturally handles noise through its emission model
3. Final result with best-effort decryption (may have some residual errors at noise sites)

### Step 9: Cipher Type Detection

**Detect a substitution cipher:**
```bash
uv run cipher detect --text "wkh txlfn eurzq ira mxpsv ryhu wkh odcb grj dqg ihow txlwh jrrg derxw lw"
```

**Expected output:**
```
Cipher Type Detection
═══════════════════════════════════════════════
  Predicted type: substitution
  Confidence: 0.75
  Reasoning: High IoC (>0.055) suggests monoalphabetic; spaces present...

  Feature Summary:
    IoC: 0.0652
    Entropy: 4.12
    Chi-squared: 15.3
    Bigram repeat: 0.042
```

**Detect a Vigenère cipher:**
```bash
uv run cipher detect --text "lxfopvefrnhr"
```

**Expected output:**
```
  Predicted type: vigenere
  Confidence: 0.60
  Reasoning: Low IoC suggests polyalphabetic...
```

### Step 10: Convergence Analysis & Diagnostic Plots

```bash
uv run cipher analyse --output ./plots
```

**Expected output:**

1. Runs MCMC with trajectory tracking enabled
2. Live MCMC display (same as demo)
3. After cracking, generates 4 diagnostic files in `./plots/`:
   - `score_trajectory.png` — Score vs iteration with acceptance rate overlay
   - `key_confidence.png` — 26×26 heatmap of cross-chain agreement
   - `frequency_comparison.png` — Letter frequency bars (ciphertext vs decrypted vs English)
   - Terminal output: sparkline trajectory, chain consensus stats

```
Chain consensus
  Agreement rate: 88.5%
  Number of chains: 5
  Consensus key: etaoinshrdlcumwfgypbvkjxqz

Score trajectory sparkline:
▁▁▂▂▃▃▄▅▅▆▆▇▇▇████████████████████████████████

Saved: ./plots/score_trajectory.png
Saved: ./plots/key_confidence.png
Saved: ./plots/frequency_comparison.png
```

### Step 11: Benchmark MCMC vs GA vs Baselines

```bash
uv run cipher benchmark --trials 3 --length 300 --output ./plots
```

**Expected output:**
```
Running benchmark: 4 methods × 3 trials on 300-char texts…

Method                Avg SER    Success Rate    Avg Time
─────────────────────────────────────────────────────────
MCMC                  0.023      100.0%          32.5s
Genetic Algorithm     0.045       66.7%          45.1s
Frequency Analysis    0.312        0.0%           0.0s
Random Restarts       0.487        0.0%           0.3s

Saved: ./plots/benchmark.png
```

### Step 12: Phase Transition Analysis

```bash
uv run cipher phase-transition --trials 2 --output ./plots
```

**Expected output:**

Tests success rate at ciphertext lengths [50, 75, 100, 125, 150, 200, 300, 500]:

```
Phase Transition Analysis: 8 lengths × 2 trials…

Length    Success Rate    Avg SER    Avg Time
──────────────────────────────────────────────
50       0.0%           0.412      12.3s
75       0.0%           0.285      14.1s
100      50.0%          0.087      18.2s
125      50.0%          0.055      22.4s
150      100.0%         0.021      25.8s
200      100.0%         0.008      28.3s
300      100.0%         0.003      31.5s
500      100.0%         0.001      38.2s

Estimated threshold: ~125 characters

Saved: ./plots/phase_transition.png
```

The phase transition typically occurs around 100–150 characters — below this, there's insufficient statistical signal for reliable cracking.

### Step 13: Adversarial Stress Tests

```bash
uv run cipher stress-test
```

**Expected output:**

Runs 8 adversarial test cases with 40k iterations and 6 restarts each:

```
Adversarial Stress Tests (8 cases)

Case 1: legal_text + frequency_preserving_key
  SER: 0.032  ✓ PASS (< 10%)

Case 2: legal_text + adversarial_key
  SER: 0.078  ✓ PASS (< 10%)

Case 3: technical_text + adversarial_key
  SER: 0.015  ✓ PASS (< 10%)

Case 4: poetry_text + frequency_preserving_key
  SER: 0.045  ✓ PASS (< 10%)

Case 5: repetitive_text + adversarial_key
  SER: 0.089  ✓ PASS (< 10%)

Case 6: short_text (~65 chars) + adversarial_key
  SER: 0.152  ✗ FAIL (text too short — expected)

Case 7: very_short_text (~11 chars) + adversarial_key
  SER: 0.545  ✗ FAIL (below phase transition — expected)

Case 8: technical_text + frequency_preserving_key
  SER: 0.008  ✓ PASS (< 10%)

Results: 6/8 passed (short texts below phase transition expected to fail)
```

### Step 14: Historical Cipher Challenges

```bash
uv run cipher historical
```

**Expected output:**

Runs 6 historical cipher challenges:

```
Running 6 historical cipher challenges…

Caesar Shift (ROT-13)
  Type: substitution, Difficulty: easy
  Time: 28.3s, Score: -1234.5
  SER: 0.0%  ✓ PASS
  Decrypted: it is a truth universally acknowledged that a single man…

Aristocrat Cipher (ACA Style)
  Type: substitution, Difficulty: medium
  Time: 35.1s, Score: -1567.2
  SER: 2.1%  ✓ PASS
  Decrypted: it was the best of times it was the worst of times…

Vigenère Cipher (Bellaso 1553)
  Type: vigenere, Difficulty: medium
  Time: 5.2s, Score: -987.6
  SER: 0.0%  ✓ PASS
  Decrypted: to be or not to be that is the question…

WWI Training Cipher
  Type: substitution, Difficulty: medium
  Time: 32.4s
  SER: 1.5%  ✓ PASS

Columnar Transposition (WWI Field)
  Type: transposition, Difficulty: medium
  Time: 45.6s
  SER: 0.3%  ✓ PASS

Playfair Cipher (Crimean War)
  Type: playfair, Difficulty: hard
  Time: 55.2s
  SER: 5.8%  ✓ PASS
```

### Step 15: Multi-Language Support

**List available languages:**
```bash
uv run cipher language list
```

**Expected output:**
```
  english
  french
  german
  spanish
```

**Train a French model:**
```bash
uv run cipher language train --lang french
```

**Expected output:**
```
Training french language model…
Done! Alphabet size: 27
```

**Detect language:**
```bash
uv run cipher language detect --text "bonjour le monde comment allez vous"
```

**Expected output:**
```
Detected language: french
  french: chi² = 12.3456
  english: chi² = 45.6789
  spanish: chi² = 38.1234
  german: chi² = 52.4567
```

```bash
uv run cipher language detect --text "the quick brown fox jumps over the lazy dog"
```

**Expected output:**
```
Detected language: english
  english: chi² = 8.1234
  french: chi² = 42.5678
  spanish: chi² = 35.9012
  german: chi² = 48.3456
```

### Step 16: Crack Custom Ciphertext

**From command line:**
```bash
uv run cipher crack --cipher substitution --method mcmc --text "wkh txlfn eurzq ira mxpsv ryhu wkh odcb grj dqg ihow txlwh jrrg derxw lw"
```

**From file:**
```bash
echo "wkh txlfn eurzq ira mxpsv ryhu wkh odcb grj" > secret.txt
uv run cipher crack --cipher substitution --text "$(cat secret.txt)"
```

**Expected output:** Live MCMC display followed by result box with decrypted English text.

---

## Edge Case Testing

The automated test suite covers all edge cases listed below. You can verify them individually:

### Empty Input

```bash
uv run pytest tests/test_ciphers.py -v -k "empty"
```

Tests: `test_substitution_empty`, `test_vigenere_empty`, `test_transposition_empty`, `test_playfair_empty`

All ciphers return empty string for empty input.

### Space-Only Input

```bash
uv run pytest tests/test_ciphers.py -v -k "only_spaces"
```

Substitution cipher preserves spaces correctly.

### Identity Key

```bash
uv run pytest tests/test_ciphers.py -v -k "identity"
```

Identity permutation key produces unchanged plaintext.

### Single-Character Key (Vigenère)

```bash
uv run pytest tests/test_ciphers.py -v -k "single_char"
```

Single-character key = Caesar shift.

### Short Text (Near Phase Transition)

```bash
uv run pytest tests/test_stress.py -v -k "short"
```

Tests MCMC on ~50-character text — may not fully decode but should run without errors.

### Noisy Cipher Resilience

```bash
uv run pytest tests/test_ciphers.py -v -k "noisy"
```

Tests zero-noise identity, corruption, and deletion rates.

### Character Encoding Edge Cases

```bash
uv run pytest tests/test_ngram.py -v -k "char_to_idx or text_to_indices"
```

Verifies space → 0, a → 1, ..., z → 26 mapping (with `include_space=True`).

### Playfair J/I Merging

```bash
uv run pytest tests/test_ciphers.py -v -k "playfair"
```

Verifies 'j' is replaced with 'i' and the 25-character alphabet is used.

### Model Serialisation Round-trip

```bash
uv run pytest tests/test_ngram.py -v -k "save_load"
```

Verifies model save/load preserves log-probability arrays exactly.

---

## Stress Testing

### Automated Stress Tests

```bash
# Full stress test suite (11 tests, ~180s)
uv run pytest tests/test_stress.py -v --timeout=300
```

**Tests include:**

| Test | Description | Pass Criterion |
|---|---|---|
| `test_mcmc_end_to_end` | MCMC on 300-char substitution cipher | SER < 20% |
| `test_mcmc_short_text` | MCMC on ~50-char text | Completes without error |
| `test_hmm_end_to_end` | HMM/EM on 300-char cipher (30 iters) | Completes, produces plaintext |
| `test_genetic_end_to_end` | GA on 300-char cipher | Completes, score improves |
| `test_vigenere_end_to_end` | Vigenère crack on keyed text | SER < 25% |
| `test_transposition_end_to_end` | Transposition crack | Completes, produces result |
| `test_detection_substitution` | Detect substitution cipher | Correct classification |
| `test_detection_vigenere` | Detect Vigenère cipher | Returns valid type |
| `test_stress_sequential` | 3 random keys × all cipher engines | All complete |
| `test_evaluation_integration` | SER + key_accuracy computation | Correct metrics |
| `test_display_not_crash` | All display classes render | No exceptions |

### Manual Stress Test

For maximum stress, run the adversarial stress test with verbose output:

```bash
uv run cipher stress-test
```

This runs 8 cases with 40k iterations and 6 restarts each — takes 5–10 minutes. Cases include adversarial keys (mapping common letters to rare ones), short texts near the phase transition, and unusual character distributions (legal/technical/poetic/repetitive text).

---

## Performance Benchmarking

### Quick Benchmark (3 trials)

```bash
uv run cipher benchmark --trials 3 --length 300
```

### Thorough Benchmark (10 trials)

```bash
uv run cipher benchmark --trials 10 --length 500 --output ./benchmark_results
```

### Phase Transition (Quick)

```bash
uv run cipher phase-transition --trials 2
```

### Phase Transition (Thorough)

```bash
uv run cipher phase-transition --trials 5 --output ./phase_results
```

### Full System Exercise

Run everything in sequence:

```bash
# 1. Train models
uv run cipher train --force

# 2. Run automated tests
uv run pytest tests/ -v --timeout=300

# 3. Demo all cipher types
uv run cipher demo --cipher substitution --method mcmc
uv run cipher demo --cipher substitution --method hmm
uv run cipher demo --cipher substitution --method genetic
uv run cipher demo --cipher vigenere
uv run cipher demo --cipher transposition
uv run cipher demo --cipher playfair
uv run cipher demo --cipher noisy-substitution --method hmm

# 4. Detection
uv run cipher detect --text "wkh txlfn eurzq ira mxpsv ryhu wkh odcb grj"

# 5. Analysis
uv run cipher analyse --output ./analysis_plots

# 6. Benchmarks
uv run cipher benchmark --trials 3 --output ./benchmark_plots
uv run cipher phase-transition --trials 2 --output ./phase_plots

# 7. Stress & Historical
uv run cipher stress-test
uv run cipher historical

# 8. Multi-language
uv run cipher language list
uv run cipher language train --lang french
uv run cipher language detect --text "bonjour le monde"
uv run cipher language detect --text "the quick brown fox"
```

This full exercise takes approximately 30–60 minutes depending on hardware.
