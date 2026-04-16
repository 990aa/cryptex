# Implementation Details 

This document describes the current implementation in `src/cryptex`.

## 1) Architecture

Primary CLI data flow:

1. Input acquisition (`--text`, `--file`, or `stdin`).
2. Optional robust preprocessing (`io.ghost_map_text`) for substitution-style routes.
3. Solver execution (MCMC/HMM/GA/Vigenere/Transposition/Playfair).
4. Postprocessing (restore punctuation/case when ghost mapping is used).
5. Metrics and visualization (`evaluation.py`, `convergence.py`, Rich displays).

## 2) Language Model (`ngram.py`)

### Core model

- Supports orders up to 5 (`max_order=5`).
- Default smoothing is interpolated Kneser-Ney (`smoothing="kneser_ney"`, `discount=0.75`).
- Trains both space-aware (`A=27`) and no-space (`A=26`) variants.
- Persisted model files:
  - `src/cryptex/data/ngram_model_v2.pkl`
  - `src/cryptex/data/ngram_model_nospace_v2.pkl`

### Performance paths

- `score_quadgrams_fast`: vectorized quadgram lookup for hot loops.
- `score_quadgrams_with_mapping`: scores candidate keys without materializing plaintext.
- Optional Numba JIT hot path for mapped quadgram scoring when Numba is installed.

### Unicode and dirty input handling at model boundary

- `text_to_indices` supports ASCII fast path and Unicode folding fallback.
- Non-letter symbols are ignored, and whitespace can be preserved when space-model is used.

## 3) Robust I/O (`io.py`)

`io.py` adds safe handling for real-world ciphertext:

- `ghost_map_text(text)`
  - extracts solver core text,
  - tracks alphabetic positions,
  - preserves case mask for reconstruction.
- `restore_ghost_text(plaintext_core, mapping)`
  - reinjects decrypted letters back into original layout,
  - preserves punctuation/digits/symbol positions.
- `discover_effective_alphabet(text)` and `likely_homophonic_cipher(text)`
  - used by CLI warnings and routing heuristics.

## 4) Substitution Solver (`mcmc.py`)

### Search strategy

`run_mcmc` combines several mechanisms:

- Simulated annealing with geometric cooling.
- Adaptive temperature scaling toward target acceptance.
- Optional parallel tempering with replica exchange.
- Occasional order-crossover jump proposals using the global best candidate.
- Multi-restart execution with global best tracking.

### Important configuration fields (`MCMCConfig`)

- `iterations`, `num_restarts`
- `t_start`, `t_min`, `cooling_rate`
- `use_parallel_tempering`, `num_replicas`, `swap_interval`
- `adapt_interval`, `target_acceptance`, `adaptation_rate`
- `crossover_jump_prob`
- `early_stop_patience`, `score_threshold`

### Result payload (`MCMCResult`)

- best key/plaintext/score
- per-chain scores and keys
- score trajectory and acceptance rates
- top candidates (useful for short text)
- replica swap acceptance rate

## 5) Other Solvers

### HMM (`hmm.py`)

- Baum-Welch EM for substitution-style decoding.
- Better tolerance to noise than pure deterministic hill-climbing.

### Genetic (`genetic.py`)

- Population search over substitution permutations.
- Tournament selection, OX1-style crossover, swap mutation, elitism.

### Vigenere (`vigenere_cracker.py`)

- Key-length detection with IoC plus Kasiski cues.
- Per-position Caesar solving via language-model scoring.

### Transposition (`transposition_cracker.py`)

- MCMC over column permutations.
- Explores candidate column counts in configured range.

### Playfair (`playfair_cracker.py`)

- Search over 5x5 key matrix mutations.
- Uses no-space model for scoring.

## 6) Detection and Auto Routing

### Detection (`detector.py`)

Computes statistical features including:

- index of coincidence,
- entropy,
- chi-squared letter-fit,
- repeat/structure heuristics.

Returns predicted type, confidence, score map, and reasoning text.

### Auto mode in CLI (`cipher crack --auto`)

- High confidence: route directly to predicted solver.
- Lower confidence: run competitive candidates (substitution MCMC, substitution HMM, Vigenere) and choose best score.
- Uses robust mapping for substitution candidates.

## 7) Evaluation and Plotting (`evaluation.py`)

### Metrics

- Symbol Error Rate (`symbol_error_rate`)
- Key accuracy (`key_accuracy`)

### Benchmarks

- Methods compared: MCMC, Genetic Algorithm, Frequency Analysis, Random Restarts.
- `run_benchmark(..., success_threshold=...)` controls success rule.
- Success criterion is `SER <= success_threshold`.
- `plot_benchmark` labels success-rate bars and embeds threshold context in title.

### Phase transition

- Runs across ciphertext lengths to estimate reliability threshold.
- Outputs success-rate/ser/time trends and threshold estimate.

## 8) CLI Behavior and Hardening (`cli.py`)

- Supports `--text`, `--file`, and `stdin` input paths.
- File read errors are handled with clean user-facing errors (no raw traceback).
- Empty and non-decodable content is rejected with clear messages.
- Windows UTF-8 output handling is configured early for Rich display compatibility.

## 9) Testing and Static Quality

Current repository quality workflow:

```bash
uvx ruff check --fix
uvx ruff format
uvx ty check
uv run pytest -q
```

Latest validated state during this update:

- Ruff: clean
- Ty: clean
- Pytest: 257 passed

## 10) Practical Limits

- Very short ciphertext can remain ambiguous.
- Playfair generally requires longer text and more compute budget.
- Symbol-only inputs are intentionally rejected for substitution cracking.
- Homophonic ciphers are only heuristically flagged, not fully solved yet.


