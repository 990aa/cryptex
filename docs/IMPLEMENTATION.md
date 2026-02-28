# Implementation Details

> Comprehensive technical documentation for the **Unsupervised Cipher Cracker** (`cipher`) package — every module, algorithm, data structure, and design decision.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Module Reference](#module-reference)
   - [corpus.py — Corpus Pipeline](#corpuspy--corpus-pipeline)
   - [ngram.py — N-Gram Language Model](#ngrampy--n-gram-language-model)
   - [ciphers.py — Cipher Engines](#cipherspy--cipher-engines)
   - [mcmc.py — MCMC Solver](#mcmcpy--mcmc-solver)
   - [hmm.py — HMM/EM Solver](#hmmpy--hmm-em-solver)
   - [genetic.py — Genetic Algorithm Solver](#geneticpy--genetic-algorithm-solver)
   - [vigenere_cracker.py — Vigenère Cracker](#vigenere_crackerpy--vigenère-cracker)
   - [transposition_cracker.py — Transposition Cracker](#transposition_crackerpy--transposition-cracker)
   - [playfair_cracker.py — Playfair Cracker](#playfair_crackerpy--playfair-cracker)
   - [convergence.py — Convergence Diagnostics](#convergencepy--convergence-diagnostics)
   - [evaluation.py — Evaluation & Benchmarking](#evaluationpy--evaluation--benchmarking)
   - [detector.py — Cipher Type Detection](#detectorpy--cipher-type-detection)
   - [languages.py — Multi-Language Support](#languagespy--multi-language-support)
   - [adversarial.py — Adversarial Stress Testing](#adversarialpy--adversarial-stress-testing)
   - [historical.py — Historical Cipher Challenges](#historicalpy--historical-cipher-challenges)
   - [display.py — Rich Terminal Visualisation](#displaypy--rich-terminal-visualisation)
   - [cli.py — Command-Line Interface](#clipy--command-line-interface)
3. [Algorithms Deep Dive](#algorithms-deep-dive)
4. [Data Flow & Pipeline](#data-flow--pipeline)
5. [Configuration & Tuning](#configuration--tuning)
6. [Performance Characteristics](#performance-characteristics)

---

## Architecture Overview

```
src/cipher/
├── __init__.py          # Package metadata (__version__)
├── corpus.py            # Gutenberg corpus download & preprocessing
├── ngram.py             # Character-level 1/2/3/4-gram language model
├── ciphers.py           # 5 cipher engines (encrypt/decrypt/keygen)
├── mcmc.py              # Metropolis-Hastings substitution solver
├── hmm.py               # Hidden Markov Model (Baum-Welch EM) solver
├── genetic.py           # Genetic algorithm substitution solver
├── vigenere_cracker.py  # IoC + frequency analysis Vigenère cracker
├── transposition_cracker.py  # MCMC columnar transposition cracker
├── playfair_cracker.py  # MCMC Playfair cracker (hill climbing)
├── convergence.py       # MCMC convergence diagnostics & plots
├── evaluation.py        # SER, key accuracy, benchmarking, phase transitions
├── detector.py          # Statistical cipher type classifier
├── languages.py         # French/German/Spanish language models & detection
├── adversarial.py       # Adversarial stress test case generation & execution
├── historical.py        # Curated historical cipher challenges
├── display.py           # Rich-powered real-time terminal displays
├── cli.py               # argparse CLI with 10 subcommands
└── data/                # Cached corpus & trained models (auto-generated)
```

### Design Principles

| Principle | Implementation |
|---|---|
| **Unsupervised** | No labelled cipher/plaintext pairs needed — only raw English text for n-gram training |
| **Modular** | Each cipher type, solver method, and analysis tool is an independent module |
| **Live Feedback** | Rich terminal UI shows decryption progress in real-time |
| **Reproducible** | Corpus caching and model serialisation ensure deterministic builds |
| **Extensible** | `CipherEngine` protocol allows adding new cipher types trivially |

### Technology Stack

| Component | Technology |
|---|---|
| Language | Python 3.12+ |
| Build System | Hatchling (PEP 517/518) |
| Package Manager | `uv` |
| Numerical | NumPy |
| Terminal UI | Rich (Live, Panel, Table, Layout, Progress) |
| HTTP | Requests |
| Plotting | Matplotlib (optional — for convergence/benchmark plots) |
| Testing | pytest + pytest-timeout |
| Linting | Ruff |

---

## Module Reference

### `corpus.py` — Corpus Pipeline

**Purpose:** Downloads, preprocesses, and caches a large English-language corpus from Project Gutenberg for training the n-gram language model.

#### Constants

| Name | Description |
|---|---|
| `GUTENBERG_URLS` | List of 8 public-domain books with their Gutenberg URLs: *Pride and Prejudice*, *Moby Dick*, *A Tale of Two Cities*, *Adventures of Sherlock Holmes*, *War and Peace*, *Great Expectations*, *The Adventures of Tom Sawyer*, *Frankenstein* |
| `DATA_DIR` | `<package>/data/` — auto-created directory for cached data |
| `CORPUS_FILE` | `<package>/data/corpus.txt` — cached corpus path |
| `ALLOWED_WITH_SPACE` | `set('abcdefghijklmnopqrstuvwxyz ')` — 27-character alphabet |
| `ALLOWED_ALPHA_ONLY` | `set('abcdefghijklmnopqrstuvwxyz')` — 26-character alphabet |

#### Functions

**`_strip_gutenberg_header_footer(text: str) -> str`**

Removes Project Gutenberg boilerplate by searching for start/end markers:
- Start markers: `"*** START OF THE PROJECT GUTENBERG"`, `"*** START OF THIS PROJECT GUTENBERG"`
- End markers: `"*** END OF THE PROJECT GUTENBERG"`, `"*** END OF THIS PROJECT GUTENBERG"`, `"End of the Project Gutenberg"`, `"End of Project Gutenberg"`

**`download_corpus(force: bool = False, include_space: bool = True) -> str`**

Pipeline:
1. Check cache (`CORPUS_FILE`) — return immediately if exists and `force=False`
2. Download each Gutenberg text with `requests.get(url, timeout=30)`
3. Strip Gutenberg headers/footers
4. Concatenate all texts
5. Lowercase → collapse whitespace (`re.sub(r"\s+", " ")`) → filter to allowed characters
6. Write to cache and return

The resulting corpus is ~4–6 million characters of clean English text.

---

### `ngram.py` — N-Gram Language Model

**Purpose:** Character-level n-gram language model with interpolated smoothing for scoring plaintext candidates.

#### Character Encoding

| `include_space=True` | `include_space=False` |
|---|---|
| space → 0, a → 1, b → 2, ..., z → 26 | a → 0, b → 1, ..., z → 25 |
| Alphabet size `A = 27` | Alphabet size `A = 26` |

**`char_to_idx(ch: str, include_space: bool = True) -> int`** — Maps character to integer index.

**`text_to_indices(text: str, include_space: bool = True) -> np.ndarray`** — Vectorised conversion using ASCII byte operations (`np.frombuffer`). Returns `int8` array.

#### `NgramModel` Class

**Constructor Parameters:**
- `include_space: bool = True` — Whether to include space as a character
- `lambdas: tuple[float, float, float, float] = (0.05, 0.10, 0.25, 0.60)` — Interpolation weights for unigram, bigram, trigram, quadgram

**Internal State:**
- `uni`, `bi`, `tri`, `quad` — Raw count arrays (filled during training, not persisted)
- `log_uni`, `log_bi`, `log_tri`, `log_quad` — Log-probability arrays (persisted)

**`train(corpus: str) -> None`**

Counts all n-gram occurrences by iterating through the corpus indices. Then calls `_compute_log_probs()`.

**`_compute_log_probs() -> None`**

Applies **Laplace smoothing** (α=1.0) and converts counts to log-probabilities:
- Unigram: `P(c) = (count(c) + α) / (N + α·A)`
- Bigram: `P(c₂|c₁) = (count(c₁,c₂) + α) / (count(c₁) + α·A)`
- Trigram: `P(c₃|c₁,c₂) = (count(c₁,c₂,c₃) + α) / (count(c₁,c₂) + α·A)`
- Quadgram: `P(c₄|c₁,c₂,c₃) = (count(c₁,c₂,c₃,c₄) + α) / (count(c₁,c₂,c₃) + α·A)`

**Scoring Methods:**

| Method | Description |
|---|---|
| `score(text: str) -> float` | Interpolated log-probability: `Σᵢ [λ₁·log_uni + λ₂·log_bi + λ₃·log_tri + λ₄·log_quad]` |
| `score_indices(idx: np.ndarray) -> float` | Same as `score()` but on pre-converted index array |
| `score_quadgrams_fast(idx: np.ndarray) -> float` | Quadgram-only scoring via NumPy fancy indexing — used in MCMC hot loops |

**Serialisation:**

The model saves only `log_*` arrays, `A`, `include_space`, and `lambdas` via `pickle`. Raw count arrays are not persisted.

**`get_model(force_retrain: bool = False, include_space: bool = True) -> NgramModel`** — Loads from cache or trains from corpus.

---

### `ciphers.py` — Cipher Engines

**Purpose:** Encrypt/decrypt implementations for five cipher types, plus a common protocol.

#### `CipherEngine` Protocol

```python
class CipherEngine(Protocol):
    name: str
    def encrypt(self, plaintext: str, key: object) -> str: ...
    def decrypt(self, ciphertext: str, key: object) -> str: ...
    def random_key(self) -> object: ...
```

#### `SimpleSubstitution`

**Key format:** 26-character string — a permutation of `abcdefghijklmnopqrstuvwxyz`. Position `i` maps ciphertext letter `i` to plaintext letter `key[i]`.

**Algorithm:** Uses `str.maketrans()` for O(n) character mapping.

- `random_key() -> str` — Random permutation via `random.sample`
- `key_from_mapping(mapping: dict[str, str]) -> str` — Build key from explicit a→b mappings
- Encrypt: `plaintext.translate(maketrans(alphabet, key))`
- Decrypt: `ciphertext.translate(maketrans(key, alphabet))` (inverse)

#### `Vigenere`

**Key format:** Lowercase keyword string (default length 6).

**Algorithm:** Modular arithmetic shift per character position:
- Encrypt: `c = (p + k) mod 26` (spaces preserved unchanged)
- Decrypt: `p = (c - k) mod 26`

#### `ColumnarTransposition`

**Key format:** `list[int]` — permutation of column indices, e.g., `[2, 0, 3, 1]`.

**Algorithm:**
1. Pad plaintext with `'x'` to fill the grid completely
2. Write plaintext row-by-row into grid of `ncols` columns
3. Read columns in key order to produce ciphertext
4. Decrypt: invert key, split ciphertext into column-length chunks, reassemble row-by-row

#### `Playfair`

**Key format:** 25-character string (5×5 matrix, J merged with I).

**Alphabet:** `"abcdefghiklmnopqrstuvwxyz"` (no J)

**Preparation:**
1. Replace 'j' with 'i'
2. Remove non-alpha characters
3. Pad odd-length text with 'x'
4. Split into digraphs; if both letters are the same, insert 'x' between them

**Encryption Rules (digraph `ab`):**
1. **Same row:** shift right (wrap around)
2. **Same column:** shift down (wrap around)
3. **Rectangle:** swap columns of the two letters

#### `NoisyCipher`

**Wraps any `CipherEngine`** and adds post-encryption noise:
- `corruption_rate: float = 0.05` — Probability of replacing each character with a random letter
- `deletion_rate: float = 0.02` — Probability of deleting each character

Useful for testing robustness of HMM solver (which naturally handles noise).

#### `get_engine(name: str) -> CipherEngine`

Lookup by name (case-insensitive, stripped). Supports `"noisy-*"` prefix to auto-wrap any engine with `NoisyCipher`.

---

### `mcmc.py` — MCMC Solver

**Purpose:** Metropolis-Hastings MCMC solver for monoalphabetic substitution ciphers.

#### Configuration

```python
@dataclass
class MCMCConfig:
    iterations: int = 30_000          # Per chain
    num_restarts: int = 5             # Number of independent chains
    initial_temp: float = 50.0        # Starting temperature
    cooling_rate: float = 0.0001      # Temperature decay per iteration
    callback_interval: int = 500      # Report every N iterations
    track_trajectory: bool = False    # Record (iter, score) trajectory
    early_stop_threshold: int = 10_000  # Stop if no improvement for this many iters
```

#### Result

```python
@dataclass
class MCMCResult:
    best_key: str                     # Best 26-char key found
    best_plaintext: str               # Decrypted text under best key
    best_score: float                 # Log-probability score
    iterations_total: int             # Total iterations across all chains
    chain_scores: list[float]         # Best score per chain
    chain_keys: list[str]             # Best key per chain
    score_trajectory: list[tuple]     # [(iter, score), ...] if tracking enabled
    acceptance_rates: list[float]     # Acceptance rate at sample points
    early_stopped: bool               # Whether early stopping triggered
```

#### Algorithm — `run_mcmc(ciphertext, model, config, callback)`

For each chain (`num_restarts` independent runs):

1. **Initialise** with a random 26-character permutation key
2. **For each iteration** (up to `iterations`):
   a. **Propose** a new key by swapping two random positions (`_swap_key`)
   b. **Score** the proposed decryption using `model.score_quadgrams_fast()`
   c. **Accept/Reject** via Metropolis criterion:
      ```
      ΔS = new_score - current_score
      Accept if ΔS > 0, else accept with probability exp(ΔS / temperature)
      ```
   d. **Cool** the temperature: `temp = max(0.001, initial_temp × exp(-cooling_rate × iter))`
   e. **Early stop** if no improvement for `early_stop_threshold` consecutive iterations
3. **Select** the chain with the highest score as the final result

#### Key Helper Functions

- `_apply_key(ct_idx, key_idx) -> np.ndarray` — Apply substitution key via NumPy indexing
- `_swap_key(key: list[int]) -> list[int]` — Propose a neighbour by swapping two random entries
- `_random_key(A: int) -> list[int]` — Generate random permutation of `[0..A-1]`

---

### `hmm.py` — HMM/EM Solver

**Purpose:** Hidden Markov Model solver using the Baum-Welch (EM) algorithm. Treats plaintext characters as hidden states and ciphertext characters as observations.

#### Configuration

```python
@dataclass
class HMMConfig:
    max_iter: int = 100               # Maximum EM iterations
    tol: float = 1e-4                 # Log-likelihood convergence tolerance
    callback_interval: int = 1        # Report every N iterations
```

#### Algorithm — `run_hmm(ciphertext, model, config, callback)`

**Model:**
- Hidden states: 27 characters (a–z + space) or 26 (no space)
- Observations: Ciphertext characters
- Transition matrix: Initialised from bigram log-probabilities (`model.log_bi`)
- Emission matrix: Initialised as near-uniform with small random perturbation

**Baum-Welch EM Iteration:**
1. **Forward pass (α):** Compute forward log-probabilities for each time step
2. **Backward pass (β):** Compute backward log-probabilities
3. **E-step:** Compute expected state occupancies (γ) and state transitions (ξ)
4. **M-step:** Re-estimate emission probabilities from the posterior
5. **Convergence check:** Stop if `|LL_new - LL_old| < tol`

**Log-space arithmetic:** All computations use `_logsumexp()` for numerical stability.

**Key decryption:** After convergence, decode the most probable state sequence using the Viterbi-like argmax on the emission matrix, then construct a substitution key.

#### Helper Functions

- `_logsumexp(a, axis=None)` — Numerically stable log-sum-exp
- `_normalise_log(log_arr, axis)` — Normalise log-probabilities along an axis

---

### `genetic.py` — Genetic Algorithm Solver

**Purpose:** Evolutionary approach to substitution cipher cracking using crossover, mutation, and selection.

#### Configuration

```python
@dataclass
class GeneticConfig:
    population_size: int = 500        # Number of individuals
    generations: int = 500            # Number of generations
    elite_fraction: float = 0.1       # Top fraction preserved unchanged
    mutation_rate: float = 0.02       # Per-gene swap probability
    tournament_size: int = 5          # Tournament selection size
    callback_interval: int = 10       # Report every N generations
```

#### Algorithm — `run_genetic(ciphertext, model, config, callback)`

1. **Initialise** random population of `population_size` permutation keys
2. **For each generation:**
   a. **Evaluate** fitness = `model.score_quadgrams_fast(decoded_text)` for each individual
   b. **Select** parents via **tournament selection** (pick `tournament_size` random individuals, take best)
   c. **Crossover** using **Order Crossover (OX1):**
      - Pick random segment from parent 1
      - Fill remaining positions from parent 2 in order, preserving permutation validity
   d. **Mutate:** For each gene position, swap with another random position with probability `mutation_rate`
   e. **Elitism:** Top `elite_fraction` individuals survive unchanged into next generation
3. Return the best individual across all generations

#### Key Helper Functions

- `_random_key(A: int) -> np.ndarray` — Random permutation as NumPy array
- `_ox1_crossover(p1, p2)` — Order crossover preserving permutation structure
- `_mutate(key, rate)` — Random swap mutation with given rate
- `_decode(ct_idx, key)` — Apply permutation key to ciphertext indices
- `_idx_to_text(plain_idx, include_space)` — Convert indices back to text

---

### `vigenere_cracker.py` — Vigenère Cracker

**Purpose:** Crack Vigenère ciphers using Index of Coincidence (IoC) for key length detection and frequency analysis for key recovery.

#### Algorithm

1. **Key Length Detection** (`detect_key_length`):
   - For each candidate length `L` in `[2..max_key_length]`:
     - Split ciphertext into `L` groups (every L-th character)
     - Compute average IoC across all groups
   - **Kasiski Examination** (`_kasiski_distances`): Find repeated trigrams and compute GCD of their spacing distances
   - Combine IoC ranking and Kasiski results to determine most likely key length

2. **Key Recovery** (`_solve_caesar`):
   - For each position in the key: try all 26 shifts, pick the one producing the highest n-gram score using `model.score()`

3. **Full Crack** (`crack_vigenere`):
   - Detect key length → recover each key byte → decrypt → return result

#### Configuration

```python
@dataclass
class VigenereConfig:
    max_key_length: int = 20          # Maximum key length to test
    callback: Callable | None = None  # Progress callback
```

#### Key Functions

- `_ioc(text: str) -> float` — Index of Coincidence: `Σ nᵢ(nᵢ-1) / N(N-1)` where nᵢ is the count of letter i
- `detect_key_length(ct, max_len) -> int` — Best key length by IoC + Kasiski
- `_kasiski_distances(ct) -> list[int]` — GCD of repeated trigram spacings
- `_solve_caesar(ct, model) -> tuple[int, float]` — Brute-force single Caesar shift
- `crack_vigenere(ct, model, config, callback) -> VigenereResult` — Full pipeline

---

### `transposition_cracker.py` — Transposition Cracker

**Purpose:** Crack columnar transposition ciphers using MCMC over column permutations.

#### Algorithm

Similar to the substitution MCMC solver but operates on column permutations:

1. For each chain:
   - Start with a random column permutation
   - Propose a neighbour by swapping two column positions
   - Score the resulting decryption with n-gram model
   - Accept/reject via simulated annealing
2. Try key lengths from 2 to `max_cols`

#### Configuration

```python
@dataclass
class TranspositionConfig:
    max_cols: int = 10                # Maximum number of columns to try
    iterations: int = 10_000          # MCMC iterations per key length
    num_restarts: int = 3             # Independent chains per key length
    callback_interval: int = 500
```

---

### `playfair_cracker.py` — Playfair Cracker

**Purpose:** Crack Playfair ciphers using MCMC with Playfair-specific key mutations.

#### Algorithm

1. Start with a random 5×5 Playfair key matrix
2. **Propose moves** (each equally likely):
   - Swap two random letters in the matrix
   - Swap two random rows
   - Swap two random columns
   - Reverse a row
   - Reverse a column
3. Score with n-gram model (no-space model, since Playfair output has no spaces)
4. Accept/reject via simulated annealing

Uses the no-space model (`include_space=False`) since Playfair ciphertext/plaintext has no spaces.

#### Configuration

```python
@dataclass
class PlayfairConfig:
    iterations: int = 50_000
    num_restarts: int = 5
    initial_temp: float = 50.0
    cooling_rate: float = 0.00005
    callback_interval: int = 500
```

---

### `convergence.py` — Convergence Diagnostics

**Purpose:** Post-hoc analysis of MCMC convergence quality with visualization.

#### Functions

**`sparkline_trajectory(result: MCMCResult, width: int = 60) -> str`**

Returns a Unicode sparkline (`▁▂▃▄▅▆▇█`) of the score trajectory. Maps scores to block characters proportional to the min/max range. Width controls the number of characters output.

**`chain_consensus(result: MCMCResult) -> dict`**

Analyses agreement across multiple MCMC chains:
- For each of the 26 cipher positions, counts which plaintext letter each chain assigned
- Uses majority vote to build a consensus key
- `agreement_rate`: fraction of positions where all chains agree
- Returns: `{"agreement_rate", "consensus_key", "per_letter_confidence", "num_chains"}`

**`plot_score_trajectory(result, reference_score, save_path, show) -> Path | None`**

Matplotlib chart showing score vs iteration with:
- Blue line: score trajectory
- Red dashed line: reference/true score (if known)
- Orange secondary Y-axis: acceptance rate over time

**`key_confidence_heatmap(result, save_path) -> Path | None`**

26×26 heatmap where cell (i,j) shows how many chains mapped cipher position i to plaintext letter j. Bright cells indicate high confidence.

**`frequency_comparison_plot(ciphertext, decrypted, save_path) -> Path | None`**

Grouped bar chart comparing letter frequency distributions of:
- Ciphertext (red)
- Decrypted text (blue)
- Expected English frequencies (green dashed)

---

### `evaluation.py` — Evaluation & Benchmarking

**Purpose:** Quantitative evaluation metrics, baseline comparisons, and phase transition analysis.

#### Metrics

**`symbol_error_rate(true_text: str, decrypted_text: str) -> float`**

Character-level error rate: fraction of positions where characters differ. Texts are compared up to the shorter length. Returns `0.0` for perfect decryption, `1.0` for completely wrong.

**`key_accuracy(true_key: str, found_key: str) -> float`**

Fraction of key positions that match: `sum(t == f for t, f in zip(true_key, found_key)) / 26`.

#### Baselines

**`frequency_analysis_baseline(ciphertext: str) -> str`**

Simple frequency-based decryption: sort ciphertext letters by frequency, map to English frequency order (`etaoinshrdlcumwfgypbvkjxqz`). Quick but inaccurate — useful as a lower bound.

**`random_restart_baseline(ciphertext: str, model: NgramModel, restarts: int = 100) -> str`**

Generate `restarts` random keys, score each, return the best. Establishes a naive random-search baseline.

#### Benchmarking

**`run_benchmark(model, corpus, text_length, trials, callback) -> list[BenchmarkEntry]`**

Compares MCMC, Genetic Algorithm, Frequency Analysis, and Random Restarts on the same ciphertexts:
- Generates `trials` random plaintext/key pairs
- Runs each method and records SER, time, and success rate
- Returns list of `BenchmarkEntry` dataclasses

**`run_phase_transition(model, corpus, lengths, trials_per_length, callback) -> PhaseTransitionResult`**

Measures success rate at different ciphertext lengths to identify the **phase transition** — the critical length below which cracking becomes unreliable:
- Tests lengths [50, 75, 100, 125, 150, 200, 300, 500] by default
- Success = SER < 0.05
- Returns threshold length estimate

#### Dataclasses

| Class | Fields |
|---|---|
| `BenchmarkEntry` | `method`, `avg_ser`, `success_rate`, `avg_time`, `trials` |
| `PhaseTransitionResult` | `lengths`, `success_rates`, `avg_ser`, `avg_times`, `threshold_length` |

#### Plot Functions

- `plot_phase_transition(result, save_path)` — Success rate and SER vs ciphertext length
- `plot_benchmark(results, save_path)` — Bar chart comparing methods

---

### `detector.py` — Cipher Type Detection

**Purpose:** Classify unknown ciphertext into one of four types using statistical analysis.

#### Statistical Features Extracted

| Feature | Formula / Description | Diagnostic Value |
|---|---|---|
| **IoC** (Index of Coincidence) | `Σ nᵢ(nᵢ-1) / N(N-1)` | English ~0.067; random ~0.038; low IoC → polyalphabetic |
| **Shannon Entropy** | `-Σ pᵢ log₂(pᵢ)` | Higher entropy → more uniform → Vigenère |
| **Chi-squared** | `Σ (observed - expected)² / expected` vs English frequencies | Low → close to English; high → scrambled |
| **Bigram Repeat Rate** | Fraction of distinct bigrams that appear >1 time | Very high → Playfair; low → transposition |
| **Autocorrelation** | Correlation of character equality at lag k | Peak at lag k → Vigenère key length k |
| **Even/Odd Balance** | Heuristic: even length, no 'j', uniform bigrams | High → Playfair |

#### Classification Rules (`detect_cipher_type`)

A weighted scoring system assigns points to each cipher type based on feature thresholds:

1. **Substitution:** High IoC (>0.055) + spaces present + low chi-squared
2. **Vigenère:** Low IoC (<0.055) + high entropy (>4.0) + autocorrelation peak
3. **Transposition:** High IoC + high chi-squared + low bigram repeat
4. **Playfair:** Medium IoC (0.045–0.060) + even length + no 'j' + no spaces

Returns `DetectionResult` with `predicted_type`, `confidence`, `all_scores`, `features`, and `reasoning`.

---

### `languages.py` — Multi-Language Support

**Purpose:** Support for French, German, and Spanish language models and language detection.

#### Components

**Language Frequency Tables** (`LANGUAGE_FREQ`):

Pre-computed letter frequency distributions for 4 languages, stored as `np.ndarray` of shape `(26,)`:
- English, French, German, Spanish

**Language Corpora** (`LANGUAGE_CORPORA`):

Gutenberg URLs for each language:
- French: Les Misérables, Le Comte de Monte-Cristo, Vingt mille lieues sous les mers
- German: Also sprach Zarathustra, Die Verwandlung, Faust
- Spanish: Don Quijote, Fortunata y Jacinta, La Regenta

**Transliteration Table** (`TRANSLITERATION_TABLE`):

Maps accented characters to their ASCII equivalents (e.g., `àáâäã` → `aaaaa`, `éèêë` → `eeee`).

#### Functions

- `strip_gutenberg_header(text) -> str` — Remove Gutenberg boilerplate (similar to corpus.py)
- `get_language_model(lang, force_retrain) -> NgramModel` — Download, transliterate, train, and cache a language-specific n-gram model
- `detect_language(text) -> tuple[str, dict]` — Chi-squared goodness-of-fit against each language's frequency table; returns best match and all scores
- `get_available_languages() -> list[str]` — Returns `["english", "french", "german", "spanish"]`

---

### `adversarial.py` — Adversarial Stress Testing

**Purpose:** Generate deliberately difficult cipher cracking scenarios to test system robustness.

#### Test Case Types

| Generator | Description | Difficulty |
|---|---|---|
| `_legal_text()` | Dense legal/contract language with unusual word distribution | Hard |
| `_technical_text()` | Technical/scientific writing | Medium |
| `_poetry_text()` | Shakespeare-style poetry with archaic vocabulary | Medium |
| `_repetitive_text()` | Highly skewed character distribution | Hard |
| `_short_text()` | ~65 characters — near phase transition boundary | Hard |
| `_very_short_text()` | 11 characters ("hello world") — below phase transition | Very Hard |

#### Key Generators

| Generator | Strategy |
|---|---|
| `_uniform_key()` | Maps common→common letters: `"etaoinshrdlcumwfgypbvkjxqz"` — minimal frequency disruption |
| `_adversarial_key()` | Maps common letters to rare letters (e→z, t→q, a→x) — maximum frequency disruption |

**`generate_stress_tests() -> list[StressTestCase]`** — Produces 8 test cases combining different text types with different key strategies.

**`run_stress_tests(model, cases, callback) -> list[StressTestResult]`** — Runs MCMC (40k iterations, 6 restarts) against each case, computes SER, and reports pass/fail (SER < 0.10).

---

### `historical.py` — Historical Cipher Challenges

**Purpose:** Curated collection of 6 real/historically-inspired cipher challenges.

#### Challenges

| Name | Type | Difficulty | Source |
|---|---|---|---|
| Caesar Shift (ROT-13) | Substitution | Easy | Classical antiquity |
| Aristocrat Cipher (ACA Style) | Substitution | Medium | ACA/Pride and Prejudice |
| Vigenère Cipher (Bellaso 1553) | Vigenère | Medium | Bellaso/Vigenère tradition |
| WWI Training Cipher | Substitution | Medium | WWI military (reconstructed) |
| Columnar Transposition (WWI Field) | Transposition | Medium | WWI/WWII field cipher |
| Playfair Cipher (Crimean War) | Playfair | Hard | Wheatstone 1854 |

**`get_historical_ciphers() -> list[HistoricalCipher]`** — Returns all 6 challenges.

**`run_historical_challenge(cipher, model, callback) -> dict`** — Routes to the appropriate solver based on `cipher_type`, measures time, computes SER if `known_plaintext` is available.

---

### `display.py` — Rich Terminal Visualisation

**Purpose:** Real-time Rich Live displays for all solver methods.

#### `MCMCDisplay` Class

Full-featured live dashboard with:
- **Header:** Chain number, iteration count, elapsed time, acceptance rate
- **Current Decryption:** Green (good score) or yellow (poor score) text
- **Score Trajectory:** Terminal sparkline showing score evolution
- **Statistics Panel:** Temperature, current/best score, chains done, temperature bar
- **Key Mapping:** Two-row compact display of cipher→plaintext letter mappings
- **Footer:** Best decryption found so far

The `callback()` method updates all internal state (chain, iteration, key, plaintext, score, temperature) and appends to `score_history`.

#### `HMMDisplay` Class

Simpler panel showing iteration count, log-likelihood, convergence sparkline (Unicode blocks ▁▂▃▄▅▆▇█), and current plaintext.

#### `GenericDisplay` Class

Minimal live display used for Vigenère, Transposition, and Playfair crackers. Shows elapsed time, best score, message log (last 10 messages), and best plaintext.

#### `print_result_box(cipher_type, method, key, plaintext, score, elapsed, extra_info)`

Final result presentation with a Rich Table inside a Panel, showing cipher type, method, key, score, time, and optional extra info (e.g., chain agreement rate).

---

### `cli.py` — Command-Line Interface

**Purpose:** 10-subcommand CLI for all system features.

#### Subcommands

| Command | Description | Key Options |
|---|---|---|
| `train` | Download corpus & train language model | `--force` |
| `demo` | Encrypt sample text, then crack live | `--cipher`, `--method` |
| `crack` | Crack user-supplied ciphertext | `--cipher`, `--method`, `--text`, `--file` |
| `analyse` | MCMC with convergence plots | `--text`, `--output` |
| `detect` | Detect cipher type | `--text`, `--file` |
| `benchmark` | Compare MCMC vs GA vs baselines | `--length`, `--trials`, `--output` |
| `phase-transition` | Success rate vs ciphertext length | `--trials`, `--output` |
| `stress-test` | Run adversarial stress tests | (none) |
| `historical` | Run historical cipher challenges | (none) |
| `language` | Multi-language support | `train|detect|list`, `--lang`, `--text`, `--force` |

#### Entry Point

`main(argv)` → `build_parser()` → parse → dispatch to `cmd_*()` function.

The package is registered as `cipher = "cipher.cli:main"` in `pyproject.toml`.

---

## Algorithms Deep Dive

### Metropolis-Hastings MCMC

The core algorithm for substitution ciphers. The key insight is treating cipher cracking as a **combinatorial optimisation** over the space of 26! ≈ 4×10²⁶ possible permutation keys.

**State space:** All permutations of the 26-letter alphabet.

**Proposal distribution:** Uniform random swap of two key positions → O(1) cost to propose a new key and re-score (only affected positions change).

**Acceptance probability:**

$$P(\text{accept}) = \min\left(1, \exp\left(\frac{\Delta S}{T}\right)\right)$$

where $\Delta S = S_{\text{new}} - S_{\text{current}}$ is the score difference and $T$ is the current temperature.

**Simulated annealing schedule:**

$$T(t) = \max\left(0.001,\ T_0 \cdot e^{-\gamma t}\right)$$

with default $T_0 = 50.0$ and $\gamma = 0.0001$.

**Multi-chain strategy:** Run `num_restarts` independent chains from different random initialisations. This helps escape local optima and provides a crude posterior approximation.

### Baum-Welch EM (HMM)

Models the cipher as a Hidden Markov Model where:
- **Hidden states** = plaintext characters
- **Observations** = ciphertext characters
- **Transition matrix** = character bigram probabilities
- **Emission matrix** = cipher substitution table (unknown, to be learned)

The Baum-Welch algorithm alternates:
1. **E-step:** Compute posterior state probabilities using Forward-Backward algorithm
2. **M-step:** Re-estimate emission matrix from posteriors

All computed in log-space for numerical stability. Converges to a local maximum of the data likelihood.

### Genetic Algorithm

Evolutionary optimisation over permutation keys:
1. **Representation:** Each individual is a permutation (NumPy int8 array)
2. **Selection:** Tournament selection (k=5) with elitism (top 10%)
3. **Crossover:** Order Crossover (OX1) — preserves permutation validity
4. **Mutation:** Random swap with per-gene probability 0.02

### Vigenère Cryptanalysis

Classical approach combining:
1. **Kasiski examination:** Find repeated trigrams, compute GCD of spacings → key length
2. **Index of Coincidence:** Average IoC across stride groups → confirm key length
3. **Frequency analysis:** For each key position, try all 26 shifts, pick best by n-gram score

### Phase Transition

Cipher cracking exhibits a **sharp phase transition** as a function of ciphertext length:
- Below ~75 characters: cracking is unreliable (insufficient statistical signal)
- Above ~150 characters: cracking succeeds consistently
- The transition region (75–150) is where success probability rapidly increases from 0 to 1

This is analogous to phase transitions in statistical physics and is measured by `run_phase_transition()`.

---

## Data Flow & Pipeline

```
Gutenberg Texts → download_corpus() → Clean Text → NgramModel.train() → Trained Model
                                                                              │
Ciphertext → detect_cipher_type() → Predicted Type ───────────────────────────┤
                                                                              │
Ciphertext + Model → Solver (MCMC/HMM/GA/Vigenère/Trans/Playfair) → Result   │
                                                                              │
Result → symbol_error_rate() → SER Metric ─┐                                 │
Result → key_accuracy() → Key Metric ──────┤                                 │
Result → convergence diagnostics ───────────┤                                 │
                                            ↓                                 │
                                      Evaluation / Benchmark                  │
```

---

## Configuration & Tuning

### MCMC Tuning Guide

| Parameter | Effect of Increase | Recommended Range |
|---|---|---|
| `iterations` | Better convergence, slower | 20,000–100,000 |
| `num_restarts` | More exploration, linear time increase | 3–10 |
| `initial_temp` | More random exploration early on | 10–100 |
| `cooling_rate` | Faster annealing → more greedy | 0.00001–0.001 |
| `early_stop_threshold` | More patience for late improvements | 5,000–20,000 |

### Genetic Algorithm Tuning

| Parameter | Effect | Recommended Range |
|---|---|---|
| `population_size` | More diversity, slower generations | 200–1000 |
| `generations` | More evolution time | 200–1000 |
| `elite_fraction` | Higher → more exploitation, less exploration | 0.05–0.20 |
| `mutation_rate` | Higher → more exploration | 0.01–0.05 |
| `tournament_size` | Higher → stronger selection pressure | 3–10 |

### N-Gram Interpolation Weights

Default: `(0.05, 0.10, 0.25, 0.60)` for (unigram, bigram, trigram, quadgram).

The heavy quadgram weight makes the model sensitive to 4-character patterns, which provides the strongest signal for English language detection. Lower-order n-grams provide smoothing for novel character sequences.

---

## Performance Characteristics

| Operation | Time Complexity | Typical Duration |
|---|---|---|
| Corpus download (first run) | Network-bound | 30–120s |
| Model training | O(N) over corpus length | 60–120s |
| MCMC single chain (30k iter) | O(iter × text_length) | 5–15s |
| Full MCMC (5 chains) | O(chains × iter × text_length) | 25–75s |
| HMM Baum-Welch (100 iter) | O(iter × A² × text_length) | 10–60s |
| Genetic Algorithm (500 gen, 500 pop) | O(gen × pop × text_length) | 30–90s |
| Vigenère crack | O(max_key_len × 26 × text_length) | 1–5s |
| Cipher type detection | O(text_length) | <1s |
| Phase transition analysis (8 lengths × 3 trials) | O(24 × MCMC time) | 10–30 min |

Memory is dominated by the quadgram log-probability table: `27⁴ × 8 bytes ≈ 4.3 MB` (with space) or `26⁴ × 8 bytes ≈ 3.7 MB` (without space).
