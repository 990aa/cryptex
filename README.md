# Unsupervised Cipher Cracking

A system that cracks classical ciphers using statistical language models, MCMC (Metropolis-Hastings), HMM (Baum-Welch), and Genetic Algorithms — with real-time terminal visualisation, convergence diagnostics, and evaluation benchmarks.

Feed it encrypted gibberish and watch it converge on readable English.

## Requirements

- **Python 3.14+**
- **uv** package manager ([install guide](https://docs.astral.sh/uv/getting-started/installation/))

## Quick Start

```bash
# Clone and enter the project
git clone <repo-url>
cd cipher

# Install dependencies (uv handles the virtual environment automatically)
uv sync

# Train the language model (downloads corpus from Project Gutenberg, ~30s first time)
uv run cipher train

# Run a live demo — encrypts sample text, then cracks it before your eyes
uv run cipher demo --cipher substitution
uv run cipher demo --cipher vigenere
uv run cipher demo --cipher transposition
uv run cipher demo --cipher playfair

# Crack your own ciphertext
uv run cipher crack --cipher substitution --text "your ciphertext here"
uv run cipher crack --cipher vigenere --text "your ciphertext here"
uv run cipher crack --cipher substitution --method hmm --text "ciphertext"
uv run cipher crack --cipher substitution --method genetic --text "ciphertext"

# Read ciphertext from a file
uv run cipher crack --cipher substitution --file secret.txt
```

## All Commands

| Command | Description |
|---------|-------------|
| `cipher train` | Download corpus and train the n-gram language model |
| `cipher demo` | Encrypt sample text and crack it live |
| `cipher crack` | Crack user-supplied ciphertext |
| `cipher analyse` | Run MCMC with convergence analysis and generate diagnostic plots |
| `cipher detect` | Detect cipher type from unknown ciphertext |
| `cipher benchmark` | Benchmark MCMC vs GA vs frequency analysis vs random restarts |
| `cipher phase-transition` | Analyse success rate vs ciphertext length |
| `cipher stress-test` | Run adversarial stress tests |
| `cipher historical` | Run historical cipher challenges |
| `cipher language` | Multi-language support (train, detect, list) |

### Demo & Crack Options

```bash
# Cipher types: substitution, vigenere, transposition, playfair, noisy-substitution
uv run cipher demo --cipher playfair

# Solving methods: mcmc (default), hmm, genetic
uv run cipher demo --method genetic

# Crack with genetic algorithm
uv run cipher crack --cipher substitution --method genetic --text "..."
```

### Convergence Analysis

```bash
# Auto-generated demo with full diagnostics
uv run cipher analyse

# Analyse your own ciphertext
uv run cipher analyse --text "your ciphertext"

# Save plots to a specific directory
uv run cipher analyse --output ./plots
```

Generates:
- `convergence_plot.png` — Score trajectory across all chains with acceptance rate
- `key_heatmap.png` — 26×26 key confidence heatmap from multi-chain consensus
- `frequency_comparison.png` — Letter frequency: English vs ciphertext vs decrypted

### Cipher Type Detection

```bash
uv run cipher detect --text "wkh txlfn eurzq ira"
uv run cipher detect --file mystery.txt
```

Uses statistical features (Index of Coincidence, entropy, chi-squared, bigram repeat rate, autocorrelation) to classify ciphertext as substitution, Vigenère, transposition, or Playfair.

### Benchmarking

```bash
# Default: 300-char texts, 3 trials per method
uv run cipher benchmark

# Custom settings
uv run cipher benchmark --length 200 --trials 5 --output ./results
```

Compares MCMC, Genetic Algorithm, frequency analysis, and random restarts on identical test cases. Generates a grouped bar chart (`benchmark.png`).

### Phase Transition Analysis

```bash
uv run cipher phase-transition --trials 3 --output ./results
```

Tests MCMC success rate at lengths [50, 75, 100, 125, 150, 200, 300, 500] to empirically find the minimum ciphertext length needed for reliable decryption.

### Stress Testing

```bash
uv run cipher stress-test
```

Runs adversarial test cases: legal text, technical prose, poetry, repetitive patterns, short texts (65 chars), very short texts (11 chars), adversarial key mappings, and near-identity keys.

### Historical Ciphers

```bash
uv run cipher historical
```

Attempts to crack curated historical ciphers: Caesar/ROT-13, ACA Aristocrat, Vigenère (Bellaso 1553), WWI training cipher, columnar transposition, and Playfair.

### Multi-Language Support

```bash
# List available languages
uv run cipher language list

# Train a French language model (downloads from Project Gutenberg)
uv run cipher language train --lang french

# Detect language from text
uv run cipher language detect --text "bonjour le monde ceci est un test"
```

Supported: English (built-in), French, German, Spanish. Each language model is trained from Gutenberg texts with accent transliteration.

## Architecture

```
src/cipher/
├── __init__.py
├── cli.py                   # CLI entry point — 10 subcommands
├── corpus.py                # Downloads & preprocesses Project Gutenberg texts
├── ngram.py                 # Quadgram language model with interpolated smoothing
├── ciphers.py               # 5 cipher engines (encrypt / decrypt / random_key)
├── mcmc.py                  # MCMC Metropolis-Hastings solver (simulated annealing)
├── hmm.py                   # HMM / Baum-Welch (EM) solver
├── genetic.py               # Genetic algorithm solver (tournament + OX1 crossover)
├── vigenere_cracker.py      # IoC + Kasiski + frequency analysis
├── transposition_cracker.py # MCMC over column permutations
├── playfair_cracker.py      # MCMC over 5×5 key matrix
├── display.py               # Rich-powered real-time terminal visualisation
├── convergence.py           # Convergence diagnostics, plots, chain consensus
├── evaluation.py            # SER, phase transition, benchmarking framework
├── detector.py              # Rule-based cipher type classifier
├── languages.py             # Multi-language corpus & model support
├── adversarial.py           # Adversarial stress test suite
└── historical.py            # Curated historical cipher challenges
```

## Cipher Complexity Progression

| Level | Cipher | Method | Status |
|-------|--------|--------|--------|
| 1 | Simple Substitution | MCMC (simulated annealing) | ✅ Solves in ~15s |
| 1 | Simple Substitution | HMM / Baum-Welch (EM) | ✅ Partial convergence |
| 1 | Simple Substitution | Genetic Algorithm | ✅ Converges (~60s) |
| 2 | Vigenère | IoC + Kasiski + frequency analysis | ✅ Instant |
| 3 | Columnar Transposition | MCMC over column permutations | ✅ Solves in ~40s |
| 4 | Playfair (digraph) | MCMC over 5×5 key matrix | ⚠️ Hard — needs Rust for speed |
| 5 | Noisy / Partial | HMM handles probabilistic emissions | ✅ Demonstrated |

## Technical Details

### N-gram Language Model
- Trained on 8 books from Project Gutenberg (~6M+ characters)
- Quadgram scoring with interpolated smoothing (unigram → quadgram)
- 27-character alphabet (a-z + space) stored as NumPy arrays for O(1) lookup
- Separate no-space model for Playfair scoring
- Vectorised scoring with NumPy fancy indexing (~100k scores/sec)

### MCMC Solver
- Metropolis-Hastings with simulated annealing (geometric cooling)
- Temperature scaled to match score magnitudes (T₀ = 50)
- Multiple random-restart chains (8 chains × 50k iterations)
- Vectorised decrypt via inverse-mapping arrays (no string operations in hot loop)
- **Early stopping**: patience-based (15k iterations without improvement) and optional score threshold
- **Convergence tracking**: score trajectory, per-chain acceptance rates
- **Chain consensus**: majority-vote key reconstruction with per-letter confidence
- Real-time Rich terminal display showing score sparkline, key mapping grid, acceptance rate

### HMM Solver
- Forward-Backward algorithm for posterior state computation
- Baum-Welch EM: learns emission probabilities while keeping transitions fixed
- Naturally handles noisy/probabilistic ciphers (Level 5)
- Transitions from bigram language model (prior knowledge of English)

### Genetic Algorithm Solver
- Population-based search over substitution key permutations
- **Selection**: tournament selection (size 5)
- **Crossover**: Order Crossover (OX1) preserving permutation validity
- **Mutation**: random swap mutations
- **Elitism**: top 10% preserved across generations
- Configurable: 500 population × 1000 generations (default)
- Same vectorised decode pattern as MCMC for performance parity

### Vigenère Cracker
- Key length detection via Index of Coincidence
- Kasiski examination for cross-validation
- Per-position Caesar shift solved by chi-squared frequency analysis

### Transposition Cracker
- Tries all valid column counts (divisors of ciphertext length)
- MCMC with column-pair swaps and simulated annealing

### Playfair Cracker
- MCMC over 25! permutations of the 5×5 key matrix
- Diverse proposals: cell swaps, row/column swaps, rotations, reversal
- Optimised with precomputed 25×25 digraph decrypt lookup table
- Uses no-space language model for better discrimination

### Cipher Type Detector
- Statistical feature extraction: IoC, entropy, chi-squared, bigram repeat rate, autocorrelation, even/odd character balance
- Rule-based weighted classifier distinguishing substitution, Vigenère, transposition, and Playfair
- Confidence scoring with reasoning explanations

### Evaluation Framework
- **Symbol Error Rate (SER)**: character-level accuracy metric
- **Phase transition analysis**: empirically determines minimum ciphertext length for reliable cracking
- **Benchmarking**: head-to-head comparison of MCMC, GA, frequency analysis, and random restarts
- Matplotlib-generated publication-quality plots

## MCMC vs HMM vs Genetic Algorithm

| Aspect | MCMC | HMM/EM | Genetic Algorithm |
|--------|------|--------|-------------------|
| Approach | Optimisation (MAP) | Sequence inference | Population search |
| Speed | Fast (vectorised) | Slower (O(T·A²)) | Medium |
| Clean ciphers | Excellent | Partial | Good |
| Noisy ciphers | Fragile | Robust | Fragile |
| Exploration | Random restarts | EM convergence | Population diversity |
| Theory basis | Bayesian posterior | Expectation-Maximisation | Evolutionary optimisation |

MCMC is the practical winner for deterministic substitution ciphers. HMM/EM shines when the cipher introduces probabilistic noise. The genetic algorithm provides an alternative population-based approach with natural diversity through crossover.

## Theoretical Connections

### Bayesian Inference
The MCMC solver samples from the posterior distribution P(key | ciphertext) ∝ P(ciphertext | key) · P(key), where the likelihood is the language model score and the prior is uniform over permutations. Simulated annealing concentrates the sampling on the MAP estimate.

### Information Theory
The phase transition in ciphertext length connects to the *unicity distance* — the minimum text length where the true key becomes uniquely determined. For 26-letter substitution ciphers, theory predicts ~25 characters; in practice, statistical methods need ~100+ characters for reliable convergence.

### Modern Cryptography
These classical ciphers are educationally valuable but trivially broken by modern standards. AES, RSA, and elliptic curve cryptography resist statistical attacks because they destroy all frequency structure. This project demonstrates *why* modern ciphers were necessary — and the probabilistic tools that drove that evolution.

## License

MIT
