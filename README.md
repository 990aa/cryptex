# Unsupervised Cipher Cracking

A system that cracks classical ciphers using statistical language models, MCMC (Metropolis-Hastings), and HMM (Baum-Welch) — with real-time terminal visualisation.

Feed it encrypted gibberish and watch it converge on readable English.

## Quick Start

```bash
# Train the language model (downloads corpus from Project Gutenberg)
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

# Read ciphertext from a file
uv run cipher crack --cipher substitution --file secret.txt
```

## Architecture

```
src/cipher/
├── __init__.py
├── cli.py                  # CLI entry point (train / demo / crack)
├── corpus.py               # Downloads & preprocesses Project Gutenberg texts
├── ngram.py                # Quadgram language model with interpolated smoothing
├── ciphers.py              # All 5 cipher engines (encrypt / decrypt / random_key)
├── mcmc.py                 # MCMC Metropolis-Hastings solver for substitution
├── hmm.py                  # HMM / Baum-Welch (EM) solver for substitution
├── vigenere_cracker.py     # IoC + Kasiski + frequency analysis
├── transposition_cracker.py# MCMC over column permutations
├── playfair_cracker.py     # MCMC over 5×5 key matrix
└── display.py              # Rich-powered real-time terminal visualisation
```

## Cipher Complexity Progression

| Level | Cipher | Method | Status |
|-------|--------|--------|--------|
| 1 | Simple Substitution | MCMC (simulated annealing) | ✅ Solves in ~15s |
| 1 | Simple Substitution | HMM / Baum-Welch (EM) | ✅ Partial convergence |
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
- Real-time Rich terminal display showing convergence

### HMM Solver
- Forward-Backward algorithm for posterior state computation
- Baum-Welch EM: learns emission probabilities while keeping transitions fixed
- Naturally handles noisy/probabilistic ciphers (Level 5)
- Transitions from bigram language model (prior knowledge of English)

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

## MCMC vs HMM: Comparative Analysis

| Aspect | MCMC | HMM/EM |
|--------|------|--------|
| Approach | Optimisation (MAP) | Sequence inference |
| Speed | Fast (vectorised) | Slower (O(T·A²) per EM step) |
| Clean ciphers | Excellent | Partial |
| Noisy ciphers | Fragile | Robust |
| Theoretical basis | Bayesian posterior sampling | Expectation-Maximisation |

MCMC is the practical winner for deterministic substitution ciphers. HMM/EM shines
when the cipher introduces probabilistic noise, because the emission model naturally
accommodates non-deterministic mappings.

## License

MIT
