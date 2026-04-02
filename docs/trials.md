# Trial Log (2026-04-02)

This file records successful command runs and successful deciphering trials executed from `docs/TESTING.md` plus additional custom-text validation runs.

## Environment

- OS: Windows
- Working directory: project root
- Runner: `uv run ...`

## Successful Script Runs (Command Completed)

1. `uv run cipher train`
- Output: `Models ready`, both space and no-space models loaded.

2. `uv run pytest tests/ -v --timeout=300`
- Output: `251 passed in 182.75s`.

3. `uv run cipher demo --cipher substitution --method mcmc`
- Output: final result box printed with decrypted English text.

4. `uv run cipher demo --cipher substitution --method hmm`
- Output: completed 100 iterations and printed final result box.

5. `uv run cipher demo --cipher substitution --method genetic`
- Output: completed 1000 generations and printed final result box.

6. `uv run cipher demo --cipher vigenere`
- Output: recovered key and printed decrypted English text.

7. `uv run cipher demo --cipher transposition`
- Output: recovered column key and printed decrypted English text.

8. `uv run cipher demo --cipher playfair`
- Output: completed all chains and printed final result box.

9. `uv run cipher demo --cipher noisy-substitution --method hmm`
- Output: completed 100 iterations and printed final result box.

10. `uv run cipher detect --text "wkh txlfn eurzq ira mxpsv ryhu wkh odcb grj dqg ihow txlwh jrrg derxw lw"`
- Output: predicted type table printed.

11. `uv run cipher detect --text "lxfopvefrnhr"`
- Output: predicted type table printed.

12. `uv run cipher analyse --output ./plots`
- Output: saved `plots/convergence_plot.png`, `plots/key_heatmap.png`, `plots/frequency_comparison.png`.

13. `uv run cipher benchmark --trials 3 --length 300 --output ./plots`
- Output: benchmark table printed and `plots/benchmark.png` saved.

14. `uv run cipher phase-transition --trials 2 --output ./plots`
- Output: phase-transition table printed and `plots/phase_transition.png` saved.

15. `uv run cipher stress-test`
- Output: full stress-test table printed (8 cases).

16. `uv run cipher historical`
- Output: all 6 historical challenges executed and reported.

17. `uv run cipher language list`
- Output: english, french, german, spanish listed.

18. `uv run cipher language train --lang french`
- Output: `Done! Alphabet size: 27`.

19. `uv run cipher language detect --text "bonjour le monde comment allez vous"`
- Output: language detection scores printed.

20. `uv run cipher language detect --text "the quick brown fox jumps over the lazy dog"`
- Output: language detection scores printed.

21. `uv run cipher crack --cipher substitution --method mcmc --text "wkh txlfn eurzq ira mxpsv ryhu wkh odcb grj dqg ihow txlwh jrrg derxw lw"`
- Output: completed run with final result box.

22. `echo "wkh txlfn eurzq ira mxpsv ryhu wkh odcb grj" > secret.txt; uv run cipher crack --cipher substitution --text "$(cat secret.txt)"`
- Output: completed run with final result box.

23. `uv run cipher crack --cipher substitution --file secret.txt`
- Output: completed run with final result box.

24. `uv run cipher crack --cipher substitution --method mcmc --file custom_ciphertext.txt`
- Output: completed run with final result box.

25. `uv run cipher crack --cipher vigenere --file custom_vigenere_ciphertext.txt`
- Output: recovered key `cipher` and printed decrypted custom plaintext.

26. `New-Item -ItemType Directory -Force results | Out-Null; uv run cipher benchmark --length 200 --trials 1 --output ./results`
- Output: benchmark completed and `results/benchmark.png` saved.

27. `uv run cipher phase-transition --trials 1 --output ./results`
- Output: phase-transition completed and `results/phase_transition.png` saved.

## Successful Deciphering Confirmations

1. Substitution demo (MCMC)
- Command: `uv run cipher demo --cipher substitution --method mcmc`
- Output plaintext started with: `it is a truth universally acknowledged...`
- Confirmation: successful deciphering of demo text.

2. Vigenere demo
- Command: `uv run cipher demo --cipher vigenere`
- Output key: `abrqsk`
- Output plaintext started with: `it is a truth universally acknowledged...`
- Confirmation: successful deciphering of demo text.

3. Transposition demo
- Command: `uv run cipher demo --cipher transposition`
- Output key: `[0, 3, 4, 1, 2, 5]`
- Output plaintext started with: `it is a truth universally acknowledged...`
- Confirmation: successful deciphering of demo text.

4. Analysis mode (MCMC)
- Command: `uv run cipher analyse --output ./plots`
- Output metric: `Symbol Error Rate: 0.0%`
- Confirmation: successful deciphering of generated analysis sample.

5. Historical challenge (Caesar)
- Command: `uv run cipher historical`
- Output metric for Caesar case: `SER: 0.0% PASS`
- Confirmation: successful deciphering for at least one historical challenge.

6. Custom text (Vigenere file)
- Command: `uv run cipher crack --cipher vigenere --file custom_vigenere_ciphertext.txt`
- Output key: `cipher`
- Output plaintext:
  `this is custom text for vigenere cracking validation and it should decrypt back to the exact original message when the model has enough context from repeated english patterns and spaces`
- Confirmation: successful deciphering of custom ciphertext.

7. Exact custom-text match check
- Command:
  `uv run python -c "from cipher.ngram import get_model; from cipher.vigenere_cracker import crack_vigenere, VigenereConfig; from pathlib import Path; ct=Path('custom_vigenere_ciphertext.txt').read_text().strip(); pt=Path('custom_vigenere_plaintext.txt').read_text().strip(); res=crack_vigenere(ct, get_model(), VigenereConfig()); print('key=',res.best_key); print('exact_match=', res.best_plaintext.strip()==pt); print('decrypted=', res.best_plaintext.strip())"`
- Output: `key= cipher`, `exact_match= True`
- Confirmation: exact plaintext recovery confirmed programmatically.
