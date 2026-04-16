"""Cipher-solving algorithms."""

from cryptex.solvers.affine import AffineConfig, AffineResult, crack_affine
from cryptex.solvers.genetic import GeneticConfig, GeneticResult, run_genetic
from cryptex.solvers.hmm import HMMConfig, HMMResult, run_hmm
from cryptex.solvers.kpa import KPAResult, known_plaintext_attack
from cryptex.solvers.mcmc import MCMCConfig, MCMCResult, run_mcmc
from cryptex.solvers.playfair import PlayfairConfig, PlayfairResult, crack_playfair
from cryptex.solvers.railfence import RailFenceConfig, RailFenceResult, crack_railfence
from cryptex.solvers.transposition import (
    TranspositionConfig,
    TranspositionResult,
    crack_transposition,
)
from cryptex.solvers.vigenere import VigenereConfig, VigenereResult, crack_vigenere

__all__ = [
    "AffineConfig",
    "AffineResult",
    "GeneticConfig",
    "GeneticResult",
    "HMMConfig",
    "HMMResult",
    "KPAResult",
    "MCMCConfig",
    "MCMCResult",
    "PlayfairConfig",
    "PlayfairResult",
    "RailFenceConfig",
    "RailFenceResult",
    "TranspositionConfig",
    "TranspositionResult",
    "VigenereConfig",
    "VigenereResult",
    "crack_affine",
    "crack_playfair",
    "crack_railfence",
    "crack_transposition",
    "crack_vigenere",
    "known_plaintext_attack",
    "run_genetic",
    "run_hmm",
    "run_mcmc",
]
