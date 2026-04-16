"""Compatibility exports for Playfair solver."""

from cryptex.solvers.playfair import *  # noqa: F401,F403
from cryptex.solvers.playfair import _build_decrypt_table, _propose

__all__ = [
	"PlayfairConfig",
	"PlayfairResult",
	"crack_playfair",
	"_build_decrypt_table",
	"_propose",
]
