"""Cipher engines — encrypt / decrypt for all five cipher types.

Each cipher exposes:
    encrypt(plaintext, key) -> ciphertext
    decrypt(ciphertext, key) -> plaintext  (where key is known)
    random_key()                           -> generate a random valid key

Level 1: Simple Substitution
Level 2: Vigenère
Level 3: Columnar Transposition
Level 4: Playfair
Level 5: Noisy / Partial ciphertext (wraps any cipher)
"""

from __future__ import annotations

import random
import string
from typing import Protocol


# =====================================================================
# Protocol (interface)
# =====================================================================


class CipherEngine(Protocol):
    """Common interface for all cipher types."""

    name: str

    def encrypt(self, plaintext: str, key: object) -> str: ...
    def decrypt(self, ciphertext: str, key: object) -> str: ...
    def random_key(self) -> object: ...


# =====================================================================
# Level 1 — Simple Substitution Cipher
# =====================================================================


class SimpleSubstitution:
    """Mono-alphabetic substitution cipher.

    Key is a permutation of the 26 lowercase letters.
    """

    name = "Simple Substitution"

    @staticmethod
    def random_key() -> str:
        letters = list(string.ascii_lowercase)
        random.shuffle(letters)
        return "".join(letters)

    @staticmethod
    def encrypt(plaintext: str, key: str) -> str:
        table = str.maketrans(string.ascii_lowercase, key)
        return plaintext.translate(table)

    @staticmethod
    def decrypt(ciphertext: str, key: str) -> str:
        table = str.maketrans(key, string.ascii_lowercase)
        return ciphertext.translate(table)

    @staticmethod
    def key_from_mapping(mapping: dict[str, str]) -> str:
        """Build a 26-char key string from a {plain: cipher} dict."""
        return "".join(mapping.get(ch, ch) for ch in string.ascii_lowercase)


# =====================================================================
# Level 2 — Vigenère Cipher
# =====================================================================


class Vigenere:
    """Poly-alphabetic Vigenère cipher.

    Key is a lowercase keyword string (e.g. 'secret').
    """

    name = "Vigenère"

    @staticmethod
    def random_key(length: int = 6) -> str:
        return "".join(random.choices(string.ascii_lowercase, k=length))

    @staticmethod
    def encrypt(plaintext: str, key: str) -> str:
        out: list[str] = []
        ki = 0
        for ch in plaintext:
            if ch in string.ascii_lowercase:
                shift = ord(key[ki % len(key)]) - ord("a")
                out.append(chr((ord(ch) - ord("a") + shift) % 26 + ord("a")))
                ki += 1
            else:
                out.append(ch)  # pass through spaces etc.
        return "".join(out)

    @staticmethod
    def decrypt(ciphertext: str, key: str) -> str:
        out: list[str] = []
        ki = 0
        for ch in ciphertext:
            if ch in string.ascii_lowercase:
                shift = ord(key[ki % len(key)]) - ord("a")
                out.append(chr((ord(ch) - ord("a") - shift) % 26 + ord("a")))
                ki += 1
            else:
                out.append(ch)
        return "".join(out)


# =====================================================================
# Level 3 — Columnar Transposition Cipher
# =====================================================================


class ColumnarTransposition:
    """Columnar transposition cipher.

    Key is a permutation of column indices, e.g. [2, 0, 3, 1].
    """

    name = "Columnar Transposition"

    @staticmethod
    def random_key(ncols: int = 6) -> list[int]:
        key = list(range(ncols))
        random.shuffle(key)
        return key

    @staticmethod
    def encrypt(plaintext: str, key: list[int]) -> str:
        ncols = len(key)
        # Pad plaintext so it fills a complete grid
        remainder = len(plaintext) % ncols
        if remainder:
            plaintext += "x" * (ncols - remainder)
        nrows = len(plaintext) // ncols

        # Build grid (row-major)
        grid = [plaintext[r * ncols : (r + 1) * ncols] for r in range(nrows)]

        # Read columns in key-order
        out: list[str] = []
        for col in key:
            for row in grid:
                out.append(row[col])
        return "".join(out)

    @staticmethod
    def decrypt(ciphertext: str, key: list[int]) -> str:
        ncols = len(key)
        nrows = len(ciphertext) // ncols

        # Invert the key
        inv_key = [0] * ncols
        for i, k in enumerate(key):
            inv_key[k] = i

        # Split ciphertext into columns (each of length nrows)
        cols: list[str] = []
        for i in range(ncols):
            start = i * nrows
            cols.append(ciphertext[start : start + nrows])

        # Rebuild plaintext row by row
        out: list[str] = []
        for r in range(nrows):
            for c in range(ncols):
                # Column inv_key[c] was written at position c in the key-order
                out.append(cols[inv_key[c]][r])
        return "".join(out)


# =====================================================================
# Level 4 — Playfair Cipher
# =====================================================================


class Playfair:
    """Playfair digraph substitution cipher.

    Key is a 25-character string forming the 5×5 matrix
    (j is merged with i).
    """

    name = "Playfair"
    ALPHABET = "abcdefghiklmnopqrstuvwxyz"  # no j

    @classmethod
    def random_key(cls) -> str:
        letters = list(cls.ALPHABET)
        random.shuffle(letters)
        return "".join(letters)

    @classmethod
    def _make_matrix(
        cls, key: str
    ) -> tuple[list[list[str]], dict[str, tuple[int, int]]]:
        matrix = [list(key[i * 5 : (i + 1) * 5]) for i in range(5)]
        pos: dict[str, tuple[int, int]] = {}
        for r in range(5):
            for c in range(5):
                pos[matrix[r][c]] = (r, c)
        return matrix, pos

    @classmethod
    def _prepare(cls, text: str) -> str:
        """Prepare text: replace j→i, insert x between duplicate pairs, pad."""
        text = text.replace("j", "i")
        text = "".join(ch for ch in text if ch in cls.ALPHABET)
        # Insert 'x' between duplicate-letter pairs
        out: list[str] = []
        i = 0
        while i < len(text):
            out.append(text[i])
            if i + 1 < len(text):
                if text[i] == text[i + 1]:
                    out.append("x")
                    i += 1
                else:
                    out.append(text[i + 1])
                    i += 2
            else:
                out.append("x")  # pad odd-length
                i += 1
        return "".join(out)

    @classmethod
    def encrypt(cls, plaintext: str, key: str) -> str:
        matrix, pos = cls._make_matrix(key)
        prepared = cls._prepare(plaintext)
        out: list[str] = []
        for i in range(0, len(prepared), 2):
            a, b = prepared[i], prepared[i + 1]
            r1, c1 = pos[a]
            r2, c2 = pos[b]
            if r1 == r2:
                out.append(matrix[r1][(c1 + 1) % 5])
                out.append(matrix[r2][(c2 + 1) % 5])
            elif c1 == c2:
                out.append(matrix[(r1 + 1) % 5][c1])
                out.append(matrix[(r2 + 1) % 5][c2])
            else:
                out.append(matrix[r1][c2])
                out.append(matrix[r2][c1])
        return "".join(out)

    @classmethod
    def decrypt(cls, ciphertext: str, key: str) -> str:
        matrix, pos = cls._make_matrix(key)
        out: list[str] = []
        for i in range(0, len(ciphertext), 2):
            a, b = ciphertext[i], ciphertext[i + 1]
            r1, c1 = pos[a]
            r2, c2 = pos[b]
            if r1 == r2:
                out.append(matrix[r1][(c1 - 1) % 5])
                out.append(matrix[r2][(c2 - 1) % 5])
            elif c1 == c2:
                out.append(matrix[(r1 - 1) % 5][c1])
                out.append(matrix[(r2 - 1) % 5][c2])
            else:
                out.append(matrix[r1][c2])
                out.append(matrix[r2][c1])
        return "".join(out)


# =====================================================================
# Level 5 — Noisy / Partial Ciphertext
# =====================================================================


class NoisyCipher:
    """Wraps any cipher engine and adds random noise.

    Some characters are replaced with random letters or removed entirely
    to simulate a corrupted / partial ciphertext.
    """

    name = "Noisy Cipher"

    def __init__(
        self,
        inner: CipherEngine,
        corruption_rate: float = 0.05,
        deletion_rate: float = 0.02,
    ):
        self.inner = inner
        self.corruption_rate = corruption_rate
        self.deletion_rate = deletion_rate

    def random_key(self) -> object:
        return self.inner.random_key()

    def encrypt(self, plaintext: str, key: object) -> str:
        ct = self.inner.encrypt(plaintext, key)
        out: list[str] = []
        for ch in ct:
            r = random.random()
            if r < self.deletion_rate:
                continue  # drop character
            if r < self.deletion_rate + self.corruption_rate:
                out.append(random.choice(string.ascii_lowercase))
            else:
                out.append(ch)
        return "".join(out)

    def decrypt(self, ciphertext: str, key: object) -> str:
        """Best-effort decrypt (no noise reversal possible)."""
        return self.inner.decrypt(ciphertext, key)


# =====================================================================
# Convenience
# =====================================================================

ALL_CIPHERS: dict[str, CipherEngine] = {
    "substitution": SimpleSubstitution(),  # type: ignore[abstract]
    "vigenere": Vigenere(),  # type: ignore[abstract]
    "transposition": ColumnarTransposition(),  # type: ignore[abstract]
    "playfair": Playfair(),  # type: ignore[abstract]
}


def get_engine(name: str) -> CipherEngine:
    name = name.lower().strip()
    if name in ALL_CIPHERS:
        return ALL_CIPHERS[name]
    # Also accept with "noisy" prefix
    if name.startswith("noisy"):
        inner_name = name.replace("noisy", "").strip().strip("-_ ")
        inner = ALL_CIPHERS.get(inner_name, SimpleSubstitution())  # type: ignore[abstract]
        return NoisyCipher(inner)  # type: ignore[return-value]
    raise ValueError(f"Unknown cipher: {name!r}. Choose from {list(ALL_CIPHERS)}")
