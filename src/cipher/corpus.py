"""Corpus downloading, caching, and preprocessing.

Downloads several public-domain texts from Project Gutenberg, strips
them to lowercase alphabetic characters (plus optional space), and
concatenates them into a single training string.
"""

from __future__ import annotations

import re
import string
from pathlib import Path

import requests

# -------------------------------------------------------------------
# Project Gutenberg plain-text URLs (mirrors tend to be stable)
# We pick a range of genres / time periods for diversity.
# -------------------------------------------------------------------
GUTENBERG_URLS: list[tuple[str, str]] = [
    ("Pride and Prejudice", "https://www.gutenberg.org/cache/epub/1342/pg1342.txt"),
    ("Moby Dick", "https://www.gutenberg.org/cache/epub/2701/pg2701.txt"),
    ("A Tale of Two Cities", "https://www.gutenberg.org/cache/epub/98/pg98.txt"),
    (
        "Adventures of Sherlock Holmes",
        "https://www.gutenberg.org/cache/epub/1661/pg1661.txt",
    ),
    ("War and Peace", "https://www.gutenberg.org/cache/epub/2600/pg2600.txt"),
    ("Great Expectations", "https://www.gutenberg.org/cache/epub/1400/pg1400.txt"),
    (
        "The Adventures of Tom Sawyer",
        "https://www.gutenberg.org/cache/epub/74/pg74.txt",
    ),
    ("Frankenstein", "https://www.gutenberg.org/cache/epub/84/pg84.txt"),
]

DATA_DIR = Path(__file__).resolve().parent / "data"
CORPUS_FILE = DATA_DIR / "corpus.txt"

# Characters we keep (lowercase letters + space)
ALLOWED_WITH_SPACE = set(string.ascii_lowercase + " ")
ALLOWED_ALPHA_ONLY = set(string.ascii_lowercase)


def _strip_gutenberg_header_footer(text: str) -> str:
    """Remove the Project Gutenberg boilerplate."""
    start_markers = [
        "*** START OF THE PROJECT GUTENBERG",
        "*** START OF THIS PROJECT GUTENBERG",
    ]
    end_markers = [
        "*** END OF THE PROJECT GUTENBERG",
        "*** END OF THIS PROJECT GUTENBERG",
        "End of the Project Gutenberg",
        "End of Project Gutenberg",
    ]
    for m in start_markers:
        idx = text.find(m)
        if idx != -1:
            text = text[idx + len(m) :]
            # skip past the rest of that line
            nl = text.find("\n")
            if nl != -1:
                text = text[nl + 1 :]
            break
    for m in end_markers:
        idx = text.find(m)
        if idx != -1:
            text = text[:idx]
            break
    return text


def download_corpus(force: bool = False, include_space: bool = True) -> str:
    """Download, preprocess, cache, and return the training corpus.

    Parameters
    ----------
    force : bool
        Re-download even if cache exists.
    include_space : bool
        If True keeps spaces (27-char alphabet); otherwise 26 letters only.

    Returns
    -------
    str
        The cleaned corpus string.
    """
    if CORPUS_FILE.exists() and not force:
        return CORPUS_FILE.read_text(encoding="utf-8")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    from rich.progress import Progress

    parts: list[str] = []
    with Progress() as progress:
        task = progress.add_task("[cyan]Downloading corpus…", total=len(GUTENBERG_URLS))
        for title, url in GUTENBERG_URLS:
            progress.update(task, description=f"[cyan]{title}")
            try:
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
                raw = resp.text
            except Exception as exc:
                progress.console.print(f"[yellow]⚠ Skipping {title}: {exc}")
                progress.advance(task)
                continue
            raw = _strip_gutenberg_header_footer(raw)
            parts.append(raw)
            progress.advance(task)

    full_text = "\n".join(parts)

    # Normalise
    full_text = full_text.lower()
    # Collapse whitespace → single space
    full_text = re.sub(r"\s+", " ", full_text)

    allowed = ALLOWED_WITH_SPACE if include_space else ALLOWED_ALPHA_ONLY
    cleaned = "".join(ch for ch in full_text if ch in allowed)

    CORPUS_FILE.write_text(cleaned, encoding="utf-8")
    return cleaned
