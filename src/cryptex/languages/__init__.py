"""Multi-language support for cipher cracking.

Extends the n-gram language model to support French, German, and Spanish
via Project Gutenberg texts. Includes automatic language detection by
comparing ciphertext/plaintext IoC and bigram statistics against each
language model.
"""

from __future__ import annotations

import re
import string
from pathlib import Path

import numpy as np

from cryptex.core.ngram import DATA_DIR, NgramModel


# Corpus URLs for each language (public domain, Project Gutenberg)


LANGUAGE_CORPORA: dict[str, list[tuple[str, str]]] = {
    "french": [
        ("Les Misérables", "https://www.gutenberg.org/cache/epub/17489/pg17489.txt"),
        (
            "Le Comte de Monte-Cristo",
            "https://www.gutenberg.org/cache/epub/17989/pg17989.txt",
        ),
        ("Vingt mille lieues", "https://www.gutenberg.org/cache/epub/5097/pg5097.txt"),
    ],
    "german": [
        ("Faust", "https://www.gutenberg.org/cache/epub/2229/pg2229.txt"),
        ("Die Verwandlung", "https://www.gutenberg.org/cache/epub/22367/pg22367.txt"),
        (
            "Also sprach Zarathustra",
            "https://www.gutenberg.org/cache/epub/7205/pg7205.txt",
        ),
    ],
    "spanish": [
        ("Don Quijote", "https://www.gutenberg.org/cache/epub/2000/pg2000.txt"),
        ("La Regenta", "https://www.gutenberg.org/cache/epub/49836/pg49836.txt"),
    ],
}

# Characters allowed per language (all reduce to a-z + space)
# Non-ASCII accented characters are stripped / transliterated
TRANSLITERATION = str.maketrans(
    "àáâãäåæçèéêëìíîïðñòóôõöùúûüýÿ",
    "aaaaaaeceeeeiiiidnooooouuuuyy",
)


def _model_path(language: str, include_space: bool = True) -> Path:
    suffix = "" if include_space else "_nospace"
    return DATA_DIR / f"ngram_model_{language}{suffix}.pkl"


def _strip_gutenberg(text: str) -> str:
    """Remove Gutenberg boilerplate."""
    for m in [
        "*** START OF THE PROJECT GUTENBERG",
        "*** START OF THIS PROJECT GUTENBERG",
    ]:
        idx = text.find(m)
        if idx != -1:
            text = text[idx + len(m) :]
            nl = text.find("\n")
            if nl != -1:
                text = text[nl + 1 :]
            break
    for m in [
        "*** END OF THE PROJECT GUTENBERG",
        "*** END OF THIS PROJECT GUTENBERG",
        "End of the Project Gutenberg",
        "End of Project Gutenberg",
    ]:
        idx = text.find(m)
        if idx != -1:
            text = text[:idx]
            break
    return text


def download_language_corpus(language: str, force: bool = False) -> str:
    """Download and preprocess corpus for a given language.

    Transliterates accented characters to a-z, strips to lowercase.
    """
    corpus_file = DATA_DIR / f"corpus_{language}.txt"
    if corpus_file.exists() and not force:
        return corpus_file.read_text(encoding="utf-8")

    import requests
    from rich.progress import Progress

    urls = LANGUAGE_CORPORA.get(language, [])
    if not urls:
        raise ValueError(
            f"Unknown language: {language}. Available: {list(LANGUAGE_CORPORA)}"
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    parts: list[str] = []

    with Progress() as progress:
        task = progress.add_task(
            f"[cyan]Downloading {language} corpus…", total=len(urls)
        )
        for title, url in urls:
            progress.update(task, description=f"[cyan]{title}")
            try:
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
                raw = resp.text
            except Exception as exc:
                progress.console.print(f"[yellow]⚠ Skipping {title}: {exc}")
                progress.advance(task)
                continue
            raw = _strip_gutenberg(raw)
            parts.append(raw)
            progress.advance(task)

    full_text = "\n".join(parts)
    full_text = full_text.lower()
    full_text = full_text.translate(TRANSLITERATION)
    full_text = re.sub(r"\s+", " ", full_text)
    cleaned = "".join(
        ch for ch in full_text if ch in string.ascii_lowercase or ch == " "
    )

    corpus_file.write_text(cleaned, encoding="utf-8")
    return cleaned


def get_language_model(
    language: str,
    force_retrain: bool = False,
    include_space: bool = True,
) -> NgramModel:
    """Return a trained n-gram model for the given language."""
    model_path = _model_path(language, include_space)
    if model_path.exists() and not force_retrain:
        return NgramModel.load(model_path)

    corpus = download_language_corpus(language)
    if not include_space:
        corpus = corpus.replace(" ", "")

    model = NgramModel(include_space=include_space)

    from rich.progress import Progress

    with Progress() as progress:
        task = progress.add_task(f"[green]Training {language} model…", total=1)
        model.train(corpus)
        progress.advance(task)

    model.save(model_path)
    return model


# Language Detection


# Approximate letter frequency profiles for quick detection
LANGUAGE_FREQ: dict[str, np.ndarray] = {
    "english": np.array(
        [
            0.08167,
            0.01492,
            0.02782,
            0.04253,
            0.12702,
            0.02228,
            0.02015,
            0.06094,
            0.06966,
            0.00153,
            0.00772,
            0.04025,
            0.02406,
            0.06749,
            0.07507,
            0.01929,
            0.00095,
            0.05987,
            0.06327,
            0.09056,
            0.02758,
            0.00978,
            0.02360,
            0.00150,
            0.01974,
            0.00074,
        ]
    ),
    "french": np.array(
        [
            0.07636,
            0.00901,
            0.03260,
            0.03669,
            0.14715,
            0.01066,
            0.00866,
            0.00737,
            0.07529,
            0.00613,
            0.00049,
            0.05456,
            0.02968,
            0.07095,
            0.05796,
            0.02521,
            0.01362,
            0.06553,
            0.07948,
            0.07244,
            0.06311,
            0.01838,
            0.00074,
            0.00427,
            0.00128,
            0.00326,
        ]
    ),
    "german": np.array(
        [
            0.06516,
            0.01886,
            0.02732,
            0.05076,
            0.16396,
            0.01656,
            0.03009,
            0.04577,
            0.06550,
            0.00268,
            0.01417,
            0.03437,
            0.02534,
            0.09776,
            0.02594,
            0.00670,
            0.00018,
            0.07003,
            0.07270,
            0.06154,
            0.04166,
            0.00846,
            0.01921,
            0.00034,
            0.00039,
            0.01134,
        ]
    ),
    "spanish": np.array(
        [
            0.11525,
            0.02215,
            0.04019,
            0.05010,
            0.12181,
            0.00692,
            0.01768,
            0.00703,
            0.06247,
            0.00493,
            0.00011,
            0.04967,
            0.03157,
            0.06712,
            0.08683,
            0.02510,
            0.00877,
            0.06871,
            0.07977,
            0.04632,
            0.02927,
            0.01138,
            0.00017,
            0.00215,
            0.01008,
            0.00467,
        ]
    ),
}


def detect_language(text: str) -> tuple[str, dict[str, float]]:
    """Detect the most likely language of a text using chi-squared frequency analysis.

    Parameters
    ----------
    text : str
        Input text (plaintext or decrypted text).

    Returns
    -------
    (best_language, scores_dict)
        The best language and chi-squared scores for each language (lower = better match).
    """
    counts = np.zeros(26)
    for ch in text.lower():
        if ch in string.ascii_lowercase:
            counts[ord(ch) - ord("a")] += 1
    total = counts.sum()
    if total == 0:
        return "english", {}

    observed = counts / total
    scores: dict[str, float] = {}

    for lang, expected in LANGUAGE_FREQ.items():
        chi2 = float(((observed - expected) ** 2 / (expected + 1e-10)).sum())
        scores[lang] = chi2

    best = min(scores, key=lambda k: scores[k])
    return best, scores


def get_available_languages() -> list[str]:
    """Return list of supported languages."""
    return ["english"] + list(LANGUAGE_CORPORA.keys())
