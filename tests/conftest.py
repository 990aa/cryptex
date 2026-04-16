"""Shared fixtures for the Unsupervised Cipher Cracker test suite."""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def trained_model():
    """Return a fully trained NgramModel (with space). Cached for the session."""
    from cryptex.ngram import get_model

    return get_model(include_space=True)


@pytest.fixture(scope="session")
def trained_model_nospace():
    """Return a trained NgramModel without space. Cached for the session."""
    from cryptex.ngram import get_model

    return get_model(include_space=False)


@pytest.fixture(scope="session")
def corpus_text():
    """Return the downloaded and cleaned corpus text."""
    from cryptex.corpus import download_corpus

    return download_corpus()


@pytest.fixture()
def sample_plaintext() -> str:
    return (
        "it is a truth universally acknowledged that a single man in possession "
        "of a good fortune must be in want of a wife"
    )


@pytest.fixture()
def long_plaintext() -> str:
    return (
        "it is a truth universally acknowledged that a single man in possession "
        "of a good fortune must be in want of a wife however little known the "
        "feelings or views of such a man may be on his first entering a "
        "neighbourhood this truth is so well fixed in the minds of the "
        "surrounding families that he is considered as the rightful property "
        "of some one or other of their daughters"
    )
