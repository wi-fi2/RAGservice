"""Shared fixtures: a deterministic fake embedding model so tests never download weights."""

import hashlib
import re

import numpy as np
import pytest

import app.indexer

DIM = 384


class FakeSentenceTransformer:
    """Hashed bag-of-words embeddings: texts sharing words get high cosine similarity."""

    def __init__(self, *args, **kwargs):
        pass

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        out = np.zeros((len(texts), DIM), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in re.findall(r"\w+", text.lower()):
                out[row, int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM] += 1.0
        if normalize_embeddings:
            out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)
        return out


# app.server builds a VectorIndexer at import time, so swap the model class before anything imports it.
app.indexer.SentenceTransformer = FakeSentenceTransformer


@pytest.fixture(autouse=True)
def _isolated_data_dir(tmp_path, monkeypatch):
    """Indexer defaults to relative data/ paths; keep test indexes out of the repo."""
    monkeypatch.chdir(tmp_path)
