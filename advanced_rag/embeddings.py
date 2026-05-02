"""Lazy MiniLM wrapper. The model isn't loaded until the first encode() call.

Tests inject a stub embedder rather than this real one — keeps `sentence-transformers`
out of the dev install.
"""
from __future__ import annotations

import numpy as np

from .config import EMBED_MODEL, EMBED_MODEL_DIMS


class Embedder:
    def __init__(self, model_name: str = EMBED_MODEL):
        self._model_name = model_name
        self._model = None

    @property
    def dim(self) -> int | None:
        """Vector dimension declared in EMBED_MODEL_DIMS, or None for unknown
        models. Used to shape the empty-input return value without loading the
        model."""
        return EMBED_MODEL_DIMS.get(self._model_name)

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        if not texts:
            # SentenceTransformer raises on an empty list; short-circuit. The
            # shape uses the declared dim for the configured model so callers
            # that swap models don't get a silent (0, 384) misfit.
            dim = self.dim
            if dim is None:
                # Unknown model: load it so we can read the real dim. Better
                # to pay the import cost once than to corrupt downstream shape
                # checks.
                from sentence_transformers import SentenceTransformer
                self._model = self._model or SentenceTransformer(self._model_name)
                dim = int(self._model.get_sentence_embedding_dimension())
            return np.zeros((0, dim), dtype=np.float32)
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
        vecs = self._model.encode(
            list(texts),
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vecs.astype(np.float32)
