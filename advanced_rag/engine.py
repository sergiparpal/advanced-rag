"""Process-wide RAG engine. Holds the (lazily loaded) BM25, embeddings array,
chunk_id list, embedder, and store. `reset()` drops cached state so a re-index
flushes the next query.
"""
from __future__ import annotations

import logging
import threading

from .config import bm25_path, npz_path
from .storage import Store

log = logging.getLogger(__name__)

_INSTANCE = None
_INSTANCE_LOCK = threading.Lock()


class EngineLoadError(RuntimeError):
    """Raised when the on-disk index artifacts are inconsistent. Surfaces a
    partial-failure scenario (e.g. .npz updated but bm25.pkl stale, or
    embed_row drift) instead of letting it manifest as a silent IndexError
    deep inside retrieval.
    """


class RAGEngine:
    def __init__(self, store: Store | None = None, embedder=None):
        self._store = store or Store()
        self._embedder = embedder
        self._bm25 = None
        self._embeddings = None
        self._chunk_ids: list[int] = []
        self._loaded = False
        self._lock = threading.Lock()

    @property
    def store(self) -> Store:
        return self._store

    def _make_default_embedder(self):
        from .embeddings import Embedder
        return Embedder()

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            if self._embedder is None:
                self._embedder = self._make_default_embedder()

            npz_p = npz_path(self._store.data_dir)
            bm25_p = bm25_path(self._store.data_dir)

            if npz_p.exists():
                self._embeddings = self._store.load_embeddings(npz_p)
                # chunk_ids is now derived from SQLite (canonical order) rather
                # than carried in the .npz — embed_row in the DB is the source
                # of truth for the row-index ↔ chunk-id mapping.
                self._chunk_ids = [c.id for c in self._store.iter_chunks_ordered()]
            else:
                self._embeddings = None
                self._chunk_ids = []

            if bm25_p.exists():
                self._bm25 = self._store.load_bm25(bm25_p)
            else:
                self._bm25 = None

            self._check_consistency()
            self._loaded = True

    def _check_consistency(self) -> None:
        """Refuse to serve queries if the loaded artifacts disagree about
        cardinality — a partial rebuild can leave .npz, bm25.pkl, and the
        SQLite chunks table in inconsistent states."""
        if self._embeddings is None:
            # No artifacts loaded; nothing to check.
            return
        n_emb = int(self._embeddings.shape[0])
        n_ids = len(self._chunk_ids)
        if n_emb != n_ids:
            self._embeddings = None
            self._chunk_ids = []
            self._bm25 = None
            raise EngineLoadError(
                f"embeddings array has {n_emb} rows but SQLite has "
                f"{n_ids} chunks — re-run `hermes rag index <path> --force`."
            )
        if self._bm25 is not None:
            bm25_n = self._bm25_doc_count()
            if bm25_n is not None and bm25_n != n_emb:
                self._embeddings = None
                self._chunk_ids = []
                self._bm25 = None
                raise EngineLoadError(
                    f"BM25 was built for {bm25_n} docs but embeddings has "
                    f"{n_emb} — re-run `hermes rag index <path> --force`."
                )

    def _bm25_doc_count(self) -> int | None:
        """Best-effort document count for the loaded BM25 instance.
        rank_bm25's BM25Okapi keeps `corpus_size`; older or stub objects may
        not, so we degrade silently rather than refuse to load."""
        for attr in ("corpus_size",):
            n = getattr(self._bm25, attr, None)
            if isinstance(n, int):
                return n
        doc_freqs = getattr(self._bm25, "doc_freqs", None)
        if isinstance(doc_freqs, list):
            return len(doc_freqs)
        return None

    def reset(self) -> None:
        with self._lock:
            self._bm25 = None
            self._embeddings = None
            self._chunk_ids = []
            self._loaded = False


def get_engine() -> RAGEngine:
    global _INSTANCE
    if _INSTANCE is None:
        with _INSTANCE_LOCK:
            if _INSTANCE is None:
                _INSTANCE = RAGEngine()
    return _INSTANCE


def set_engine_for_tests(engine: RAGEngine | None) -> None:
    """Test-only helper. Replaces the singleton (or clears it with None)."""
    global _INSTANCE
    _INSTANCE = engine
