"""Process-wide RAG engine. Holds the (lazily loaded) BM25, embeddings array,
chunk_id list, embedder, and store. `reset()` drops cached state so a re-index
flushes the next query.

BM25 is rebuilt from SQLite at load time rather than read from a pickle —
see `storage.py` module docstring for the security reasoning.
"""
from __future__ import annotations

import logging
import threading

from .config import npz_path
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

    # --- public read-only views over the loaded state ---
    #
    # Sibling modules (`retrieval`, `hooks`) read these to score queries.
    # Properties (vs raw attribute access) declare the access boundary and
    # let us swap the backing store later (e.g. memmap) without touching
    # every caller.

    @property
    def store(self) -> Store:
        return self._store

    @property
    def embedder(self):
        return self._embedder

    @property
    def embeddings(self):
        return self._embeddings

    @property
    def chunk_ids(self) -> list[int]:
        return self._chunk_ids

    @property
    def bm25(self):
        return self._bm25

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

            if npz_p.exists():
                self._embeddings = self._store.load_embeddings(npz_p)
                self._chunk_ids = self._store.get_chunk_ids_ordered()
                self._bm25 = self._build_bm25()
            else:
                self._embeddings = None
                self._chunk_ids = []
                self._bm25 = None

            self._check_consistency()
            self._loaded = True

    def _build_bm25(self):
        """Build BM25Okapi from the SQLite chunks table. Returns None for an
        empty corpus. Identical tokenizer to query time — `retrieval._tokenize`
        is the single source so index- and query-side tokens stay aligned.
        """
        from rank_bm25 import BM25Okapi

        from .retrieval import _tokenize

        tokenized = [_tokenize(t) for t in self._store.iter_bm25_texts_ordered()]
        if not tokenized:
            return None
        return BM25Okapi(tokenized)

    def _invalidate_and_raise(self, message: str) -> None:
        """Scrub the loaded state and raise ``EngineLoadError``. Centralises
        the three-line reset that every consistency check would otherwise
        copy-paste."""
        self._embeddings = None
        self._chunk_ids = []
        self._bm25 = None
        raise EngineLoadError(message)

    def _check_consistency(self) -> None:
        """Refuse to serve queries if the loaded artifacts disagree about
        cardinality or model identity. A partial rebuild can leave .npz and
        the SQLite chunks table in inconsistent states; catching that here
        is much better than letting it surface as an IndexError deep in
        retrieval.
        """
        if self._embeddings is None:
            return
        self._check_cardinality()
        self._check_disk_dim()
        self._check_configured_dim()
        self._check_model_drift()

    def _check_cardinality(self) -> None:
        n_emb = int(self._embeddings.shape[0])
        n_ids = len(self._chunk_ids)
        if n_emb != n_ids:
            self._invalidate_and_raise(
                f"embeddings array has {n_emb} rows but SQLite has "
                f"{n_ids} chunks — re-run `hermes rag index <path> --force`."
            )

    def _check_disk_dim(self) -> None:
        """Catch the silent corruption case where the .npz was built with a
        different model than the currently configured one."""
        on_disk_dim = self._store.get_meta("embed_dim")
        if on_disk_dim is None:
            return
        try:
            disk_dim = int(on_disk_dim)
        except ValueError:
            return
        live_dim = int(self._embeddings.shape[1])
        if disk_dim != live_dim:
            self._invalidate_and_raise(
                f"embeddings.npz dim {live_dim} disagrees with stored "
                f"meta dim {disk_dim} — re-run "
                "`hermes rag index <path> --force`."
            )

    def _check_configured_dim(self) -> None:
        configured_dim = getattr(self._embedder, "dim", None)
        if not configured_dim:
            return
        live_dim = int(self._embeddings.shape[1])
        if configured_dim != live_dim:
            self._invalidate_and_raise(
                f"configured embedder dim {configured_dim} disagrees with "
                f".npz dim {live_dim} — re-run "
                "`hermes rag index <path> --force` "
                "(or unset HERMES_RAG_EMBED_MODEL / HERMES_RAG_EMBED_DIM)."
            )

    def _check_model_drift(self) -> None:
        """Dim matches but the model name doesn't — quality may degrade.
        Loud, but non-fatal: we still serve queries."""
        on_disk_model = self._store.get_meta("embed_model")
        configured_model = getattr(self._embedder, "model_name", None) or getattr(
            self._embedder, "_model_name", None
        )
        if on_disk_model and configured_model and on_disk_model != configured_model:
            log.warning(
                "embedding-model drift: index was built with %r but the "
                "current configuration is %r. Dimensions match so retrieval "
                "will still run, but quality may degrade until a "
                "`hermes rag index --force` rebuilds the .npz.",
                on_disk_model, configured_model,
            )

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


def set_engine_for_tests(engine: RAGEngine) -> None:
    """Test-only helper. Replaces the process singleton with ``engine``.
    Use :func:`reset_for_tests` to clear instead."""
    global _INSTANCE
    _INSTANCE = engine


def reset_for_tests() -> None:
    """Test-only helper. Clears the process singleton."""
    global _INSTANCE
    _INSTANCE = None
