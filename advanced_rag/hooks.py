"""Ambient pre-LLM-call hook. Injects up to AMBIENT_TOP_PARENTS parents into
the prompt when the user message looks substantive and the top result clears
the threshold. Must never raise — return None on any failure path.

Ambient pipeline:
    hybrid → top-30 chunks → MAX rollup → top-10 parents
        → local cross-encoder rerank → top-3 parents
        → 0.25 threshold (post-rerank) → 1500-token cap

The local-only rerank is intentional: Cohere's API latency would defeat the
purpose of a cheap per-turn injection layer. The explicit `rag_search` path
keeps using Cohere when available.
"""
from __future__ import annotations

import logging
import threading

from . import convo, rerank, retrieval, state
from .config import (
    AMBIENT_RERANK_POOL,
    AMBIENT_SCORE_THRESHOLD,
    AMBIENT_TOKEN_CAP,
    AMBIENT_TOP_PARENTS,
)
from .engine import get_engine

log = logging.getLogger(__name__)

_MIN_MESSAGE_LEN = 8

# Track Hermes hook kwargs we don't consume — surface them once so that an
# upstream signature change adding a useful field doesn't go unnoticed.
_HOOK_KNOWN_KWARGS = frozenset({
    "session_id", "user_message", "conversation_history", "model", "platform",
})
_HOOK_SEEN_EXTRA_KWARGS: set[str] = set()
_HOOK_KWARG_LOG_LOCK = threading.Lock()


def _log_unfamiliar_kwargs(kwargs: dict) -> None:
    """One-shot debug log per never-before-seen kwarg name. Helps spot a
    Hermes upgrade that started passing something the hook should be
    reading. Cheap — we only log first occurrence."""
    extras = set(kwargs) - _HOOK_KNOWN_KWARGS - _HOOK_SEEN_EXTRA_KWARGS
    if not extras:
        return
    with _HOOK_KWARG_LOG_LOCK:
        new = extras - _HOOK_SEEN_EXTRA_KWARGS
        if not new:
            return
        _HOOK_SEEN_EXTRA_KWARGS.update(new)
        log.debug("ambient_pre_llm_call: ignoring new kwargs %s", sorted(new))


def ambient_pre_llm_call(
    *,
    session_id: str | None = None,
    user_message: str = "",
    conversation_history=None,
    model: str | None = None,
    platform: str | None = None,
    **kwargs,
):
    """Return `{"context": str}` to inject ambient context, or `None` to do
    nothing. Never raises.

    Hermes passes additional kwargs that this hook doesn't use today
    (`is_first_turn`, `sender_id`, etc.); they're absorbed by ``**kwargs`` so
    upstream signature drift never breaks the wire. First-seen extras are
    logged once at debug so an upgrade that starts passing something
    *useful* doesn't go unnoticed.
    """
    try:
        if kwargs:
            _log_unfamiliar_kwargs(kwargs)
        if not state.is_ambient_enabled(session_id):
            return None
        if not user_message or len(user_message.strip()) < _MIN_MESSAGE_LEN:
            return None

        engine = get_engine()
        engine._ensure_loaded()
        if engine.embeddings is None or engine.embeddings.shape[0] == 0:
            return None

        hits = _ambient_hybrid_search(engine, user_message, session_id)
        if not hits:
            return None
        parents = retrieval.chunks_to_parents(
            engine, hits, top=AMBIENT_RERANK_POOL,
        )
        if not parents:
            return None

        # Local-only rerank — never Cohere on the per-turn path.
        parents = rerank.rerank_local(
            user_message, parents, top_k=AMBIENT_TOP_PARENTS,
        )
        if not parents:
            return None

        # Threshold applies to the post-rerank score. Identity fallback (no
        # cross-encoder) keeps `rerank_score=None`, so `effective_score`
        # falls back to RRF. Note: RRF scores are tiny (~0.03-0.06), so a
        # 0.25 threshold effectively gates ambient OFF when the cross-encoder
        # is unavailable. That's intentional — without a reranker we don't
        # have enough confidence to inject silently.
        if parents[0].effective_score < AMBIENT_SCORE_THRESHOLD:
            return None

        context = retrieval.format_context(parents, token_cap=AMBIENT_TOKEN_CAP)
        if not context:
            return None
        return {"context": context}
    except Exception as e:
        log.warning("ambient_pre_llm_call failed: %s", e)
        return None


def _ambient_hybrid_search(engine, user_message: str, session_id: str | None):
    """Hybrid search for the ambient path. When convo memory is enabled,
    mix the current query embedding with the previous turns' embeddings
    before dense scoring; BM25 always operates on the literal current
    message so lexical search isn't contaminated."""
    if convo.is_enabled() and session_id:
        # Compute the query embedding once, mix with history, push into ring.
        qvec_batch = engine.embedder.encode([user_message])
        if qvec_batch.shape[0] == 0:
            return retrieval.hybrid_search(engine, user_message, k_pool=30)
        cur = qvec_batch[0]
        ring = convo.get_ring(session_id)
        mixed = convo.mix_with_history(cur, ring)
        convo.push(session_id, cur)
        return retrieval.hybrid_search_with_vec(
            engine, user_message, mixed, k_pool=30,
        )
    return retrieval.hybrid_search(engine, user_message, k_pool=30)
