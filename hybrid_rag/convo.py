"""Ambient conversational memory (opt-in).

When `HERMES_RAG_AMBIENT_CONVO_MEMORY=1`, the ambient retrieval path mixes
the current user turn's query embedding with the embeddings of the previous
1–2 user turns. This helps with follow-ups ("explain more about that")
where the chunk text in the corpus matches the prior topic rather than the
literal current message.

Trade-off: when the user changes topic, the prior embeddings contaminate
retrieval. That's why it is off by default.

Ring buffer is keyed by session id and lives only in memory — it never
persists across process restarts.
"""
from __future__ import annotations

import threading
from collections import OrderedDict

import numpy as np

from .config import env_flag

# Weights apply to current/previous/older user turn embeddings, normalized
# before mixing. Owned here because this module is the only consumer.
AMBIENT_CONVO_MEMORY_WEIGHTS = (1.0, 0.25, 0.1)

# Per-session ring buffers are kept in memory across LLM turns. We cap the
# number of distinct sessions tracked so a long-running Hermes process
# can't leak unboundedly as users churn through session ids. Each entry
# holds ~3 × dim × 4 bytes; at the default 1024-dim embedder that's ~12
# KiB per session, so 4096 sessions cap RSS contribution at ~48 MiB.
_MAX_RINGS = 4096

_LOCK = threading.Lock()
_RINGS: "OrderedDict[str, list[np.ndarray]]" = OrderedDict()
_RING_SIZE = len(AMBIENT_CONVO_MEMORY_WEIGHTS)  # current + N priors


def is_enabled() -> bool:
    return env_flag("HERMES_RAG_AMBIENT_CONVO_MEMORY")


def push(session_id: str, vec: np.ndarray) -> None:
    """Insert `vec` as the newest entry for `session_id`. Older entries
    drop off the tail when the buffer reaches RING_SIZE. Touching a
    session refreshes its LRU position; once the total number of tracked
    sessions exceeds _MAX_RINGS, the least-recently-used session's ring
    is evicted."""
    if not session_id:
        return
    with _LOCK:
        ring = _RINGS.get(session_id)
        if ring is None:
            ring = []
            _RINGS[session_id] = ring
        else:
            _RINGS.move_to_end(session_id)
        ring.insert(0, np.asarray(vec, dtype=np.float32))
        if len(ring) > _RING_SIZE:
            del ring[_RING_SIZE:]
        while len(_RINGS) > _MAX_RINGS:
            _RINGS.popitem(last=False)


def get_ring(session_id: str) -> list[np.ndarray]:
    """Snapshot copy of the ring buffer for `session_id`, newest first.
    Reading also refreshes the LRU position so an actively-read session
    isn't evicted out from under a pending push."""
    if not session_id:
        return []
    with _LOCK:
        ring = _RINGS.get(session_id)
        if ring is None:
            return []
        _RINGS.move_to_end(session_id)
        return list(ring)


def reset_for_tests() -> None:
    with _LOCK:
        _RINGS.clear()


def mix_with_history(
    current: np.ndarray,
    history: list[np.ndarray],
    weights: tuple[float, ...] = AMBIENT_CONVO_MEMORY_WEIGHTS,
) -> np.ndarray:
    """Linearly combine `current` with up to `len(weights)-1` history vectors
    (newest first), then L2-normalize. Weights are normalized to sum to 1.

    If `history` is empty, returns `current` unchanged (already normalized
    by the embedder)."""
    if not history:
        return current
    vecs = [current] + list(history)[: len(weights) - 1]
    ws = list(weights[: len(vecs)])
    total = float(sum(ws))
    if total <= 0:
        return current
    ws = [w / total for w in ws]
    out = np.zeros_like(current, dtype=np.float32)
    for v, w in zip(vecs, ws):
        # Defensive: skip any malformed entry (e.g. dim drift across an
        # in-process reindex) rather than crashing the ambient hook.
        if v.shape == current.shape:
            out += np.asarray(v, dtype=np.float32) * w
    norm = float(np.linalg.norm(out))
    if norm == 0.0:
        return current
    return (out / norm).astype(np.float32)
