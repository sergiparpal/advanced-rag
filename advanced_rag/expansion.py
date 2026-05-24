"""LLM-based query expansion. Returns the original query plus paraphrases and
a HyDE document. Always returns at least [q]; never raises out to the caller.
"""
from __future__ import annotations

import json
import logging
import os
import re

from .config import ANTHROPIC_MODEL

log = logging.getLogger(__name__)

_PROMPT = """You are helping a retrieval system find relevant documents.

Original query: {q}

Output a single JSON object (no surrounding text, no code fences) with two keys:
  - "paraphrases": list of 3 distinct rewrites of the query, each preserving the
    user's intent but using different vocabulary.
  - "hyde": one short hypothetical answer paragraph (1-3 sentences) that, if it
    existed in the corpus, would likely be retrieved for this query.

Return only the JSON. Do not explain.
"""

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _strip_fences(text: str) -> str:
    m = _FENCE_RE.search(text)
    return m.group(1).strip() if m else text.strip()


# Module-level Anthropic client cache. Reusing a single client across calls
# keeps the underlying httpx connection pool warm and lets the SDK's
# prompt-caching layer share state — both noticeably improve cache-hit rate
# and remove a few ms of per-call setup. The client is lazy because the
# `anthropic` package is an optional dep and may not be installed.
_ANTHROPIC_CLIENT = None


def _get_anthropic_client():
    """Return a cached `anthropic.Anthropic` client, or None if the SDK is not
    installed. Importing inside the function keeps the optional dependency
    truly optional."""
    global _ANTHROPIC_CLIENT
    if _ANTHROPIC_CLIENT is not None:
        return _ANTHROPIC_CLIENT
    try:
        import anthropic
    except ImportError:
        return None
    _ANTHROPIC_CLIENT = anthropic.Anthropic()
    return _ANTHROPIC_CLIENT


def _reset_anthropic_client_for_tests() -> None:
    """Test helper. Forces the next `expand_query` call to re-import and
    re-create the client — needed because `mock_anthropic` swaps `sys.modules`
    per test."""
    global _ANTHROPIC_CLIENT
    _ANTHROPIC_CLIENT = None


def expand_query(q: str) -> list[str]:
    """Return [q] (fallback) or [q, p1, p2, p3, hyde] when expansion succeeds.

    Paraphrases are deduplicated against the original query AND against each
    other (case- and whitespace-insensitive), so a model that returns
    ``["foo", "FOO", "bar"]`` does not waste a hybrid_search round on the
    duplicate.

    Failure modes that fall back silently to [q]:
      - `import anthropic` fails (package not installed)
      - ANTHROPIC_API_KEY env var unset
      - Any exception raised by the SDK call
      - Response missing the expected JSON structure
    """
    if not q or not q.strip():
        return [q] if q else []
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return [q]
    client = _get_anthropic_client()
    if client is None:
        return [q]

    try:
        msg = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=512,
            messages=[{"role": "user", "content": _PROMPT.format(q=q)}],
        )
        text = "".join(getattr(part, "text", "") for part in msg.content)
        payload = json.loads(_strip_fences(text))
        paraphrases = payload.get("paraphrases", []) or []
        hyde = payload.get("hyde", "") or ""
        out = [q]
        # Dedupe FIRST (canonical = stripped + lowercased), THEN cap at 3 — so
        # a model that returns ["foo", "foo", "bar", "baz"] still contributes
        # three useful paraphrases instead of being undermined by duplicates.
        seen = {q.strip().lower()}
        kept = 0
        for p in paraphrases:
            if kept >= 3:
                break
            if not isinstance(p, str):
                continue
            stripped = p.strip()
            if not stripped:
                continue
            key = stripped.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(stripped)
            kept += 1
        # Dedupe HyDE against the original query and the surviving paraphrases
        # too — otherwise a model that echoes the query into the `hyde` field
        # wastes a hybrid_search variant on an exact duplicate.
        if isinstance(hyde, str) and hyde.strip():
            stripped = hyde.strip()
            if stripped.lower() not in seen:
                out.append(stripped)
        return out or [q]
    except Exception as e:
        log.warning("query expansion failed: %s", e)
        return [q]
