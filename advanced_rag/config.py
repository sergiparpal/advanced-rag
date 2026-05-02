"""Configuration constants and data-dir resolution.

The single rule: explicit `Store(data_dir=...)` arg > `HERMES_RAG_DATA_DIR`
env var > default `~/.hermes/plugins/advanced-rag/data/`.
"""
from __future__ import annotations

import os
from pathlib import Path

DEFAULT_DATA_DIR = Path.home() / ".hermes" / "plugins" / "advanced-rag" / "data"


def get_data_dir() -> Path:
    env = os.environ.get("HERMES_RAG_DATA_DIR")
    return Path(env) if env else DEFAULT_DATA_DIR


def db_path(data_dir: Path | None = None) -> Path:
    return (data_dir or get_data_dir()) / "rag.sqlite"


def npz_path(data_dir: Path | None = None) -> Path:
    return (data_dir or get_data_dir()) / "embeddings.npz"


def bm25_path(data_dir: Path | None = None) -> Path:
    return (data_dir or get_data_dir()) / "bm25.pkl"


def toggles_path(data_dir: Path | None = None) -> Path:
    return (data_dir or get_data_dir()) / "toggles.json"


# Tunables
MAX_CHUNK = 300
CHUNK_OVERLAP = 50
MAX_PARENT_CHARS = 8000
# Markdown text before the first `##` heading is captured as a synthetic
# "preamble" parent only when its body length (after stripping any leading
# `# H1` line) clears this threshold. Below it the prefix is dropped on the
# assumption that it's boilerplate (a stray title line, frontmatter, etc.).
PREAMBLE_MIN_CHARS = 200
RRF_K = 60
AMBIENT_TOP_PARENTS = 3
AMBIENT_TOKEN_CAP = 1500
AMBIENT_SCORE_THRESHOLD = 0.25
EMBED_MODEL = "all-MiniLM-L6-v2"
# Embedding dimensionality keyed on the model name. Used to pre-allocate the
# `(0, dim)` empty-result array in `Embedder.encode([])` without loading the
# model. Add an entry here when you swap the model — otherwise the empty-input
# fallback would silently use the wrong shape and break downstream consumers.
EMBED_MODEL_DIMS: dict[str, int] = {
    "all-MiniLM-L6-v2": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-small-en-v1.5": 384,
}
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
COHERE_RERANK_MODEL = "rerank-english-v3.0"
