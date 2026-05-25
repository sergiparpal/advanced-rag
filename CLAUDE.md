# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Sibling docs

- `@README.md` — user-facing overview, architecture diagram, and the authoritative pipeline shapes (ambient + explicit). The mermaid in there is the source of truth for "what runs when".
- `REQUIREMENTS.md` — the spec: module responsibilities, DDL, invariants, acceptance criteria.
- `HERMES_API.md` — verified Hermes plugin signatures. Check here first when the adapter layer misbehaves.

## Dev machine ≠ runtime machine

This separation drives most design choices:

- **Light deps only on dev.** `numpy`, `rank_bm25`, `pyyaml`, `pytest`. Do **not** install `sentence-transformers`, `anthropic`, `cohere`, `pypdf` here — `tests/conftest.py` stubs them via `sys.modules` patching (`mock_anthropic`, `mock_cohere`, `mock_cross_encoder`, `StubEmbedder`).
- **Runtime state never appears in the repo.** `data/` (SQLite, `.npz`, BM25 sidecar) is gitignored and created lazily on the runtime machine. Tests must route through `tmp_data_dir` (sets `HERMES_RAG_DATA_DIR=tmp_path`) or pass `data_dir=tmp_path` to `Store`.
- **No real Hermes integration test on dev.** The adapter layer is verified manually post-deploy; dev verifies pure logic only.

## Architecture: pure core + thin Hermes adapter

The codebase splits into two layers. **Pure modules** import no Hermes and are unit-tested directly — that's everything in `hybrid_rag/` *except* `__init__.py` and `adapters.py`.

The **Hermes-coupled surface** is exactly those two files:
- `hybrid_rag/__init__.py::register(ctx)` — wires everything into `ctx.register_*`.
- `hybrid_rag/adapters.py` — closures reshaping pure handlers to whatever signature Hermes wants; lazy imports inside each closure so a missing pure module fails loud during dev.

When Hermes signatures shift, the fix lives in those two files. Don't push Hermes shapes (e.g. `**kwargs`, `dict | None` returns) into the pure modules.

## Invariants you must not break

- **Retrieval target is always a parent, never a chunk.** Chunks are the search space; parents are what the agent receives.
- **`embed_row` invariant.** Chunk row N in canonical SQLite ordering (`SELECT … ORDER BY parent_id, ord`) ↔ row N of `embeddings.npz`. `Store.bulk_update_embed_rows` writes row indices back after a rebuild. Indexing rebuilds the whole `.npz` and `bm25_state.json` from this canonical ordering and renames atomically (`.tmp` → final).
- **Identical tokenizer at index time and query time.** `retrieval._tokenize` is the single source — `indexing._build_bm25_state` and `retrieval.hybrid_search` both go through it.
- **Parent rollup uses MAX of children's RRF scores**, not SUM/MEAN — avoids penalizing parents whose other children are unrelated.
- **Second-level RRF fuses chunk rankings**, not parent rankings — fusion benefits from all matched evidence; the parent rollup happens once afterward.
- **Hooks must never raise.** `hooks.ambient_pre_llm_call` wraps the entire body in `try/except Exception: return None`. `state.is_ambient_enabled()` fails open (errors → True). Tools return JSON-encoded errors, never raise.
- **Data-dir precedence (`config.py`).** Explicit `Store(data_dir=…)` arg > `HERMES_RAG_DATA_DIR` env > default `~/.hermes/plugins/hybrid-rag/data/`. Don't add a fourth path.

## Optional dependency degradation

`COHERE_API_KEY`, `ANTHROPIC_API_KEY`, and `pypdf` are all optional; the plugin must never block on a missing one. Fallbacks live in `rerank.py`, `expansion.py`, and `indexing.py` respectively, and tests cover each path with mocked modules — keep that coverage.

## Common commands

```bash
python -c "from hybrid_rag import register"                              # smoke-imports the plugin
python -c "import yaml; yaml.safe_load(open('hybrid_rag/plugin.yaml'))"  # validates manifest
```

Runtime-only commands (do **not** run on dev — they would pollute `~/.hermes/...`):
```bash
hermes rag index <path> [--force]
hermes rag stats
hermes rag clear
# In a Hermes session: /rag, /rag on, /rag off, /rag stats
```

## Conventions

- Tests for new pure modules go in `tests/test_<module>.py` and must run without heavy deps — use the existing stub/mock fixtures in `conftest.py`.
- `hybrid_rag/requirements.txt` is a byte-identical copy of the repo-root `requirements.txt` so a single `rsync -av hybrid_rag/ …` carries deps to runtime. If you change one, change both.
- Slash handler signature is `(raw_args: str) -> str | None` with **no kwargs** (`HERMES_API.md` §4) — per-session toggle is impossible in v0.1, only a process-global `_default` key.
- `register_skill` requires `pathlib.Path`, not `str` (`HERMES_API.md` §5) — e.g. `Path(__file__).parent / "skills" / "rag-usage" / "SKILL.md"`.
