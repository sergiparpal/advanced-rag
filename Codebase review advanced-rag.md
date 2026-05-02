Codebase review — advanced-rag                           
                                                       
  Read every source and test file, ran the full suite (113/113 passing), and verified the more subtle findings empirically. The codebase is well-structured: pure-core/adapter split is real, 
  invariants are documented, hooks/tools wrap exceptions, atomic writes are correct. Below are the issues that are worth your attention, ordered by severity.
                                                                                                                                                                                              
  ---                                                        
  High severity                                                                                                                                                                               
                                                                                                                                                                                              
  H1. extract_md silently drops everything before the first ##  heading                                                                                                                       
                                                                                                                                                                                              
  advanced_rag/parents.py:46-58 — starts = [m.start() for m in matches] skips text from 0 to starts[0]. Verified: a doc with # Title + 2-paragraph intro + two ##  sections produces only the 
  two sections; the intro is discarded.                                                                                                                                                       
                                                                                                                                                                                              
  The fixture tests/fixtures/docs/alpha.md actually documents this ("It should not become a parent on its own"), so it's intentional, but it's a footgun for real users (TL;DRs, abstracts, # 
  H1 body content all vanish). At minimum, surface it in README.md troubleshooting and the rag-usage skill. Better: emit a synthetic "preamble" parent for the prefix when its len > 
  some_threshold.                                                                                                                                                                             
                                                                                                                                                                                              
  H2. cli rag clear --yes will rmtree whatever HERMES_RAG_DATA_DIR points to                                                                                                                  
  
  advanced_rag/cli.py:46-58 — no sanity guard on data_dir. If a user (or a hook/init script) sets HERMES_RAG_DATA_DIR=/, --yes deletes their root with no prompt.                             
                  
  Add a guard: refuse to rmtree if data_dir is /, ~, doesn't end in something looking like a data dir, or isn't under ~/.hermes/ and HERMES_RAG_DATA_DIR was set. Even a basic if data_dir == 
  Path.home() or str(data_dir) == "/": raise would prevent the worst.
                                                                                                                                                                                              
  H3. Empty Cohere result short-circuits the local-cross-encoder fallback                                                                                                                     
  
  advanced_rag/rerank.py:68-78 — if cohere_out is not None: return cohere_out. Verified: a Cohere response with results=[] returns [], and [] is not None is True, so we never try the local  
  reranker and the user gets zero results.
                                                                                                                                                                                              
  Fix: change the check from is not None to truthy: if cohere_out: return cohere_out. Or: distinguish "Cohere succeeded with N>0 results" from "Cohere returned empty" and fall through in the
   empty case.
                                                                                                                                                                                              
  ---             
  Medium severity
                 
  M1. Partial-failure recovery for indexing is "use --force"
                                                                                                                                                                                              
  advanced_rag/indexing.py:147-166 — _index_file writes file/parent/chunk rows in their own transactions, then rebuild_artifacts runs. If rebuild_artifacts raises (embedder OOM, BM25 build  
  fails, disk full mid-write), the SQLite has the new rows but embed_row=0 placeholder, the .npz/.pkl are stale (atomic-rename never committed), and engine.reset() is never reached.         
                                                                                                                                                                                              
  Subsequent runs see those files as "unchanged" via manifest_diff (mtime/size match) and skip them, leaving the index permanently inconsistent until --force. Two reasonable fixes: (1) on   
  every load, validate that len(_chunk_ids) == sqlite_chunk_count and force rebuild if mismatched; (2) wrap the whole index_path body so a rebuild_artifacts failure rolls back the inserted
  file rows (one outer transaction).                                                                                                                                                          
                  
  M2. rebuild_artifacts is two-step and can leave .npz and .pkl desynced                                                                                                                      
  
  advanced_rag/indexing.py:185-213 — save_embeddings is atomic individually, save_bm25 is atomic individually, but they're not atomic together. If save_bm25 fails after save_embeddings      
  succeeded, the engine loads the new dense vectors with the old BM25 — chunk ID space mismatch → engine._chunk_ids[i] indexing errors at query time (see M3).
                                                                                                                                                                                              
  Fix: write both .npz.tmp and .pkl.tmp, then os.replace both back-to-back (not truly atomic, but the window shrinks to microseconds). Or: do a sanity check at engine load                   
  (len(bm25.doc_freqs) == embeddings.shape[0]) and refuse to load on mismatch.
                                                                                                                                                                                              
  M3. Retrieval indexing has no defensive bounds check                                                                                                                                        
  
  advanced_rag/retrieval.py:58-86 — _bm25_topk and _dense_topk both do engine._chunk_ids[i] without confirming len(_chunk_ids) == len(scores) / embeddings.shape[0]. If M2 happens (e.g., the 
  BM25 was rebuilt but the .npz wasn't), this raises IndexError mid-query — caught by tools.tool_rag_search's outer try/except as a generic JSON error but with no useful diagnostic for the
  user.                                                                                                                                                                                       
                  
  Fix: at engine load, assert len(self._chunk_ids) == self._embeddings.shape[0]; if BM25 is non-None, additionally check len(self._bm25.doc_freqs) == self._embeddings.shape[0]. Refuse to    
  load and log a clear error.
                                                                                                                                                                                              
  M4. Recursive split's overlap pass produces chunks larger than max_size                                                                                                                     
  
  advanced_rag/chunking.py:83-92 — verified: with max_size=200, overlap=40, a real text produces sizes [198, 203, 41, 240, 101, 41, 238, 82]. Three chunks exceed max_size. The bound is      
  max_size + overlap.
                                                                                                                                                                                              
  This is consistent with the docstring intent (overlap adds context) but test_paragraph_split_respects_max_size only checks len(c) <= max_size when overlap=0, hiding it. Either:            
  - Document explicitly that effective max size is max_size + overlap and update the docstring, or
  - Tighten the merge: if len(merged) <= max_size (not max_size + overlap).                                                                                                                   
                                                                           
  The downstream MAX_PARENT_CHARS=8000 bound on parents already absorbs the slop, so this is mostly a documentation issue — but the spec in REQUIREMENTS.md §3.8 says "fixed-size hard split  
  with overlap" without acknowledging the inflation.                                                                                                                                          
                                                                                                                                                                                              
  M5. manifest_diff keyed only on (mtime, size) misses content-only changes                                                                                                                   
                  
  advanced_rag/storage.py:123-154. A user editing a file in-place (same byte count, file system that preserves mtime, e.g., touching with os.utime) would silently skip reindexing. The       
  content_hash column is computed and stored during reindex but never compared — the spec already mentions "Hashing happens only on miss/change" but doesn't justify why we keep the column at
   all.                                                                                                                                                                                       
                  
  Either start using content_hash as a tiebreaker when (mtime, size) matches but you want a stronger guarantee, or drop the column entirely (see L4 below).                                   
  
  M6. extract_md regex matches ##  inside fenced code blocks                                                                                                                                  
                  
  advanced_rag/parents.py:30 — _H2_RE = re.compile(r"^##\s+", re.MULTILINE). A markdown file like:                                                                                            
                  
  # Title                                                                                                                                                                                     
  ```python       
  ## comment in code
  ```                                                                                                                                                                                         
  
  …would split on the ## comment if it isn't indented (the indented case is fine because ^ requires position 0). Real-world markdown samples (Python, shell, MDX) trip this. Acceptable for   
  v0.1 but worth a code-fence-aware split in v0.2.
                                                                                                                                                                                              
  M7. pyproject.toml license syntax is deprecated                                                                                                                                             
  
  pyproject.toml:12 — license = { file = "LICENSE" }. PEP 639 finalized the SPDX expression form; setuptools ≥77 emits a deprecation warning, and the LICENSE here is GPL-3.0. Replace with:  
                  
  license = "GPL-3.0-or-later"                                                                                                                                                                
  license-files = ["LICENSE"]

  (Confirmed setuptools 80.10.2 in this env.)                                                                                                                                                 
  
  M8. requires_env in plugin.yaml lists optional vars as required                                                                                                                             
                  
  advanced_rag/plugin.yaml:14-17 lists COHERE_API_KEY, ANTHROPIC_API_KEY, HERMES_RAG_DATA_DIR under requires_env. The README.md and REQUIREMENTS.md §3.13 both make clear all three are       
  optional. Per HERMES_API.md §6 the field is informational only, but the name is misleading — hermes plugin list will show them as requirements.
                                                                                                                                                                                              
  If Hermes ever adds a check_fn flow, this becomes a real bug. For now, either rename the section or drop the optional ones.                                                                 
  
  ---                                                                                                                                                                                         
  Low severity    
              
  L1. embed_row column is set, indexed, and never read at query time
                                                                                                                                                                                              
  advanced_rag/storage.py:61-64 — embed_row INTEGER NOT NULL plus idx_chunks_embed_row. bulk_update_embed_rows writes it after every rebuild. Only test_indexing.py::test_embed_row_invariant 
  reads it. The actual hot path uses engine._chunk_ids (loaded from the .npz), so embed_row is dead. The index is also dead.                                                                  
                                                                                                                                                                                              
  Either drop the column + index + the bulk_update_embed_rows call (saves an UPDATE … WHERE id=? per chunk on every rebuild), or wire the query path to use it (allows skipping the chunk_ids 
  array in the .npz).
                                                                                                                                                                                              
  L2. content_hash column is computed but never read

  advanced_rag/indexing.py:33-41 (_hash_file SHA-256 over the whole file) and storage.py:38 — but no WHERE content_hash = ? anywhere. SHA-256 over every new/changed file is non-trivial      
  wasted I/O (esp. for PDFs). See M5 for the same column from the diff angle.
                                                                                                                                                                                              
  L3. index_path force=True does duplicate work                                                                                                                                               
  
  advanced_rag/indexing.py:116-142 — confirmed: in force mode, _existing_id_for is called once per file via Bash, even though manifest_diff already classified them. The [p for p, fid in     
  changed_now if fid is None] + diff["new"] line creates duplicate Path entries that are then deduplicated by a seen set. Also [p for _, p in [(fid, p) for p, fid in changed_now]] is [p for 
  p, _ in changed_now] after a no-op transformation.                                                                                                                                          
                  
  Suggested rewrite:

  if force:
      existing_ids = {Path(r["path"]): r["id"] for r in conn.execute(...)}
      changed_now = [(p, existing_ids[p]) for p in files if p in existing_ids]
      new_now = [p for p in files if p not in existing_ids]                                                                                                                                   
  else:                                                                                                                                                                                       
      changed_now = diff["changed"]                                                                                                                                                           
      new_now = diff["new"]                                                                                                                                                                   
  unchanged_now = [] if force else diff["unchanged"]
  deleted_ids = diff["deleted"]

  L4. print(...) in indexing.py mixes warnings into stdout                                                                                                                                    
  
  advanced_rag/indexing.py:152 — print(f"[advanced-rag] failed to index {p}: {e}") lands in stdout. The CLI then print(json.dumps(summary)) to the same stream. A consumer parsing JSON output
   will choke on the warning prefix. Use print(..., file=sys.stderr) or the existing log.warning.
                                                                                                                                                                                              
  L5. N+1 SQL queries in retrieval and rollup                                                                                                                                                 
  
  advanced_rag/retrieval.py:99-104 (loop of parent_id_for_chunk) and :118-132 (loop of get_parent). Each tool_rag_search call does ~30+10 = 40 queries; with five expansions, ~150 queries per
   search. For local SQLite this is microseconds, so not urgent — but a single SELECT id, parent_id FROM chunks WHERE id IN (?, ?, …) would be cleaner and avoids any future regression if the
   DB moves out of process.                                                                                                                                                                   
                  
  L6. Embedder.encode([]) returns hardcoded shape (0, 384)                                                                                                                                    
  
  advanced_rag/embeddings.py:22-24. The 384 is MiniLM's dim. If EMBED_MODEL is changed to a different model (e.g., bge-base-en-v1.5 at 768), shape mismatches downstream. Currently no caller 
  depends on the dim of an empty result, so harmless — but a self-documenting MODEL_DIMS = {EMBED_MODEL: 384, ...} lookup would make the assumption explicit.
                                                                                                                                                                                              
  L7. rerank mutates input ParentResult objects                                                                                                                                               
  
  advanced_rag/rerank.py:39-42, 60-61 — p.rerank_score = ... writes back into the caller's parent. Tests don't share state across calls, so they pass, but a user calling rerank(q, parents,  
  k) twice (e.g., A/B testing models) will see surprising state on the second call. If you want this to stay mutable, add a # mutates note to the docstring; if not, return new ParentResult
  instances.                                                                                                                                                                                  
                  
  L8. expand_query doesn't dedupe paraphrases against each other                                                                                                                              
  
  advanced_rag/expansion.py:65-71 — only checks p.strip() != q.strip(). If the model returns ["foo", "foo", "bar"], you get [q, foo, foo, bar, hyde] and the duplicate paraphrase wastes a    
  hybrid_search round.
                                                                                                                                                                                              
  L9. expand_query rebuilds the Anthropic client on every call                                                                                                                                
  
  advanced_rag/expansion.py:55 — client = anthropic.Anthropic() per call. The SDK is happy to reuse clients (and prompt caching prefers it). Module-level singleton would shave a few ms per  
  query and align with prompt-caching best practices for higher cache hit rates.
                                                                                                                                                                                              
  L10. Adapter's make_session_warm_hook background thread can run before register returns                                                                                                     
  
  advanced_rag/adapters.py:64-81 — fires immediately on on_session_start, spawns a daemon thread, returns. If the import of engine fails inside the thread the swallow is silent. Tests cover 
  the swallow but the user has no signal that warming failed (cold load on first ambient call is the fallback as designed). Consider a log.debug so users can opt in to seeing it.
                                                                                                                                                                                              
  L11. state.py cache is module-global without thread-synchronization                                                                                                                         
  
  advanced_rag/state.py:14-17, 20-39, 42-57. Two threads racing on _load/_store can leave stale _CACHE. Acceptable because (a) toggle is a fail-open hint and (b) TTL is 1 s, but worth a     
  one-line comment so a future refactor doesn't add a critical assumption on top.
                                                                                                                                                                                              
  L12. advanced_rag/requirements.txt duplicates root requirements.txt                                                                                                                         
  
  This is documented (CLAUDE.md "deliberate copy"). Risk: they will drift. Consider a Makefile target / pre-commit hook (diff -q advanced_rag/requirements.txt requirements.txt) so the       
  duplication is enforced.
                                                                                                                                                                                              
  L13. parents.extract_pdf uses a stale global read                                                                                                                                           
  
  advanced_rag/parents.py:92 — global PdfReader is declared but only read, not assigned. Python only needs global for assignment. Cosmetic, but it suggests a bug to readers.                 
                  
  ---                                                                                                                                                                                         
  Test coverage gaps worth filling
                                                                                                                                                                                              
  - Engine consistency at load: no test asserts len(_chunk_ids) == _embeddings.shape[0] after a partial-failure scenario. Add one that builds an inconsistent pair on disk and confirms the
  engine refuses to load (or recovers).                                                                                                                                                       
  - extract_md preamble loss: the behavior is correct-by-design but should have an explicit test showing what happens, so future refactors don't accidentally "fix" it (or are forced to
  acknowledge the change).                                                                                                                                                                    
  - Fenced-code-block false positive in extract_md: add a fixture with ##  inside a fenced block and pin the current behavior (M6).
  - rerank empty Cohere response → fallback: covers H3.                                                                                                                                       
  - rmtree safety: confirm that clear --yes with HERMES_RAG_DATA_DIR=/ (or other dangerous paths) refuses (after fix).                                                                        
  - Negative / zero k in rag_search: pin behavior.                                                                                                                                            
  - recursive_split chunk-size invariant with overlap > 0: explicitly check len(c) <= max_size + overlap so the relaxed bound is documented in code.                                          
                                                                                                                                                                                              
  ---                                                                                                                                                                                         
  Dependency / packaging notes                                                                                                                                                                
                                                                                                                                                                                              
  - pyproject.toml: license syntax (M7).
  - pypdf >= 4.0: current is 5.x with API breaks in some helpers; extract_text() still works. Consider pinning to >=4.0,<6 until tested against 5.x.                                          
  - cohere >= 5.0: current is 5.18+. ClientV2 is now stable; the runtime getattr(cohere, "ClientV2", None) or cohere.Client still works, but you can simplify to cohere.ClientV2(...).        
  - anthropic >= 0.40: as of Jan 2026 the SDK is on 0.79+. The client.messages.create shape used here is stable across the range. Consider pinning to a tested version anyway (>=0.40,<1.0).  
  - sentence-transformers >= 3.0: 3.4+ is current; no known breaks.                                                                                                                           
  - Dev install instruction in README.md is consistent with CLAUDE.md.                                                                                                                        
                  
  ---
  What's done well (worth preserving)

  - Pure-core/adapter split is real and the adapter layer is genuinely thin (~80 LOC).
  - Atomic writes (.tmp → os.replace) are correct in Store.save_embeddings / save_bm25 / state._store.
  - Hooks and tools both wrap with try/except; is_ambient_enabled fails open. test_hook.test_never_raises_when_engine_misbehaves pins the contract.                                           
  - Test coverage routes through tmp_data_dir — the HERMES_RAG_DATA_DIR precedence isolates real user data from the test suite. The set_engine_for_tests(None) discipline is consistent.      
  - HERMES_API.md has source-line citations and is clearly maintained against the actual Hermes signatures — drift will be obvious.                                                           
  - Conftest stubs (StubEmbedder, mock_anthropic, mock_cohere, mock_cross_encoder) keep heavy deps off the dev box.                                                                           
                                                                                                                                                                                              
  ---                                                                                                                                                                                         
  Suggested triage order if you want to act on this                                                                                                                                           
                                                   
  1. H2 (rmtree guard) — 5-line fix, prevents catastrophic data loss.
  2. H3 (Cohere empty fallback) — one-line change, currently silently breaks user queries.                                                                                                    
  3. M3 (engine load consistency check) — a few lines, makes M1/M2 visible instead of corrupting silently.                                                                                    
  4. M7 (license syntax) — trivial, suppresses setuptools warning.                                                                                                                            
  5. L4 (stderr for warnings) — trivial, fixes JSON-output composition.                                                                                                                       
  6. H1 (extract_md preamble) — design call; document or fix.                                                                                                                                 
  7. The rest in your own order.                                                                                                                                                              
                                                                                                                                                                                              
  Want me to /schedule a follow-up agent to implement the trivial fixes (H2, H3, M3, M7, L4) once you've decided on the design questions for H1 / M5 / L1?                                    
                  
✻ Worked for 12m 38s                                                                                                                                                                          
                  
※ recap: You asked for an exhaustive code review of the advanced-rag plugin; I delivered findings ranked by severity, with H2 (rmtree guard), H3 (Cohere empty fallback), and H1 (extract_md  
  drops intro) as the top issues. Next: pick which fixes to implement. (disable recaps in /config)
