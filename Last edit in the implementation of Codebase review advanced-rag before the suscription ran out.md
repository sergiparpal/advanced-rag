● Update(tests/test_indexing.py)                                                                                                                                                              
  ⎿  Added 23 lines, removed 6 lines                                                                                                                                                        
       72                                                                                                                                                                                     
       73                                                                                                                                                                                     
       74  def test_embed_row_invariant(tmp_data_dir, tmp_path, stub_embedder):
       75 -    """Chunk row N in canonical SQLite order ↔ row N of embeddings.npz."""                                                                                                         
       75 +    """Chunk row N in canonical SQLite order ↔ row N of embeddings.npz, and                                                                                                        
       76 +    `chunks.embed_row` is the on-disk source of truth for that mapping.                                                                                                     
       77 +    `.npz` no longer carries `chunk_ids`; the engine derives the list from                                                                                                         
       78 +    SQLite via `iter_chunks_ordered()`."""                                                                                                                                       
       79      docs = _stage(tmp_path)                                                                                                                                                        
       80      store = Store()                                                                                                                                                              
       81      index_path(docs, store=store, embedder=stub_embedder)                                                                                                                          
       82                                                                          
       80 -    embeddings, chunk_ids_in_npz = store.load_embeddings(store.npz_path)                                                                                                    
       81 -    canonical = [c.id for c in store.iter_chunks_ordered()]                                                                                                                 
       82 -    assert chunk_ids_in_npz == canonical                                                                                                                                    
       83 -    assert embeddings.shape[0] == len(canonical)                                                                                                                            
       84 -    # embed_row column in SQLite must match the row index it occupies                                                                                                       
       83 +    embeddings = store.load_embeddings(store.npz_path)                                                                                                                      
       84      rows = list(store.iter_chunks_ordered())
       85 +    assert embeddings.shape[0] == len(rows)                                                                                                                                 
       86 +    # embed_row column in SQLite must match the row index it occupies — this                                                                                                
       87 +    # is now what the query path uses to map dense-search hits back to chunks.                                                                                              
       88      for row_idx, row in enumerate(rows):
       89          assert row.embed_row == row_idx
       90 +                                                                                                                                                                            
       91 +                                                                                                                                                                            
       92 +def test_engine_chunk_ids_match_canonical_order(tmp_data_dir, tmp_path, stub_embedder):                                                                                     
       93 +    """After indexing, the engine's `_chunk_ids` (loaded from SQLite) must                                                                                                  
       94 +    align with the .npz row index order."""                                                                                                                                 
       95 +    from advanced_rag.engine import RAGEngine                                                                                                                               
       96 +    docs = _stage(tmp_path)                                                                                                                                                 
       97 +    store = Store()                                                                                                                                                         
       98 +    index_path(docs, store=store, embedder=stub_embedder)                                                                                                                   
       99 +                                                                                                                                                                            
      100 +    eng = RAGEngine(store=store, embedder=stub_embedder)                                                                                                                    
      101 +    eng._ensure_loaded()                                                                                                                                                    
      102 +    canonical = [c.id for c in store.iter_chunks_ordered()]                                                                                                                 
      103 +    assert eng._chunk_ids == canonical                                                                                                                                      
      104 +    assert eng._embeddings.shape[0] == len(canonical)                                                                                                                       

● Bash(python -m pytest -q tests/test_indexing.py 2>&1 | tail -10)
  ⎿  .......                                                                  [100%]
  ⎿  You've hit your limit · resets 4am (Europe/Madrid)
                                                            
✻ Brewed for 5m 2s                                                    
                                                           
  11 tasks (3 done, 1 in progress, 7 open)                                                                                                                                                    
  ◼ L1: Drop chunk_ids from .npz; query path uses embed_row from DB
  ◻ M7: Update pyproject.toml license to SPDX expression                                                                                                                                      
  ◻ L4: Route indexing warnings to stderr/logging
  ◻ H1: Create synthetic preamble parent in extract_md
  ◻ M5: Use content_hash as tiebreaker in manifest_diff                                                                                                                                       
   … +3 pending, 3 completed

