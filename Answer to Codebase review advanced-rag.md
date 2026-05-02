Referencing your **Codebase review advanced-rag.md** report, please implement the identified fixes and improvements according to the following design decisions and triage priorities:

**1. Chosen Design Options:**
*   **H1:** Implement **Option 2** (create a synthetic 'preamble' parent for text preceding the first `##` heading).
*   **M5:** Implement **Option 1** (use `content_hash` as a tiebreaker in `manifest_diff` when `mtime` and `size` match).
*   **L1:** Implement **Option 2** (wire the query path to use the `embed_row` column and remove the redundant `_chunk_ids` array from the `.npz` file).

**2. Implementation Order:**
Follow the **Suggested triage order** defined at the end of your report exactly (starting with H2, H3, M3, M7, and L4)[cite: 1]. Once those are complete, proceed with the implementation of H1, M5, L1 as specified above, followed by all remaining Medium and Low severity items.

**3. Engineering Constraints:**
*   Maintain the **pure-core/adapter split**.
*   Ensure all file operations remain **atomic** (using `.tmp` and `os.replace`).
*   Update the test suite to cover the new logic for the preamble (H1), the `rmtree` safety guard (H2), and the database-driven query path (L1). 
*   Verify that all 113 existing tests continue to pass after your changes.

Begin with the high-priority safety fix (H2)."
