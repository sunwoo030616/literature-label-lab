# Post-hoc Korean query rewrite diagnostic

Registered after the first retrieval ranking results, before any planner outputs.

All six fixed Korean queries had no known target in top10 under any initial method. Diagnose the English-corpus/language mismatch by running the already frozen planner-v1 once per Korean query, then search its selected English keywords with the dev-selected BM25 method. Preserve generated plans, tool errors and target rank. Compare to the original Korean BM25 ranks on these same six queries.

No prompt edits, no default retrieval policy edits, no new relevance labels. This is post-hoc analysis on previously inspected queries, not a new blind test or proof of Korean search generalization. Query rewriting adds a model call; measure and report latency.
