# Post-hoc strict source constraint diagnosis

The soft identifier guard did not improve target citation (3/4 remained 3/4). Its OR BM25 query still retrieved articles without the requested patient count. Worse, the LLM proposed the question's 3449 as an answer while quoting an unrelated hydrogen-bond sentence, which passed occurrence checks. Preserve this failure.

Before strict replay: add an opt-in required-numeric-source filter to lexical search. For every 3+-digit numeric anchor extracted from the question, require its exact numeric token (comma-normalized) in the candidate's original title/abstract. This is a parameterized SQLite predicate; it does not use gold target IDs. Quotes remain restricted to the visible excerpt. No matching source means an empty context, not forced answering.

Replay the same original planner outputs with this filter, reusing unchanged answers. Save separate agent_strict files. No prompt changes, no baseline overwrite, no new blind claim. Numeric matching is not semantic entailment and can reject valid sources with numbers spelled out, alternate notation or irrelevant numeric requests.
