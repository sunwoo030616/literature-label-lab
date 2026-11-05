---
name: literature-evidence-audit
description: Audit this repository's LitCovid retrieval and local Qwen evidence runs, or answer a literature question through its read-only MCP workflow. Use for reproducible V4 retrieval and quotation checks, not clinical recommendations or full-paper review.
---

# Literature evidence audit

Work from this repository root. V4 code is in `extension_v4`; prior V1–V3 artifacts are preserved controls.

For an existing result audit, run `scripts/run_audit.py` using a Python environment with the V4 requirements. It recomputes evaluation metrics and checks locked inputs and displayed quotations without generating new model answers. Explain failures instead of replacing the evidence files.

For a new question, run the helper with `--question` and `--output` pointing to a new JSON file. It uses the local free Qwen planner, actual stdio MCP tools and the evidence-first prompt. Read the JSON trace, including failed plans and rejected citations. Report only returned exact excerpts as mechanically verified. Any generated answer remains an unverified proposal; occurrence of a quote does not establish relevance or entailment.

If model/index/data files are missing, follow `extension_v4/docs/REPRODUCE.md`; do not substitute a paid API. Do not silently recreate locked evaluation outputs or retune on the held-out queries. New experiments need separate result names and a new predeclared policy.

When interpreting results, read `extension_v4/docs/RESULTS_WALKTHROUGH.md`. The retrieval benchmark is AI-authored known-item search, not expert relevance judgments. The corpus contains titles/abstracts, including historical official train/dev/test documents; it must not be reused as classifier training data. SQL is parameterized inside the server. Allowed tools are search, PMID lookup, quote validation and corpus counts; arbitrary SQL or writes are not exposed.

The helper does not publish, push, merge or install account-wide tools. Follow the current user's authorization for those actions.

For a new question that names a specific source, `--preserve-identifiers` binds an explicit PMID and preserves 3+-digit numeric anchors in search queries. This is a limited heuristic; small counts and semantic intent can still be lost. `--repair-edge-ellipsis` optionally recovers only a unique exact visible substring after stripping leading/trailing ellipses. Both are opt-in post-hoc improvements and preserve the rejected originals. Do not describe their repeated-task diagnostic results as a new blinded evaluation.

Numeric keywords in an OR query are optional matches: merely appending a missing count did not resolve an observed source mismatch and led to an irrelevant but exact quotation. For questions tied to an explicit 3+-digit count, `--require-numeric-anchors` additionally requires comma-normalized exact numeric tokens in the original candidate text. This is lexical source filtering, not entailment; it can reject a source using a different notation. Original soft-guard failures are retained in `agent_guard_rows.json`.
