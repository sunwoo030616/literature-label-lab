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
