"""Recompute saved metrics and context-bound quotes; never re-generates model output."""

import argparse
import json
import sqlite3
from pathlib import Path
import numpy as np
from settings import DB, ROOT, V4, dump, sha
from evidence import validate, visible_document, verify_quote
from agent import validate_plan
from evaluate_retrieval import metrics


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    args = parser.parse_args()
    checks = {}
    lock = read(V4 / "results/evaluation_lock.json")
    checks["evaluation_inputs_unchanged"] = all(
        sha(V4 / p) == h for p, h in lock["sha256"].items()
    )
    queries = read(V4 / "evaluation/queries.json")["queries"]
    qbyid = {q["id"]: q for q in queries}
    checks["query_groups_disjoint"] = not (
        {q["source_group"] for q in queries if q["split"] == "dev"}
        & {q["source_group"] for q in queries if q["split"] == "test"}
    )
    rows = read(V4 / "results/retrieval_rows.json")
    summaries = read(V4 / "results/retrieval_summary.json")["summaries"]
    checks["retrieval_coverage"] = len(rows) == len(queries) * 5 and len(
        {(r["query_id"], r["method"]) for r in rows}
    ) == len(rows)
    checks["retrieval_ranks_recomputed"] = all(
        r["target_rank"]
        == (
            r["retrieved"].index(r["target_pmid"]) + 1
            if r["target_pmid"] in r["retrieved"]
            else None
        )
        and r["target_pmid"] == qbyid[r["query_id"]]["target_pmid"]
        and r["split"] == qbyid[r["query_id"]]["split"]
        for r in rows
    )
    summary_ok = True
    for name, methods in summaries.items():
        split, kind = name.split("-")
        for method, recorded in methods.items():
            actual = metrics(
                [
                    r
                    for r in rows
                    if r["method"] == method
                    and r["split"] == split
                    and (kind == "all" or r["kind"] == kind)
                ]
            )
            summary_ok &= all(abs(actual[k] - recorded[k]) < 1e-12 for k in actual)
    checks["retrieval_metrics_recomputed"] = summary_ok
    policy = read(V4 / "results/retrieval_policy.json")
    preference = ["bm25", "hybrid", "dense", "tfidf", "bm25_raw"]
    expected = max(preference, key=lambda m: summaries["dev-all"][m]["mrr10"])
    checks["policy_selected_only_from_dev"] = policy[
        "selected_method"
    ] == expected and policy["evaluation_lock_sha256"] == sha(
        V4 / "results/evaluation_lock.json"
    )
    db = sqlite3.connect(DB.resolve().as_uri() + "?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    corpus = {
        r["pmid"]: dict(r)
        for r in db.execute("SELECT pmid,title,abstract,content_hash FROM documents")
    }
    checks["query_target_hashes_match"] = all(
        corpus[q["target_pmid"]]["content_hash"] == q["target_content_hash"]
        for q in queries
    )
    try:
        db.execute("UPDATE documents SET title=title WHERE 0")
        readonly = False
    except sqlite3.OperationalError:
        readonly = True
    checks["database_read_only"] = readonly
    llm_rows = read(V4 / "results/llm_rows.json")
    tasks = read(V4 / "evaluation/qa_tasks.json")
    tbyid = {t["id"]: t for t in tasks}
    checks["llm_conditions_complete"] = (
        len(llm_rows) == 36
        and len({(r["task_id"], r["condition"]) for r in llm_rows}) == 36
    )
    quote_ok = True
    metadata_ok = True
    for row in llm_rows:
        docs = {pmid: visible_document(corpus[pmid]) for pmid in row["retrieved_pmids"]}
        quote_ok &= all(
            sha_text(text) == row["visible_context_hashes"][pmid]
            for pmid, text in docs.items()
        )
        actual = validate(row["raw_output"], docs)
        quote_ok &= all(
            row[k] == actual[k]
            for k in ["schema_valid", "status", "citations", "verified_excerpts"]
        )
        if row["condition"] == "no-context":
            quote_ok &= not docs and not row["verified_excerpts"]
        task = tbyid[row["task_id"]]
        metadata_ok &= (
            row["target_pmid"] == task["target_pmid"]
            and row["answerable"] == task["answerable"]
        )
        metadata_ok &= row["retrieval_method"] == policy["selected_method"]
    checks["llm_quotes_revalidated_visible_context"] = bool(quote_ok)
    checks["llm_task_and_policy_metadata"] = bool(metadata_ok)
    llm_summaries = read(V4 / "results/llm_summary.json")["summaries"]
    llm_ok = True
    for condition, recorded in llm_summaries.items():
        selected = [r for r in llm_rows if r["condition"] == condition]
        cited = [c for r in selected for c in r["citations"]]
        supported = [r for r in selected if r["answerable"]]
        unsupported = [r for r in selected if not r["answerable"]]
        actual = dict(
            n=len(selected),
            schema_valid=sum(r["schema_valid"] for r in selected),
            citation_items=len(cited),
            exact_valid_items=sum(c["valid"] for c in cited),
            supported_with_context=sum(r["target_in_context"] for r in supported),
            supported_target_cited=sum(r["target_cited"] for r in supported),
            supported_with_any_excerpt=sum(
                bool(r["verified_excerpts"]) for r in supported
            ),
            unsupported_abstained=sum(not r["verified_excerpts"] for r in unsupported),
            unsupported_n=len(unsupported),
            median_seconds=float(np.median([r["seconds"] for r in selected])),
        )
        llm_ok &= all(abs(recorded[k] - v) < 1e-12 for k, v in actual.items())
    checks["llm_summaries_recomputed"] = bool(llm_ok)
    agent = read(V4 / "results/agent_rows.json")
    agent_ok = True
    for row in agent:
        agent_ok &= len(row["trace"]) <= 7 and all(
            t["tool"] in {"search_documents", "get_document", "verify_evidence"}
            for t in row["trace"]
        )
        if "plan" in row:
            agent_ok &= validate_plan(row["planner"]["raw_output"]) == row["plan"]
        visible = {}
        for t in row["trace"]:
            if t["tool"] == "search_documents":
                visible.update(
                    {
                        r["pmid"]: r["visible_text"]
                        for r in t["result"].get("documents", [])
                    }
                )
            if t["tool"] == "get_document" and t["result"].get("found"):
                visible[t["result"]["pmid"]] = t["result"]["visible_text"]
        agent_ok &= all(
            verify_quote(e["pmid"], e["quote"], visible)["valid"]
            for e in row["user_response"]["excerpts"]
        )
    checks["agent_plans_bounds_and_citations"] = bool(agent_ok)
    smoke = read(V4 / "results/mcp_smoke.json")
    checks["mcp_protocol_smoke"] = smoke["passed"] and all(smoke["checks"].values())
    old = read(ROOT / "followup_v2/results/v1_preservation_sha256.json")
    checks["v1_preserved"] = all(sha(ROOT / p) == h for p, h in old.items())
    frozen = read(ROOT / "extension_v3/results/final_audit_locked_model_hashes.json")

    def locate(p):
        if p.startswith("external_llm_snapshot/"):
            import os

            model = Path(
                os.environ.get(
                    "LITERATURE_LLM_PATH",
                    str(
                        ROOT.parent
                        / "kkbb-ddook-ddak-back/var/research_models/canvas_generator"
                    ),
                )
            )
            return model / Path(p).name
        return ROOT.parent / p

    checks["v3_frozen_models_and_policies_preserved"] = all(
        sha(locate(p)) == h for p, h in frozen.items()
    )
    from verify_diagnostics import check_diagnostics

    checks.update(check_diagnostics(corpus))
    manifest = V4 / "results/delivery_manifest.json"
    if manifest.exists():
        checks["delivery_hashes_match"] = all(
            sha(ROOT / p) == h for p, h in read(manifest).items()
        )
    report = dict(
        passed=all(checks.values()),
        checks=checks,
        v1_preserved_files=len(old),
        v3_preserved_files=len(frozen),
        retrieval_rows=len(rows),
        llm_runs=len(llm_rows),
        agent_runs=len(agent),
    )
    if args.output:
        dump(args.output, report)
    print(json.dumps(report, indent=2))
    db.close()
    if not report["passed"]:
        raise SystemExit(1)


def sha_text(text):
    import hashlib

    return hashlib.sha256(text.encode()).hexdigest()


if __name__ == "__main__":
    main()
