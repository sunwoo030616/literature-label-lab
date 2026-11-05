"""Cross-check post-hoc improvements while leaving the locked baseline unchanged."""

import json
import re
from agent import validate_plan
from evidence import validate, visible_document, verify_quote
from identifier_guard import preserve_identifiers
from quote_recovery import recover_edge_ellipsis
from settings import V4


def read(name):
    return json.loads((V4 / "results" / name).read_text())


def check_diagnostics(corpus):
    checks = {}
    baseline = {(r["task_id"], r["condition"]): r for r in read("llm_rows.json")}
    recovery = read("quote_recovery.json")
    ok = True
    new = 0
    for row in recovery["runs"]:
        source = baseline[(row["task_id"], row["condition"])]
        visible = {p: visible_document(corpus[p]) for p in source["retrieved_pmids"]}
        recalculated = recover_edge_ellipsis(source["raw_output"], visible)
        ok &= row["recovered"] == recalculated["recovered_excerpts"] and row[
            "after_excerpts"
        ] == len(recalculated["verified_excerpts"])
        new += len(recalculated["recovered_excerpts"])
    checks["quote_recovery_recomputed"] = bool(
        ok and new == recovery["new_exact_excerpts"]
    )
    original = {r["id"]: r for r in read("agent_rows.json")}
    for prefix, strict in [("agent_guard", False), ("agent_strict", True)]:
        ok = True
        items = read(prefix + "_rows.json")
        for row in items:
            source = original[row["id"]]
            try:
                expected, changes = preserve_identifiers(
                    validate_plan(source["planner"]["raw_output"]),
                    source["question"],
                    strict,
                )
            except ValueError as exc:
                ok &= row.get("error") == str(exc)
                continue
            ok &= row["effective_plan"] == expected and row["changes"] == changes
            if row.get("reused_original"):
                ok &= row["target_cited"] == source["target_cited"]
            else:
                tool = row["tool_result"]
                documents = (
                    tool.get("documents", [])
                    if expected["tool"] == "search_documents"
                    else [tool] if tool.get("found") else []
                )
                visible = {r["pmid"]: r["visible_text"] for r in documents}
                result = validate(row["generated"]["raw_output"], visible)
                ok &= (
                    result["verified_excerpts"]
                    == row["validation"]["verified_excerpts"]
                )
                ok &= row["target_cited"] == any(
                    e["pmid"] == row["target_pmid"] for e in result["verified_excerpts"]
                )
                for doc in documents:
                    text = (
                        corpus[doc["pmid"]]["title"]
                        + " "
                        + corpus[doc["pmid"]]["abstract"]
                    )
                    numbers = {
                        n.replace(",", "") for n in re.findall(r"\d[\d,]*", text)
                    }
                    ok &= all(
                        anchor.replace(",", "") in numbers
                        for anchor in expected["arguments"].get(
                            "required_numeric_anchors", []
                        )
                    )
        summary = read(prefix + "_summary.json")
        ok &= summary["guarded_target_cited"] == sum(r["target_cited"] for r in items)
        checks[prefix + "_recomputed"] = bool(ok)
    rewrite = read("rewrite_rows.json")
    summary = read("rewrite_summary.json")
    checks["rewrite_summary_recomputed"] = (
        len(rewrite) == summary["n"] == 6
        and sum(bool(r["rewritten_rank"]) for r in rewrite)
        == summary["rewritten_hit10"]
        and all(
            r["rewritten_rank"]
            == (
                r["retrieved"].index(r["target_pmid"]) + 1
                if r["target_pmid"] in r.get("retrieved", [])
                else None
            )
            for r in rewrite
        )
    )
    skill = read("skill_question_strict.json")
    expected, changes = preserve_identifiers(
        validate_plan(skill["planner"]["raw_output"]), skill["question"], True
    )
    checks["skill_full_path_source_binding"] = (
        skill["plan"] == expected
        and skill["identifier_guard_changes"] == changes
        and any(e["pmid"] == "32882643" for e in skill["user_response"]["excerpts"])
        and bool(skill["trace"])
    )
    checks["skill_full_path_quotes_valid"] = all(
        verify_quote(
            e["pmid"],
            e["quote"],
            {p: visible_document(corpus[p]) for p in corpus if p == e["pmid"]},
        )["valid"]
        for e in skill["user_response"]["excerpts"]
    )
    return checks
