"""Score paired planner outputs with fixed evidence budgets on the released index."""

import argparse
import asyncio
from collections import defaultdict
from contextlib import closing
import json
from pathlib import Path
import sqlite3
from statistics import median

from evaluate_plans import VERSIONS, build_cases
from knowledge_base import DEFAULT_DB, V5, connect, search, snapshot, words
from science_agent import Facet, connect_server, review


def near_duplicate(facets):
    terms = [set(words(f["query"])) for f in facets]
    return any(
        left | right and len(left & right) / len(left | right) >= 0.75
        for index, left in enumerate(terms) for right in terms[index + 1:]
    )


def fixed_candidates(packet, limit=3):
    """Give each facet a turn, then fill the same total passage budget."""
    accepted = []
    for depth in range(3):
        for facet in packet["facets"]:
            if depth < len(facet["evidence"]):
                accepted.append(facet["evidence"][depth]["pmid"])
                if len(accepted) == limit:
                    return accepted
    return accepted


async def score(path, db_path, allow_partial):
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as run:
        fingerprint, raw_manifest = run.execute("SELECT * FROM manifest").fetchone()
        cases = json.loads(raw_manifest)["cases"]
        sample = sum(r["id"].startswith("sample-") for r in cases)
        evaluation_snapshot, fresh_cases = build_cases(db_path, sample)
        if cases != fresh_cases:
            raise ValueError("Sampled cases changed between the planner run and scoring")
        results = {(case, version): json.loads(row)
                   for case, version, row in run.execute("SELECT case_id, version, result FROM plans")}
    paired_cases = [case for case in cases if all((case["id"], version) in results for version in VERSIONS)]
    if len(paired_cases) != len(cases) and not allow_partial:
        raise ValueError("The checkpoint is incomplete. Use --allow-partial for a diagnostic")
    if not paired_cases:
        raise ValueError("No paired results available")
    rows = []
    with closing(connect(db_path)) as db:
        if snapshot(db) != evaluation_snapshot:
            raise ValueError("Scoring database changed during setup")
        async with connect_server(db_path) as session:
            for case in paired_cases:
                target = case["pmid"]
                baseline = None
                if not case["source_scope"]:
                    baseline = any(r["pmid"] == target for r in search(
                        db, case["question"][:250], limit=3
                    )["candidates"])
                row = dict(id=case["id"], kind=case["kind"], requested_facets=case["requested_facets"],
                           target=target, source_scope=case["source_scope"], baseline_hit=baseline, methods={})
                for version in VERSIONS:
                    generated = results[case["id"], version]
                    plan = generated.get("plan")
                    item = dict(valid_plan=plan is not None, planner_seconds=round(generated["seconds"], 3),
                                hit=False, correct_facet_count=False, duplicate_facets=False,
                                source_constraint_ok=False if case["source_scope"] else None)
                    if plan:
                        facets = [Facet(**entry, **generated["scope"]) for entry in plan]
                        packet = await review(session, facets)
                        if packet["snapshot"] != evaluation_snapshot:
                            raise ValueError("Scoring database changed during retrieval")
                        ids = fixed_candidates(packet)
                        item.update(hit=target in ids,
                                    correct_facet_count=len(facets) == case["requested_facets"],
                                    duplicate_facets=near_duplicate(plan), returned_pmids=ids,
                                    source_constraint_ok=(bool(ids) and set(ids) == {target})
                                    if case["source_scope"] else None,
                                    queries=[f.query for f in facets])
                    elif "error" in generated:
                        item["error"] = generated["error"]
                    row["methods"][version] = item
                rows.append(row)
    groups = defaultdict(list)
    for row in rows:
        groups[row["kind"]].append(row)
        groups["all"].append(row)
    summary = {}
    for name, group in groups.items():
        by_version = {
            version: {
                **{key: sum(r["methods"][version][key] is True for r in group)
                   for key in ("valid_plan", "hit", "correct_facet_count", "duplicate_facets", "source_constraint_ok")},
                "median_planner_seconds": round(median(r["methods"][version]["planner_seconds"] for r in group), 3),
            }
            for version in VERSIONS
        }
        first, second = VERSIONS
        paired = [(row["methods"][first]["hit"], row["methods"][second]["hit"]) for row in group]
        summary[name] = dict(n=len(group), scoped_n=sum(r["source_scope"] for r in group),
                             baseline_n=sum(r["baseline_hit"] is not None for r in group),
                             baseline_hits=sum(r["baseline_hit"] is True for r in group),
                             v2_gains=sum(not a and b for a, b in paired),
                             v2_regressions=sum(a and not b for a, b in paired), versions=by_version)
    return dict(checkpoint_fingerprint=fingerprint, planned_snapshot=json.loads(raw_manifest)["snapshot"],
                evaluation_snapshot=evaluation_snapshot, cases_planned=len(cases), cases_paired=len(rows),
                summary=summary, rows=rows)


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    result = await score(args.checkpoint, args.db, args.allow_partial)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
