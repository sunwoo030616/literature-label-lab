"""Post-hoc, clearly labelled diagnosis on the same six Korean query probes."""

import asyncio
import json
from agent import validate_plan
from local_llm import LocalLLM
from mcp_client import connect_server, unpack
from settings import V4, dump, sha


async def main():
    queries = [
        q
        for q in json.loads((V4 / "evaluation/queries.json").read_text())["queries"]
        if q["kind"] == "korean"
    ]
    original = {
        r["query_id"]: r
        for r in json.loads((V4 / "results/retrieval_rows.json").read_text())
        if r["method"] == "bm25"
    }
    llm = LocalLLM()
    rows = []
    async with connect_server() as (session, _):
        for q in queries:
            gen = llm.generate("planner-v1", dict(question=q["query"]))
            result = dict(
                id=q["id"],
                query=q["query"],
                target_pmid=q["target_pmid"],
                original_rank=original[q["id"]]["target_rank"],
                **gen
            )
            try:
                plan = validate_plan(gen["raw_output"])
                if plan["tool"] != "search_documents":
                    raise ValueError("planner_did_not_search")
                args = dict(plan["arguments"])
                args.update(limit=10, method="bm25")
                tool = await session.call_tool("search_documents", args)
                if tool.is_error:
                    raise ValueError("mcp_tool_error")
                ids = [r["pmid"] for r in unpack(tool)["documents"]]
                result.update(
                    plan=plan,
                    retrieved=ids,
                    rewritten_rank=(
                        ids.index(q["target_pmid"]) + 1
                        if q["target_pmid"] in ids
                        else None
                    ),
                )
            except ValueError as exc:
                result.update(error=str(exc), rewritten_rank=None)
            rows.append(result)
            dump(V4 / "results/rewrite_rows.json", rows)
            print(q["id"], result.get("plan"), result["rewritten_rank"], flush=True)
    dump(
        V4 / "results/rewrite_summary.json",
        dict(
            n=len(rows),
            original_hit10=sum(bool(r["original_rank"]) for r in rows),
            rewritten_hit10=sum(bool(r["rewritten_rank"]) for r in rows),
            rewritten_hit5=sum(
                bool(r["rewritten_rank"]) and r["rewritten_rank"] <= 5 for r in rows
            ),
            mean_generation_seconds=sum(r["seconds"] for r in rows) / len(rows),
            scope="Post-hoc repeated six Korean probes, not a fresh holdout",
            diagnostic_protocol_sha256=sha(V4 / "evaluation/REWRITE_DIAGNOSTIC.md"),
        ),
    )


if __name__ == "__main__":
    asyncio.run(main())
