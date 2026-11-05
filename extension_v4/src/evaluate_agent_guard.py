"""Post-hoc paired replay using the same raw planner outputs, not fresh generation."""

import asyncio
import argparse
import json
from agent import validate_plan
from identifier_guard import preserve_identifiers
from evidence import validate
from local_llm import LocalLLM
from mcp_client import connect_server, unpack
from settings import V4, dump


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    prefix = "agent_strict" if args.strict else "agent_guard"
    original = json.loads((V4 / "results/agent_rows.json").read_text())
    llm = LocalLLM()
    rows = []
    async with connect_server() as (session, _):
        for row in original:
            result = dict(
                id=row["id"],
                original_target_cited=row["target_cited"],
                target_pmid=row["target_pmid"],
                original_plan=row.get("plan"),
                raw_planner_output=row["planner"]["raw_output"],
            )
            try:
                plan = validate_plan(row["planner"]["raw_output"])
                effective, changes = preserve_identifiers(
                    plan, row["question"], require_numeric=args.strict
                )
                result.update(effective_plan=effective, changes=changes)
                if not changes:
                    result.update(
                        reused_original=True,
                        target_cited=row["target_cited"],
                        user_response=row["user_response"],
                    )
                else:
                    call = await session.call_tool(
                        effective["tool"], effective["arguments"]
                    )
                    if call.is_error:
                        raise ValueError("mcp_tool_error")
                    found = unpack(call)
                    docs = (
                        found["documents"]
                        if effective["tool"] == "search_documents"
                        else [found] if found.get("found") else []
                    )
                    visible = {d["pmid"]: d["visible_text"] for d in docs}
                    generated = llm.generate(
                        "evidence-first-v1",
                        dict(
                            QUESTION=row["question"],
                            DOCUMENTS=[
                                dict(pmid=k, text=v) for k, v in visible.items()
                            ],
                        ),
                    )
                    checked = validate(generated["raw_output"], visible)
                    result.update(
                        reused_original=False,
                        tool_result=found,
                        generated=generated,
                        validation=checked,
                        target_cited=any(
                            e["pmid"] == row["target_pmid"]
                            for e in checked["verified_excerpts"]
                        ),
                        user_response=checked["user_response"],
                    )
            except ValueError as exc:
                result.update(error=str(exc), target_cited=False)
            rows.append(result)
            dump(V4 / f"results/{prefix}_rows.json", rows)
            print(
                row["id"],
                result.get("changes"),
                result["target_cited"],
                result.get("error"),
                flush=True,
            )
    dump(
        V4 / f"results/{prefix}_summary.json",
        dict(
            n=len(rows),
            strict=args.strict,
            changed_plans=sum(bool(r.get("changes")) for r in rows),
            original_target_cited=sum(r["original_target_cited"] for r in rows),
            guarded_target_cited=sum(r["target_cited"] for r in rows),
            answerable_n=sum(bool(r["target_pmid"]) for r in rows),
            scope="Post-hoc diagnostic replay of saved planner outputs on same tasks; unchanged plans reuse original answers; not fresh blinded agent benchmark",
        ),
    )


if __name__ == "__main__":
    asyncio.run(main())
