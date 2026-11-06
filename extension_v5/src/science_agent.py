"""Plan a bounded evidence review, query MCP, and independently recheck every quotation."""

import argparse
import asyncio
from contextlib import asynccontextmanager
import json
from pathlib import Path
import re
import sys
import time
from typing import Annotated

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from knowledge_base import DEFAULT_DB, V5, normalize_doi

MAX_FACETS = 3
MAX_PASSAGES = 3
PMID_MENTION = re.compile(r"\bPMID\s*[:=]?\s*([1-9][0-9]{0,11})(?![0-9])", re.IGNORECASE)
PUBMED_URL = re.compile(r"https?://pubmed\.ncbi\.nlm\.nih\.gov/([1-9][0-9]{0,11})(?![0-9])", re.IGNORECASE)
DOI_MENTION = re.compile(r"\b10\.[0-9]{4,9}/[^\s<>\"']+", re.IGNORECASE)


class Topic(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, strict=True)
    label: Annotated[str, Field(min_length=1, max_length=100)]
    query: Annotated[str, Field(min_length=1, max_length=250)]


class Facet(Topic):
    pmid: Annotated[str, Field(pattern=r"^(?:[1-9][0-9]{0,11})?$")] = ""
    doi: Annotated[str, Field(max_length=200)] = ""
    require_all: bool = False

    @model_validator(mode="after")
    def validate_source(self):
        if self.pmid and self.doi:
            raise ValueError("Specify a PMID or DOI for each facet")
        if self.doi and not normalize_doi(self.doi):
            raise ValueError("Invalid DOI in facet")
        return self


Topics = TypeAdapter(Annotated[list[Topic], Field(min_length=1, max_length=MAX_FACETS)])
Facets = TypeAdapter(Annotated[list[Facet], Field(min_length=1, max_length=MAX_FACETS)])


@asynccontextmanager
async def connect_server(db_path):
    params = StdioServerParameters(
        command=sys.executable, args=[str(V5 / "src/science_mcp.py"), "--db", str(db_path)]
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


def explicit_scope(question):
    pmids = set(PMID_MENTION.findall(question)) | set(PUBMED_URL.findall(question))
    dois = {m.rstrip(".,;:?!").lower() for m in DOI_MENTION.findall(question)}
    if len(pmids) + len(dois) > 1:
        raise ValueError("Use explicit facets to assign multiple source identifiers to their topics")
    return dict(pmid=next(iter(pmids), ""), doi=next(iter(dois), ""))


def plan_question(llm, question):
    question = question.strip()
    if not 1 <= len(question) <= 1500:
        raise ValueError("Question must be 1–1500 characters")
    scope = explicit_scope(question)
    generated = llm.generate("science-facets-v1", {"question": question})
    topics = Topics.validate_json(generated["raw_output"])
    return [Facet(**topic.model_dump(), **scope) for topic in topics], generated


def local_model():
    # The V4 runner provides the existing offline model, sampler, and prompt hashes.
    sys.path.append(str(V5.parent / "extension_v4/src"))
    from local_llm import LocalLLM

    return LocalLLM(registry_path=V5 / "prompts.json")


async def review(session, facets, limit=MAX_PASSAGES):
    if not 1 <= len(facets) <= MAX_FACETS or not 1 <= limit <= MAX_PASSAGES:
        raise ValueError("Facet or passage budget exceeded")
    trace = []
    packet = []
    identity = None
    started = time.perf_counter()

    async def call(tool, args):
        if len(trace) >= MAX_FACETS * (1 + MAX_PASSAGES):
            raise RuntimeError("Tool call budget exhausted")
        start = time.perf_counter()
        response = await session.call_tool(tool, args)
        audit_args = {name: value for name, value in args.items() if name != "quote"}
        trace.append({"tool": tool, "arguments": audit_args, "error": response.is_error,
                      "milliseconds": round((time.perf_counter() - start) * 1000, 3)})
        if response.is_error or not isinstance(response.structured_content, dict):
            raise RuntimeError(f"MCP call failed: {tool}")
        return response.structured_content

    def result(status):
        return dict(status=status, snapshot=identity, facets=packet, trace=trace,
                    seconds=round(time.perf_counter() - started, 4),
                    scope="LitCovid abstracts; exact quotations, human interpretation required")

    for facet in facets:
        found = await call("search_evidence", {
            "query": facet.query, "pmid": facet.pmid, "doi": facet.doi,
            "limit": limit, "require_all": facet.require_all,
        })
        if identity is None:
            identity = found["snapshot"]
        if found["snapshot"] != identity:
            packet.clear()
            return result("snapshot_changed")

        verified = []
        rejected = []
        for candidate in found["candidates"]:
            key = {field: candidate[field] for field in ("pmid", "version", "start", "end", "quote")}
            check = await call("verify_passage", {**key, "expected_snapshot": identity})
            if check["reason"] == "snapshot_changed":
                packet.clear()
                return result("snapshot_changed")
            if check["valid"]:
                verified.append({**candidate, "provenance": check["provenance"]})
            else:
                rejected.append({"pmid": candidate["pmid"], "reason": check["reason"]})
        packet.append(dict(label=facet.label, query=facet.query, pmid=facet.pmid, doi=facet.doi,
                           status="verified_candidates" if verified else (
                               "verification_failed" if rejected else found["status"]
                           ), evidence=verified, rejected=rejected))
    covered = sum(bool(item["evidence"]) for item in packet)
    status = "complete" if covered == len(packet) else "partial" if covered else "no_evidence"
    return result(status)


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    request = parser.add_mutually_exclusive_group(required=True)
    request.add_argument("--question")
    request.add_argument("--facets", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.facets:
        try:
            facets = Facets.validate_json(args.facets.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            parser.error(str(exc))
        planner = None
    else:
        if not 1 <= len(args.question.strip()) <= 1500:
            parser.error("Question must be 1–1500 characters")
        try:
            explicit_scope(args.question)
            facets, planner = plan_question(local_model(), args.question)
        except ValueError as exc:
            parser.error(str(exc))
    async with connect_server(args.db) as session:
        try:
            result = await review(session, facets)
        except RuntimeError as exc:
            parser.error(str(exc))
    if planner:
        result["planner"] = planner
    encoded = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")


if __name__ == "__main__":
    asyncio.run(main())
