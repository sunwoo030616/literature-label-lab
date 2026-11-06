"""Compare exact evidence coverage and verify the KB against the entire real snapshot."""

import argparse
import asyncio
from collections import Counter
from contextlib import closing
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from statistics import median
import time

from knowledge_base import DEFAULT_DB, V5, connect, file_hash, search, snapshot, verify, words
from science_agent import Facet, connect_server, review

PROTOCOL = V5 / "evaluation/protocol.json"
V4 = V5.parent / "extension_v4"


def visible_v4(db, pmid):
    row = db.execute("SELECT title, abstract FROM documents WHERE pmid=?", (pmid,)).fetchone()
    return (row[0] + "\n" + row[1])[:1200] if row else ""


def has_anchor(text, anchors):
    return any(anchor.casefold() in text.casefold() for anchor in anchors)


def verify_corpus(db):
    start = time.perf_counter()
    bad = []
    total = 0
    paper_counts = Counter()
    last_pmid, last_end = None, 0
    for row in db.execute(
        """SELECT d.pmid, d.abstract, p.start, p.end, p.text
        FROM papers d LEFT JOIN passages p USING (pmid) ORDER BY d.pmid, p.start"""
    ):
        pmid, abstract, lower, upper, passage = row
        total += 1
        paper_counts[pmid] += 1
        if pmid != last_pmid:
            last_end = 0
        if (lower is None or not 0 < upper - lower <= 500 or
            abstract[lower:upper] != passage or abstract[last_end:lower].strip()):
            bad.append((pmid, lower, "invalid_span_or_gap"))
        if last_pmid == pmid and lower is not None and lower < last_end:
            bad.append((pmid, lower, "overlap"))
        last_pmid, last_end = pmid, upper or 0
    for pmid, length, tail in db.execute(
        """SELECT d.pmid, length(d.abstract), max(p.end)
        FROM papers d LEFT JOIN passages p USING (pmid) GROUP BY d.pmid"""
    ):
        if length != tail:
            bad.append((pmid, tail, "missing_tail"))
    return dict(papers=len(paper_counts), passages=total, failures=len(bad),
                examples=bad[:10], seconds=round(time.perf_counter() - start, 4))


def late_passage_probe(v4_db, db, sample_size=256):
    """Known-source structural probe: derive searchable anchors from real late passages."""
    pool = list(db.execute(
        """SELECT p.pmid, p.start, p.end, p.text FROM passages p
        WHERE p.start >= 1200 AND p.end = (
            SELECT max(q.end) FROM passages q WHERE q.pmid=p.pmid
        )"""
    ))
    pool.sort(key=lambda row: hashlib.sha256(
        f"v5-late-passage-v1:{row['pmid']}".encode()
    ).digest())
    outcomes = []
    for row in pool:
        old = visible_v4(v4_db, row["pmid"])
        if not old:
            continue
        previous = set(words(old))
        anchors = [
            token for token in dict.fromkeys(words(row["text"]))
            if len(token) >= 5 and token.isalpha() and token not in previous
        ][:3]
        if len(anchors) != 3:
            continue
        found = search(db, " ".join(anchors), limit=3, pmid=row["pmid"], require_all=True)
        selected = next((r for r in found["candidates"] if r["start"] == row["start"]), None)
        checked = bool(selected and verify(
            db, selected["pmid"], selected["version"], selected["start"],
            selected["end"], selected["quote"], found["snapshot"]
        )["valid"])
        outcomes.append(dict(pmid=row["pmid"], anchors=anchors, start=row["start"],
                             v4_visible_has_all=False, v5_exact_passage_verified=checked))
        if len(outcomes) == sample_size:
            break
    if len(outcomes) != sample_size:
        raise ValueError("Not enough long, shared abstracts for the fixed probe")
    return dict(total=len(outcomes), v4_visible_has_all=0,
                v5_exact_passage_verified=sum(row["v5_exact_passage_verified"] for row in outcomes),
                cases=outcomes)


async def protocol_checks(db_path, v4_db):
    cases = json.loads(PROTOCOL.read_text(encoding="utf-8"))["checks"]
    rows = []
    latencies = []
    async with connect_server(db_path) as session:
        for offset in range(0, len(cases), 3):
            group = cases[offset:offset + 3]
            facets = [
                Facet.model_validate({k: v for k, v in item.items() if k in Facet.model_fields})
                for item in group
            ]
            result = await review(session, facets)
            latencies.extend(call["milliseconds"] for call in result["trace"])
            for case, facet in zip(group, result["facets"], strict=True):
                row = dict(id=case["id"], status=facet["status"], candidates=len(facet["evidence"]))
                if "expected_status" in case:
                    row["expected_status"] = case["expected_status"]
                    row["pass"] = row["status"] == row["expected_status"]
                else:
                    target, anchors = case["expected_pmid"], case["anchors"]
                    row["only_target_pmid"] = bool(facet["evidence"]) and all(
                        e["pmid"] == target for e in facet["evidence"]
                    )
                    row["v4_first_1200_anchor"] = has_anchor(visible_v4(v4_db, target), anchors)
                    row["v5_top_3_anchor"] = any(has_anchor(e["quote"], anchors) for e in facet["evidence"])
                    row["pass"] = row["only_target_pmid"] and row["v5_top_3_anchor"]
                rows.append(row)

        # Challenge the verifier via the actual server using a real returned passage.
        response = await session.call_tool("search_evidence", {
            "query": "statin 3449", "pmid": "32882643", "limit": 1,
        })
        found = response.structured_content
        candidate = found["candidates"][0]
        args = {k: candidate[k] for k in ("pmid", "version", "start", "end", "quote")}
        args["expected_snapshot"] = found["snapshot"]
        guards = {}
        for key, changed, expected in [
            ("quote", "fabricated quotation", "quote_mismatch"),
            ("version", "0" * 64, "source_version_mismatch"),
            ("expected_snapshot", "0" * 64, "snapshot_changed"),
            ("end", candidate["start"], "invalid_offsets"),
        ]:
            reply = await session.call_tool("verify_passage", {**args, key: changed})
            guards[expected] = reply.is_error or reply.structured_content["reason"] == expected
        guards["arbitrary_write_tool_rejected"] = (
            await session.call_tool("execute_sql", {"sql": "DELETE FROM papers"})
        ).is_error
    known = [row for row in rows if "v4_first_1200_anchor" in row]
    return dict(protocol_sha256=file_hash(PROTOCOL), cases=rows,
                known_item_total=len(known), v4_first_1200_anchor=sum(r["v4_first_1200_anchor"] for r in known),
                v5_top_3_anchor=sum(r["v5_top_3_anchor"] for r in known),
                source_scope_correct=sum(r["only_target_pmid"] for r in known), guards=guards,
                mcp_calls=len(latencies), mcp_median_ms=median(latencies))


def retrieval_baseline(v4_db, v5_db):
    # These are the original V4 known-item search queries, with no PMID/DOI passed to either method.
    groups = json.loads((V4 / "evaluation/queries.json").read_text(encoding="utf-8"))
    rows = []
    for case in groups["queries"]:
        query = case["query"]
        target = case["target_pmid"]
        terms = list(dict.fromkeys(re.findall(r"[^\W_]+", query.lower())))[:12]
        expression = " OR ".join(f'"{term}"' for term in terms)
        old = v4_db.execute(
            """SELECT d.pmid FROM documents_fts JOIN documents d ON documents_fts.rowid=d.rowid
            WHERE documents_fts MATCH ? ORDER BY bm25(documents_fts,3.0,1.0), d.pmid LIMIT 5""",
            (expression,),
        ).fetchall() if expression else []
        passage = search(v5_db, query, limit=5, method="passage")["candidates"]
        document = search(v5_db, query, limit=5, method="document")["candidates"]
        rows.append(dict(split=case["split"], kind=case["kind"], id=case["id"], target=target,
                         v4_top5=[r[0] for r in old], passage_top5=[r["pmid"] for r in passage],
                         document_top5=[r["pmid"] for r in document]))
    return dict(total=len(rows), v4_hit_at_5=sum(r["target"] in r["v4_top5"] for r in rows),
                v5_passage_only_hit_at_5=sum(r["target"] in r["passage_top5"] for r in rows),
                v5_document_first_hit_at_5=sum(r["target"] in r["document_top5"] for r in rows), rows=rows)


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--v4-db", type=Path, default=V4 / "data/corpus.sqlite3")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with closing(sqlite3.connect(args.v4_db.resolve().as_uri() + "?mode=ro", uri=True)) as v4_db, closing(connect(args.db)) as v5_db:
        identity = snapshot(v5_db)
        corpus = verify_corpus(v5_db)
        protocol = await protocol_checks(args.db, v4_db)
        retrieval = retrieval_baseline(v4_db, v5_db)
        late = late_passage_probe(v4_db, v5_db)
    result = dict(snapshot=identity, corpus=corpus, protocol=protocol,
                  unscoped_retrieval=retrieval, late_passage_probe=late)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(dict(corpus=corpus, protocol=protocol, unscoped_retrieval={
        k: v for k, v in retrieval.items() if k != "rows"
    }, late_passage_probe={k: v for k, v in late.items() if k != "cases"}), indent=2, ensure_ascii=False))
    if corpus["failures"] or not all(protocol["guards"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
