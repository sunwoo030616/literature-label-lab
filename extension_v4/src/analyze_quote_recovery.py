"""Post-hoc repair diagnostic; initial strict outputs remain untouched."""

import json
import sqlite3
from settings import V4, DB, dump
from evidence import visible_document
from quote_recovery import recover_edge_ellipsis


def main():
    rows = json.loads((V4 / "results/llm_rows.json").read_text())
    db = sqlite3.connect(DB.resolve().as_uri() + "?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    corpus = {
        r["pmid"]: visible_document(dict(r))
        for r in db.execute("SELECT pmid,title,abstract FROM documents")
    }
    recovered = []
    for row in rows:
        docs = {p: corpus[p] for p in row["retrieved_pmids"]}
        result = recover_edge_ellipsis(row["raw_output"], docs)
        recovered.append(
            dict(
                task_id=row["task_id"],
                condition=row["condition"],
                before_excerpts=len(row["verified_excerpts"]),
                after_excerpts=len(result["verified_excerpts"]),
                recovered=result["recovered_excerpts"],
                target_cited_after=any(
                    e["pmid"] == row["target_pmid"] for e in result["verified_excerpts"]
                ),
            )
        )
    dump(
        V4 / "results/quote_recovery.json",
        dict(
            scope="Post-hoc deterministic diagnostic on original saved outputs, not a new blinded test; original strict outputs unchanged",
            runs=recovered,
            new_exact_excerpts=sum(len(r["recovered"]) for r in recovered),
        ),
    )
    print("Recovered exact excerpts:", sum(len(r["recovered"]) for r in recovered))
    db.close()


if __name__ == "__main__":
    main()
