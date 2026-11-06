"""Check sampled citations against original CSV records outside the search pipeline."""

import argparse
from contextlib import closing
import csv
import hashlib
import json
from pathlib import Path
import re
import unicodedata

from knowledge_base import DEFAULT_DB, connect, file_hash, snapshot

SEED = "science-source-trace-audit-v1"


def canonical_text(raw):
    # An independent normalization expression, not the indexer's helper.
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", raw)).strip()


def audit(db_path, paths, per_source):
    files = {Path(path).name: Path(path) for path in paths}
    if len(files) != len(paths):
        raise ValueError("Source filenames must be distinct")
    with closing(connect(db_path)) as db:
        identity = snapshot(db)
        hashes = dict(db.execute("SELECT name, sha256 FROM sources"))
        if files.keys() != hashes.keys():
            raise ValueError("Provide the same source files used for the snapshot")
        summaries = []
        for name, digest in hashes.items():
            path = files[name]
            if file_hash(path) != digest:
                raise ValueError(f"Source hash mismatch: {name}")
            rows = list(db.execute(
                "SELECT pmid, title, source_record FROM papers WHERE source=?", (name,)
            ))
            rows.sort(key=lambda row: hashlib.sha256(
                f"{SEED}:{row['pmid']}".encode()
            ).digest())
            sample = {row["source_record"]: row for row in rows[:per_source]}
            if len(sample) != per_source:
                raise ValueError(f"Not enough retained records in {name}")
            issues = []
            passages = 0
            with path.open(encoding="utf-8-sig", newline="") as stream:
                for number, raw in enumerate(csv.DictReader(stream), 2):
                    if number not in sample:
                        continue
                    selected = sample.pop(number)
                    pmid = selected["pmid"]
                    if canonical_text(raw["pmid"]) != pmid or canonical_text(raw["title"]) != selected["title"]:
                        issues.append({"pmid": pmid, "record": number, "reason": "source_record_mismatch"})
                    abstract = canonical_text(raw["abstract"])
                    for start, end, text in db.execute(
                        "SELECT start,end,text FROM passages WHERE pmid=? ORDER BY start", (pmid,)
                    ):
                        passages += 1
                        if abstract[start:end] != text:
                            issues.append({"pmid": pmid, "record": number, "reason": "source_span_mismatch"})
            issues.extend({"record": number, "reason": "record_missing"} for number in sample)
            summaries.append(dict(source=name, sha256=digest, sampled_records=per_source,
                                  verified_passages=passages, failures=issues))
    return dict(snapshot=identity, selection=SEED, sources=summaries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--source", type=Path, nargs="+", required=True)
    parser.add_argument("--per-source", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.per_source <= 1000:
        parser.error("Sample size must be 1–1000")
    result = audit(args.db, args.source, args.per_source)
    output = json.dumps(result, indent=2) + "\n"
    args.output.write_text(output, encoding="utf-8")
    print(output, end="")
    if any(row["failures"] for row in result["sources"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
