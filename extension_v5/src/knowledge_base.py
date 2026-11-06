"""Build and query an auditable snapshot of LitCovid title/abstract evidence."""

import argparse
from collections import Counter
from contextlib import closing
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import tempfile
import time
import unicodedata

V5 = Path(__file__).resolve().parents[1]
DEFAULT_DB = V5 / "data/science.sqlite3"
NORMALIZATION = "nfc-space-v1"
INDEX_VERSION = "document-passage-kb-v3"
SCHEMA = """
CREATE TABLE sources (
    name TEXT PRIMARY KEY, sha256 TEXT NOT NULL
);
CREATE TABLE papers (
    pmid TEXT PRIMARY KEY, doi TEXT NOT NULL, title TEXT NOT NULL,
    abstract TEXT NOT NULL, journal TEXT NOT NULL, publication_type TEXT NOT NULL,
    labels TEXT NOT NULL, version TEXT NOT NULL, content_key TEXT NOT NULL,
    source TEXT NOT NULL REFERENCES sources(name), source_record INTEGER NOT NULL
);
CREATE INDEX papers_content_key ON papers(content_key);
CREATE INDEX papers_doi ON papers(doi) WHERE doi <> '';
CREATE TABLE passages (
    id INTEGER PRIMARY KEY, pmid TEXT NOT NULL REFERENCES papers(pmid),
    start INTEGER NOT NULL, end INTEGER NOT NULL, text TEXT NOT NULL,
    UNIQUE(pmid, start, end)
);
CREATE VIRTUAL TABLE paper_fts USING fts5(title, abstract, content='papers', content_rowid='rowid');
CREATE VIRTUAL TABLE passage_fts USING fts5(
    text, content='passages', content_rowid='id', tokenize='unicode61 remove_diacritics 2'
);
CREATE TABLE audit (
    source TEXT NOT NULL REFERENCES sources(name), source_record INTEGER NOT NULL,
    pmid TEXT, reason TEXT NOT NULL, retained_pmid TEXT REFERENCES papers(pmid),
    PRIMARY KEY (source, source_record, reason)
);
CREATE TABLE metadata (name TEXT PRIMARY KEY, value TEXT NOT NULL);
"""
PMID = re.compile(r"[1-9][0-9]{0,11}")
DOI = re.compile(r"10\.[0-9]{4,9}/[^\s]+")
SENTENCE_BREAK = re.compile(r"(?<=[.!?]) (?=[A-Z0-9])")
WORD = re.compile(r"[^\W_]+", re.UNICODE)


def sha256(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def normalize(value):
    return " ".join(unicodedata.normalize("NFC", value or "").split())


def normalize_doi(value):
    value = normalize(value).lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if value.startswith(prefix):
            value = value.removeprefix(prefix).strip()
            break
    return value if DOI.fullmatch(value) else ""


def passage_spans(text, limit=500):
    """Pack sentence boundaries where possible; split overlong sentences at spaces."""
    boundaries = [0, *(m.end() for m in SENTENCE_BREAK.finditer(text)), len(text)]
    start = end = 0
    for boundary in boundaries[1:]:
        if boundary - start <= limit:
            end = boundary
            continue
        if end > start:
            yield start, end - (text[end - 1] == " ")
            start = end
        while boundary - start > limit:
            cut = text.rfind(" ", start + 1, start + limit + 1)
            if cut == -1:
                cut = start + limit
            yield start, cut
            start = cut + (text[cut] == " ")
        end = boundary
    if end > start:
        yield start, end - (text[end - 1] == " ")


def import_source(db, path, digest, manifest_digest):
    db.execute("INSERT INTO sources VALUES (?, ?)", (path.name, digest))
    counts = Counter()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"pmid", "title", "abstract", "doi", "journal", "pub_type", "label"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path.name}: missing LitCovid columns")
        for record, row in enumerate(reader, 2):
            counts["rows"] += 1
            pmid = normalize(row["pmid"])
            title = normalize(row["title"])
            abstract = normalize(row["abstract"])
            doi = normalize_doi(row["doi"])

            def audit(reason, retained=None):
                counts[reason] += 1
                db.execute(
                    "INSERT INTO audit VALUES (?, ?, ?, ?, ?)",
                    (path.name, record, pmid if PMID.fullmatch(pmid) else None, reason, retained),
                )

            if not PMID.fullmatch(pmid) or not title or not abstract:
                audit("invalid_pmid" if not PMID.fullmatch(pmid) else "missing_title_or_abstract")
                continue
            if normalize(row["doi"]) and not doi:
                audit("invalid_doi")
            fields = (doi, title, abstract, normalize(row["journal"]),
                      normalize(row["pub_type"]), normalize(row["label"]))
            version = sha256(json.dumps([NORMALIZATION, *fields], ensure_ascii=False))
            content_key = sha256(json.dumps([title.casefold(), abstract.casefold()], ensure_ascii=False))
            prior = db.execute(
                "SELECT version FROM papers WHERE pmid = ?", (pmid,)
            ).fetchone()
            if prior:
                audit("duplicate_pmid" if prior[0] == version else "conflicting_pmid", pmid)
                continue
            prior = db.execute("SELECT pmid FROM papers WHERE content_key = ? ORDER BY source, source_record LIMIT 1", (content_key,)).fetchone()
            if prior:
                # Different stable IDs are evidence of distinct source records even if their text matches.
                audit("shared_content", prior[0])
            cursor = db.execute(
                "INSERT INTO papers VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (pmid, *fields, version, content_key, path.name, record),
            )
            db.execute("INSERT INTO paper_fts(rowid,title,abstract) VALUES (?, ?, ?)",
                       (cursor.lastrowid, title, abstract))
            counts["papers"] += 1
            for start, end in passage_spans(abstract):
                text = abstract[start:end]
                cursor = db.execute(
                    "INSERT INTO passages(pmid,start,end,text) VALUES (?, ?, ?, ?)",
                    (pmid, start, end, text),
                )
                db.execute("INSERT INTO passage_fts(rowid,text) VALUES (?, ?)", (cursor.lastrowid, text))
                counts["passages"] += 1
            manifest_digest.update(f"{pmid}\t{version}\n".encode())
    if file_hash(path) != digest:
        raise ValueError(f"{path.name}: source changed during ingestion")
    return {"source": path.name, "sha256": digest, **counts}


def build(destination, paths):
    started = time.perf_counter()
    destination = Path(destination).resolve()
    paths = [Path(p).resolve() for p in paths]
    if not paths or len({p.name for p in paths}) != len(paths):
        raise ValueError("Provide source files with distinct names")
    if destination in paths:
        raise ValueError("The database destination must differ from the source files")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temp:
        staging = Path(temp) / "build.sqlite3"
        manifest = hashlib.sha256()
        manifest.update(f"{INDEX_VERSION}\t{NORMALIZATION}\n".encode())
        with closing(sqlite3.connect(staging)) as db, db:
            db.execute("PRAGMA foreign_keys=ON")
            db.executescript(SCHEMA)
            sources = []
            for path in paths:
                digest = file_hash(path)
                manifest.update(f"{path.name}\t{digest}\n".encode())
                sources.append(import_source(db, path, digest, manifest))
            identity = manifest.hexdigest()
            db.executemany(
                "INSERT INTO metadata VALUES (?, ?)",
                [("snapshot", identity), ("normalization", NORMALIZATION)],
            )
            ambiguous_dois = db.execute(
                "SELECT count(*) FROM (SELECT doi FROM papers WHERE doi <> '' GROUP BY doi HAVING count(*) > 1)"
            ).fetchone()[0]
            paper_count = db.execute("SELECT count(*) FROM papers").fetchone()[0]
            passage_count = db.execute("SELECT count(*) FROM passages").fetchone()[0]
            if not paper_count or not passage_count:
                raise ValueError("No searchable documents in the supplied sources")
            if db.execute("PRAGMA foreign_key_check").fetchone():
                raise ValueError("Invalid reference in staged snapshot")
            db.execute("INSERT INTO passage_fts(passage_fts,rank) VALUES ('integrity-check',1)")
            db.execute("INSERT INTO paper_fts(paper_fts,rank) VALUES ('integrity-check',1)")
        os.replace(staging, destination)
    return dict(snapshot=identity, normalization=NORMALIZATION, index_version=INDEX_VERSION, sources=sources,
                papers=paper_count, passages=passage_count, ambiguous_dois=ambiguous_dois,
                database_bytes=destination.stat().st_size,
                seconds=round(time.perf_counter() - started, 4))


def connect(path):
    db = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    return db


def snapshot(db):
    return db.execute("SELECT value FROM metadata WHERE name = 'snapshot'").fetchone()[0]


def resolve(db, pmid="", doi=""):
    if pmid and doi:
        raise ValueError("Specify a PMID or DOI")
    if pmid:
        if not PMID.fullmatch(pmid):
            raise ValueError("Invalid PMID")
        rows = db.execute("SELECT pmid FROM papers WHERE pmid = ?", (pmid,)).fetchall()
    elif doi:
        canonical = normalize_doi(doi)
        if not canonical:
            raise ValueError("Invalid DOI")
        rows = db.execute("SELECT pmid FROM papers WHERE doi = ? ORDER BY pmid LIMIT 2", (canonical,)).fetchall()
    else:
        return "unscoped", ""
    if not rows:
        return "source_not_found", ""
    if len(rows) > 1:
        return "ambiguous_doi", ""
    return "resolved", rows[0][0]


def words(text):
    folded = unicodedata.normalize("NFD", text.lower())
    plain = "".join(c for c in folded if unicodedata.category(c) != "Mn")
    return WORD.findall(plain)


def paper_passages(db, pmid, terms, require_all, limit, allow_title_match=False):
    rows = db.execute(
        """SELECT p.*, d.version, d.title, d.doi, d.journal, d.publication_type
        FROM passages p JOIN papers d USING (pmid) WHERE p.pmid = ? ORDER BY p.start""",
        (pmid,),
    )
    scored = []
    for row in rows:
        overlap = len(set(words(row["text"])).intersection(terms))
        if require_all and overlap != len(terms):
            continue
        if overlap or allow_title_match:
            scored.append((-overlap, row["start"], row))
    return [(row, "passage_terms" if score else "document_context") for score, _, row in sorted(scored)[:limit]]


def search(db, query, limit=5, pmid="", doi="", require_all=False, method="document"):
    if not 1 <= limit <= 10 or not 1 <= len(query) <= 250:
        raise ValueError("Invalid search budget")
    if method not in {"document", "passage"}:
        raise ValueError("Unknown search method")
    identity = snapshot(db)
    scope, pmid = resolve(db, pmid, doi)
    if scope not in {"unscoped", "resolved"}:
        return dict(snapshot=identity, status=scope, candidates=[])
    terms = list(dict.fromkeys(words(query)))[:12]
    if not terms:
        return dict(snapshot=identity, status="no_search_terms", candidates=[])

    operator = " AND " if require_all else " OR "
    expression = operator.join(f'"{term}"' for term in terms)
    if scope == "resolved":
        matches = paper_passages(db, pmid, terms, require_all, limit)
    elif method == "document":
        # Rank a bounded set of papers first, then select one passage per paper.
        # This preserves title cues and prevents one long abstract from filling the result list.
        candidates = db.execute(
            """SELECT d.pmid FROM paper_fts JOIN papers d ON d.rowid = paper_fts.rowid
            WHERE paper_fts MATCH ? ORDER BY bm25(paper_fts, 3.0, 1.0), d.pmid LIMIT ?""",
            (expression, limit * 3 if require_all else limit),
        ).fetchall()
        matches = []
        for candidate in candidates:
            matches.extend(paper_passages(
                db, candidate["pmid"], terms, require_all, limit=1,
                allow_title_match=not require_all,
            ))
            if len(matches) == limit:
                break
    else:
        rows = db.execute(
            """SELECT p.*, d.version, d.title, d.doi, d.journal, d.publication_type
            FROM passage_fts JOIN passages p ON passage_fts.rowid = p.id
            JOIN papers d USING (pmid) WHERE passage_fts MATCH ?
            ORDER BY bm25(passage_fts), p.id LIMIT ?""",
            (operator.join(f'text : "{term}"' for term in terms), limit),
        )
        matches = [(row, "passage_terms") for row in rows]
    candidates = [
        dict(pmid=row["pmid"], doi=row["doi"], title=row["title"], journal=row["journal"],
             publication_type=row["publication_type"], version=row["version"],
             start=row["start"], end=row["end"], quote=row["text"], match=match,
             url=f"https://pubmed.ncbi.nlm.nih.gov/{row['pmid']}/")
        for row, match in matches
    ]
    return dict(snapshot=identity, status="candidates_found" if candidates else "no_matching_passage",
                candidates=candidates)


def verify(db, pmid, version, start, end, quote, expected_snapshot):
    if snapshot(db) != expected_snapshot:
        return {"valid": False, "reason": "snapshot_changed"}
    row = db.execute("SELECT version, abstract FROM papers WHERE pmid = ?", (pmid,)).fetchone()
    if row is None or row["version"] != version:
        return {"valid": False, "reason": "source_version_mismatch"}
    if not 0 <= start < end <= len(row["abstract"]) or end - start > 500:
        return {"valid": False, "reason": "invalid_offsets"}
    if row["abstract"][start:end] != quote:
        return {"valid": False, "reason": "quote_mismatch"}
    source = db.execute(
        "SELECT p.source, p.source_record, s.sha256 FROM papers p JOIN sources s ON p.source=s.name WHERE p.pmid=?",
        (pmid,),
    ).fetchone()
    return {"valid": True, "reason": "exact_normalized_abstract_span", "provenance": dict(source)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--source", type=Path, nargs="+", required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = json.dumps(build(args.db, args.source), ensure_ascii=False, indent=2) + "\n"
    if args.report:
        args.report.write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
