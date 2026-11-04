"""Streaming, idempotent LitCovid ingestion and parameterized FTS5 retrieval."""
import argparse
import csv
import hashlib
import json
import re
import sqlite3
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
 pmid TEXT PRIMARY KEY, title TEXT NOT NULL, abstract TEXT NOT NULL,
 doi TEXT NOT NULL, labels TEXT NOT NULL, content_hash TEXT NOT NULL UNIQUE
);
CREATE INDEX IF NOT EXISTS documents_doi ON documents(doi) WHERE doi <> '';
CREATE TABLE IF NOT EXISTS ingestion_runs (
 id INTEGER PRIMARY KEY, source TEXT NOT NULL, source_sha256 TEXT NOT NULL,
 rows_seen INTEGER NOT NULL, inserted INTEGER NOT NULL, skipped INTEGER NOT NULL,
 quarantined INTEGER NOT NULL, seconds REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS quarantine (
 source TEXT NOT NULL, row_number INTEGER NOT NULL, reason TEXT NOT NULL,
 PRIMARY KEY(source, row_number)
);
CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
 title, abstract, content='documents', content_rowid='rowid'
);
CREATE TRIGGER IF NOT EXISTS documents_ai AFTER INSERT ON documents BEGIN
 INSERT INTO documents_fts(rowid,title,abstract) VALUES(new.rowid,new.title,new.abstract);
END;
"""

def connect(path):
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA journal_mode=WAL')
    db.executescript(SCHEMA)
    return db


def ingest(db, source, batch_size=500):
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    start = time.perf_counter()
    source = Path(source)
    digest = hashlib.sha256()
    with source.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    counts = dict(rows_seen=0, inserted=0, skipped=0, quarantined=0)
    # Earlier committed batches survive interruption; replay is safe by keys.
    with source.open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        required = {'pmid', 'title', 'abstract', 'doi', 'label'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError('missing required CSV columns')
        for number, row in enumerate(reader, 2):
            counts['rows_seen'] += 1
            pmid = row['pmid'].strip()
            title, abstract = row['title'].strip(), row['abstract'].strip()
            if not pmid.isdigit() or not title or not abstract:
                db.execute('INSERT OR REPLACE INTO quarantine VALUES(?,?,?)',
                           (str(source), number, 'invalid PMID or empty title/abstract'))
                counts['quarantined'] += 1
            else:
                content_hash = hashlib.sha256(' '.join((title+' '+abstract).lower().split()).encode()).hexdigest()
                cursor = db.execute('INSERT OR IGNORE INTO documents VALUES(?,?,?,?,?,?)',
                                    (pmid, title, abstract, row['doi'].strip().lower(), row['label'], content_hash))
                counts['inserted' if cursor.rowcount else 'skipped'] += 1
            if counts['rows_seen'] % batch_size == 0:
                db.commit()
    elapsed = time.perf_counter() - start
    db.execute('INSERT INTO ingestion_runs(source,source_sha256,rows_seen,inserted,skipped,quarantined,seconds) VALUES(?,?,?,?,?,?,?)',
               (str(source), digest.hexdigest(), *counts.values(), elapsed))
    db.commit()
    return dict(source=source.name, source_sha256=digest.hexdigest(), **counts, seconds=elapsed)


def search(db, query, limit=5):
    if not 1 <= limit <= 50:
        raise ValueError('limit must be 1..50')
    # Quote individual lexical tokens: user text cannot inject FTS operators or SQL.
    tokens = re.findall(r'[^\W_]+', query, flags=re.UNICODE)[:32]
    if not tokens:
        return []
    expression = ' OR '.join('"' + token + '"' for token in tokens)
    rows = db.execute('''SELECT d.pmid,d.title,d.abstract,d.doi,bm25(documents_fts,3.0,1.0) AS score
      FROM documents_fts JOIN documents d ON d.rowid=documents_fts.rowid
      WHERE documents_fts MATCH ? ORDER BY score,d.pmid LIMIT ?''', (expression, limit))
    return [dict(row) for row in rows]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', required=True)
    parser.add_argument('--source', nargs='+', required=True)
    parser.add_argument('--report', required=True)
    args = parser.parse_args()
    Path(args.db).parent.mkdir(parents=True, exist_ok=True)
    db = connect(args.db)
    first = [ingest(db, path) for path in args.source]
    replay = [ingest(db, path) for path in args.source]
    report = dict(first_pass=first, replay=replay,
                  unique_documents=db.execute('SELECT COUNT(*) FROM documents').fetchone()[0],
                  fts_documents=db.execute('SELECT COUNT(*) FROM documents_fts').fetchone()[0],
                  pmid_query_plan=[tuple(r) for r in db.execute('EXPLAIN QUERY PLAN SELECT title FROM documents WHERE pmid=?', ('32519164',))],
                  search_examples={q:[{'pmid':r['pmid'],'title':r['title']} for r in search(db,q)]
                                   for q in ['tissue factor thrombosis', 'vaccine antibody response', 'PCR diagnosis']})
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in {'search_examples','pmid_query_plan'}}, indent=2))
    db.close()

if __name__ == '__main__':
    main()
