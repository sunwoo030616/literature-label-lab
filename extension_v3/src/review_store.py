"""Local auditable document review state; no external transmission."""

import sqlite3, json, hashlib, time
from pathlib import Path


class ConflictError(Exception):
    pass


class ReviewStore:
    def __init__(self, path):
        self.path = str(path)
        with self.connect() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS documents (id INTEGER PRIMARY KEY, digest TEXT NOT NULL, model_version TEXT NOT NULL, title TEXT NOT NULL, abstract TEXT NOT NULL, prediction TEXT NOT NULL, review TEXT, revision INTEGER NOT NULL DEFAULT 0, created REAL NOT NULL, updated REAL NOT NULL, UNIQUE(digest,model_version))"
            )

    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        return db

    def register(self, title, abstract, model_version, prediction):
        digest = hashlib.sha256((title.strip() + "\n" + abstract.strip()).encode()).hexdigest()
        now = time.time()
        with self.connect() as db:
            db.execute(
                "INSERT OR IGNORE INTO documents(digest,model_version,title,abstract,prediction,created,updated) VALUES(?,?,?,?,?,?,?)",
                (
                    digest,
                    model_version,
                    title,
                    abstract,
                    json.dumps(prediction, ensure_ascii=False),
                    now,
                    now,
                ),
            )
            row = db.execute(
                "SELECT * FROM documents WHERE digest=? AND model_version=?",
                (digest, model_version),
            ).fetchone()
        return self.decode(row)

    def get(self, id):
        with self.connect() as db:
            row = db.execute("SELECT * FROM documents WHERE id=?", (id,)).fetchone()
        if row is None:
            raise KeyError(id)
        return self.decode(row)

    def review(self, id, expected_revision, labels, evidence, allowed_labels):
        if len(labels) != len(set(labels)) or not set(labels) <= set(allowed_labels):
            raise ValueError("unknown_or_duplicate_label")
        row = self.get(id)
        text = row["title"] + " " + row["abstract"]
        for e in evidence:
            if (
                e.get("label") not in labels
                or not isinstance(e.get("quote"), str)
                or len(e["quote"]) < 15
                or e["quote"] not in text
            ):
                raise ValueError("unverifiable_evidence")
        review = {
            "labels": labels,
            "evidence": evidence,
            "note": "Human supplied labels; quote presence is not semantic proof.",
        }
        with self.connect() as db:
            cur = db.execute(
                "UPDATE documents SET review=?,revision=revision+1,updated=? WHERE id=? AND revision=?",
                (json.dumps(review, ensure_ascii=False), time.time(), id, expected_revision),
            )
            if cur.rowcount != 1:
                raise ConflictError("stale_revision")
        return self.get(id)

    def append_llm_run(self, document_id, result):
        self.get(document_id)
        with self.connect() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS llm_runs (id INTEGER PRIMARY KEY, document_id INTEGER NOT NULL, result TEXT NOT NULL, created REAL NOT NULL)"
            )
            cur = db.execute(
                "INSERT INTO llm_runs(document_id,result,created) VALUES(?,?,?)",
                (document_id, json.dumps(result, ensure_ascii=False), time.time()),
            )
            run_id = cur.lastrowid
        return {"run_id": run_id, "document_id": document_id, "result": result}

    def llm_runs(self, document_id):
        self.get(document_id)
        with self.connect() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS llm_runs (id INTEGER PRIMARY KEY, document_id INTEGER NOT NULL, result TEXT NOT NULL, created REAL NOT NULL)"
            )
            rows = db.execute(
                "SELECT * FROM llm_runs WHERE document_id=? ORDER BY id", (document_id,)
            ).fetchall()
        return [
            {"run_id": r["id"], "created": r["created"], "result": json.loads(r["result"])}
            for r in rows
        ]

    @staticmethod
    def decode(row):
        obj = dict(row)
        obj["prediction"] = json.loads(obj["prediction"])
        obj["review"] = json.loads(obj["review"]) if obj["review"] else None
        obj["status"] = "reviewed" if obj["review"] else "awaiting_review"
        return obj
