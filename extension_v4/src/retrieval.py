"""Same-corpus lexical, dense and reciprocal rank fusion retrieval."""

import json
import re
import sqlite3
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from settings import DB, V4, dump
from corpus import search as raw_bm25

METHODS = ["bm25_raw", "bm25", "tfidf", "dense", "hybrid"]


class Retriever:
    def __init__(self, db_path=DB, load_dense=True):
        # MCP handlers can run on different worker threads; the server serializes calls.
        self.db = sqlite3.connect(
            Path(db_path).resolve().as_uri() + "?mode=ro",
            uri=True,
            check_same_thread=False,
        )
        self.db.row_factory = sqlite3.Row
        self.rows = [
            dict(r)
            for r in self.db.execute(
                "SELECT pmid,title,abstract,doi,content_hash FROM documents ORDER BY pmid"
            )
        ]
        self.by_id = {r["pmid"]: r for r in self.rows}
        self.encoder = None
        self.dense = None
        if load_dense:
            manifest = json.loads((V4 / "data/dense_manifest.json").read_text())
            if [(r["pmid"], r["content_hash"]) for r in self.rows] != [
                (r["pmid"], r["content_hash"]) for r in manifest
            ]:
                raise ValueError(
                    "Dense index and corpus IDs/content do not match; rebuild index"
                )
            self.dense = np.load(V4 / "data/minilm.npy", mmap_mode="r")
        self.vectorizer = None
        self.sparse = None

    def close(self):
        self.db.close()

    def fit_tfidf(self):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=100000
        )
        self.sparse = self.vectorizer.fit_transform(
            [r["title"] + " " + r["abstract"] for r in self.rows]
        )

    def encode_query(self, query):
        if self.encoder is None:
            from dense_index import Encoder

            self.encoder = Encoder()
        return self.encoder.encode([query])[0]

    def search(
        self, query, method="bm25", limit=5, vector=None, required_numeric_anchors=None
    ):
        if method not in METHODS:
            raise ValueError("unknown retrieval method")
        if not isinstance(query, str) or not 1 <= len(query) <= 1000:
            raise ValueError("query length must be 1..1000")
        if not 1 <= limit <= 50:
            raise ValueError("limit must be 1..50")
        if required_numeric_anchors and method not in {"bm25", "bm25_raw"}:
            raise ValueError("Numeric source filters require lexical retrieval")
        if method in {"bm25", "bm25_raw"}:
            if method == "bm25":
                query = " ".join(
                    t
                    for t in re.findall(r"[^\W_]+", query)
                    if t.lower() not in ENGLISH_STOP_WORDS
                )
            return raw_bm25(self.db, query, limit, required_numeric_anchors)
        if method == "hybrid":
            lexical = self.search(query, "bm25", 50)
            dense = self.search(query, "dense", 50, vector)
            scores = {}
            for items in [lexical, dense]:
                for rank, row in enumerate(items, 1):
                    scores[row["pmid"]] = scores.get(row["pmid"], 0.0) + 1.0 / (
                        60 + rank
                    )
            ids = sorted(scores, key=lambda pmid: (-scores[pmid], pmid))[:limit]
            return [dict(self.by_id[i], score=scores[i]) for i in ids]
        if method == "dense":
            if self.dense is None:
                raise ValueError("dense index not loaded")
            scores = np.asarray(
                self.dense @ (self.encode_query(query) if vector is None else vector)
            )
        else:
            if self.vectorizer is None:
                self.fit_tfidf()
            scores = (
                (self.sparse @ self.vectorizer.transform([query]).T).toarray().ravel()
            )
        order = np.lexsort((np.arange(len(self.rows)), -scores))[:limit]
        return [dict(self.rows[int(i)], score=float(scores[i])) for i in order]

    def get(self, pmid):
        if not isinstance(pmid, str) or not re.fullmatch(r"[0-9]{1,12}", pmid):
            raise ValueError("invalid PMID")
        row = self.db.execute(
            "SELECT pmid,title,abstract,doi,content_hash FROM documents WHERE pmid=?",
            (pmid,),
        ).fetchone()
        return dict(row) if row else None
