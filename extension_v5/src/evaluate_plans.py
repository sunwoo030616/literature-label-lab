"""Run an offline, resumable paired Qwen planner comparison."""

import argparse
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
import time

from knowledge_base import DEFAULT_DB, V5, connect, file_hash, snapshot
from science_agent import explicit_scope, local_model, Topics

VERSIONS = ("science-facets-v1", "science-facets-v2")
SEED = "literature-v5-paired-planner-nov06-v1"


def build_cases(db_path, sample):
    with closing(connect(db_path)) as db:
        identity = snapshot(db)
        available = list(db.execute(
            """SELECT pmid, title FROM papers
            WHERE length(title) BETWEEN 30 AND 180 AND length(abstract) > 1200
            AND pmid NOT IN (SELECT pmid FROM audit WHERE pmid IS NOT NULL)"""
        ))
    chosen = sorted(available, key=lambda row: hashlib.sha256(
        f"{SEED}:{row['pmid']}".encode()
    ).digest())[:sample]
    known_path = V5.parent / "extension_v4/evaluation/queries.json"
    known = json.loads(known_path.read_text(encoding="utf-8"))["queries"]
    cases = [dict(id=r["id"], question=r["query"], pmid=r["target_pmid"], kind="known_" + r["kind"],
                  source_scope=False, requested_facets=1) for r in known]
    for index, row in enumerate(chosen):
        title, pmid = row["title"], row["pmid"]
        variants = [
            ("english_title", f'Find the scientific paper titled "{title}" and locate its abstract.', 1),
            ("korean_title", f'"{title}" 논문의 초록에서 주요 연구 결과를 찾고 싶어요.', 1),
            ("english_scoped", f'For PMID {pmid}, "{title}", look for passages on study methodology and outcome findings separately.', 2),
            ("korean_scoped", f'PMID {pmid}에서 연구 설계와 결과 근거를 각각 찾아줘. 논문 제목은 "{title}"입니다.', 2),
        ]
        kind, question, requested_facets = variants[index % len(variants)]
        # Known source binding is only expected when the question actually supplies an identifier.
        cases.append(dict(id=f"sample-{pmid}", question=question, pmid=pmid, kind=kind,
                          source_scope=kind.endswith("scoped"), requested_facets=requested_facets))
    return identity, cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--sample", type=int, default=240)
    parser.add_argument("--output", type=Path, required=True, help="Resume from a SQLite checkpoint")
    args = parser.parse_args()
    if not 1 <= args.sample <= 1000:
        parser.error("Sample must be 1–1000")
    identity, cases = build_cases(args.db, args.sample)
    manifest = json.dumps(dict(snapshot=identity, seed=SEED, cases=cases,
                               prompts=file_hash(V5 / "prompts.json"), versions=VERSIONS),
                          ensure_ascii=False, sort_keys=True)
    fingerprint = hashlib.sha256(manifest.encode()).hexdigest()
    with closing(sqlite3.connect(args.output)) as results:
        results.executescript("""
        CREATE TABLE IF NOT EXISTS manifest (fingerprint TEXT PRIMARY KEY, content TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS plans (
            case_id TEXT, version TEXT, result TEXT, PRIMARY KEY(case_id, version)
        );
        """)
        prior = results.execute("SELECT fingerprint, content FROM manifest").fetchone()
        if prior and prior[0] != fingerprint:
            previous, requested = json.loads(prior[1]), json.loads(manifest)
            previous.pop("snapshot")
            requested.pop("snapshot")
            if previous != requested:
                raise ValueError("Checkpoint belongs to other questions or prompt versions")
        if not prior:
            results.execute("INSERT INTO manifest VALUES (?, ?)", (fingerprint, manifest))
            results.commit()
        completed = set(results.execute("SELECT case_id, version FROM plans"))
        if len(completed) == len(cases) * len(VERSIONS):
            print("All plan rows already available", flush=True)
            return
        model = local_model()
        identities = set(results.execute(
            "SELECT DISTINCT json_extract(result, '$.model'), json_extract(result, '$.revision') FROM plans"
        ))
        if identities and identities != {(model.identity["repo_id"], model.identity["revision"])}:
            raise ValueError("Checkpoint was created with a different local model revision")
        started = time.perf_counter()
        for number, case in enumerate(cases, 1):
            for version in VERSIONS:
                if (case["id"], version) in completed:
                    continue
                generated = model.generate(version, {"question": case["question"]})
                try:
                    scope = explicit_scope(case["question"])
                    topics = Topics.validate_json(generated["raw_output"])
                    generated["plan"] = [topic.model_dump() for topic in topics]
                    generated["scope"] = scope
                except ValueError as exc:
                    generated["error"] = type(exc).__name__
                results.execute("INSERT INTO plans VALUES (?, ?, ?)",
                                (case["id"], version, json.dumps(generated, ensure_ascii=False)))
                results.commit()
            if number % 10 == 0 or number == len(cases):
                print(f"{number}/{len(cases)} paired cases; elapsed {time.perf_counter() - started:.1f}s",
                      flush=True)


if __name__ == "__main__":
    main()
