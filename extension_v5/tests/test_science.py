import asyncio
from contextlib import closing
import csv

import pytest

from knowledge_base import build, connect, passage_spans, search, verify
from science_agent import Facet, connect_server, plan_question, review

HEADERS = ("pmid", "title", "abstract", "doi", "journal", "pub_type", "label")


def source(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(HEADERS)
        writer.writerows((*row, "Science", "Journal Article", "Treatment") for row in rows)
    return path


def test_snapshot_audit_and_failed_build_keep_existing_database(tmp_path):
    path = source(tmp_path / "litcovid.csv", [
        ("11", "Café study", "A long  controlled result.", "https://doi.org/10.1000/SHARED"),
        ("11", "Cafe\u0301 study", "A long controlled result.", "10.1000/shared"),
        ("11", "Different title", "Different evidence.", ""),
        ("12", "Café study", "A long controlled result.", "doi: 10.1000/shared"),
        ("13", "Broken", "", "10.1000/broken"),
        ("14", "Valid record", "Second source result.", "bad DOI"),
    ])
    destination = tmp_path / "science.sqlite3"
    first = build(destination, [path])
    assert first["papers"] == 3
    assert first["ambiguous_dois"] == 1
    assert first["snapshot"] == build(destination, [path])["snapshot"]
    malformed = tmp_path / "bad.csv"
    malformed.write_text("pmid,title\n14,missing columns\n")
    with pytest.raises(ValueError, match="missing LitCovid columns"):
        build(destination, [path, malformed])
    with closing(connect(destination)) as db:
        assert search(db, "controlled result", pmid="11")["snapshot"] == first["snapshot"]
        assert search(db, "controlled result", doi="10.1000/SHARED")["status"] == "ambiguous_doi"
        hit = search(db, "controlled result", pmid="11")["candidates"][0]
        check = verify(db, "11", hit["version"], hit["start"], hit["end"], hit["quote"], first["snapshot"])
        assert set(check["review_flags"]) == {
            "duplicate_pmid", "conflicting_pmid", "shared_content", "ambiguous_doi",
        }
        assert [tuple(row) for row in db.execute(
            "SELECT reason, retained_pmid FROM audit ORDER BY source_record"
        )] == [
            ("duplicate_pmid", "11"), ("conflicting_pmid", "11"), ("shared_content", "11"),
            ("missing_title_or_abstract", None), ("invalid_doi", None),
        ]


def test_passage_covers_long_normalized_abstract_and_verifies_exact_offset(tmp_path):
    abstract = "Background details. " * 140 + "Observed 417 patients in the final analysis."
    path = source(tmp_path / "litcovid.csv", [("11", "Cohort", abstract, "10.1000/cohort")])
    destination = tmp_path / "science.sqlite3"
    first = build(destination, [path])
    spans = list(passage_spans(abstract))
    assert " ".join(abstract[a:b] for a, b in spans) == abstract
    assert all(1 <= b - a <= 500 for a, b in spans)
    with closing(connect(destination)) as db:
        evidence = search(db, "417 patients", pmid="11", require_all=True)["candidates"][0]
        assert evidence["start"] > 1200
        args = [evidence[k] for k in ("pmid", "version", "start", "end", "quote")]
        check = verify(db, *args, first["snapshot"])
        assert check["valid"] and check["provenance"]["source_record"] == 2
        assert verify(db, *args[:-1], "fabricated quote", first["snapshot"])["reason"] == "quote_mismatch"
        assert search(db, "417 patients", pmid="99")["status"] == "source_not_found"
    source(path, [("11", "Cohort", "Observed 418 patients.", "10.1000/cohort")])
    build(destination, [path])
    with closing(connect(destination)) as db:
        assert verify(db, *args, first["snapshot"])["reason"] == "snapshot_changed"


def test_explicit_source_binding_and_model_plan_validation():
    class FakePlanner:
        raw = '[{"label":"Cohort","query":"hospital cohort size"}]'

        def generate(self, version, payload):
            return {"raw_output": self.raw}

    planner = FakePlanner()
    facets, _ = plan_question(planner, "PMID 123에서 코호트 수를 찾아줘")
    assert facets[0].pmid == "123"
    with pytest.raises(ValueError, match="multiple source"):
        plan_question(planner, "Compare PMID 123 and PMID 456")
    planner.raw = '[{"label":"Cohort","query":"hospital","pmid":"999"}]'
    with pytest.raises(ValueError):
        plan_question(planner, "PMID 123에서 코호트 수를 찾아줘")


def test_real_stdio_agent_keeps_available_and_missing_facets_separate(tmp_path):
    path = source(tmp_path / "litcovid.csv", [
        ("11", "Cohort report", "Observed 417 patients in the final analysis.", ""),
        ("12", "Other paper", "Observed 417 patients in the final analysis.", ""),
    ])
    destination = tmp_path / "science.sqlite3"
    build(destination, [path])

    async def run():
        async with connect_server(destination) as session:
            listed = await session.list_tools()
            assert {tool.name for tool in listed.tools} == {"search_evidence", "verify_passage"}
            response = await review(session, [
                Facet(label="Cohort", query="417 patients", pmid="11", require_all=True),
                Facet(label="Missing", query="417 patients", pmid="99"),
                Facet(label="Strict", query="417 astronaut", pmid="11", require_all=True),
            ])
            assert response["status"] == "partial"
            assert [item["status"] for item in response["facets"]] == [
                "verified_candidates", "source_not_found", "no_matching_passage",
            ]
            assert [item["pmid"] for item in response["facets"][0]["evidence"]] == ["11"]

    asyncio.run(run())


def test_document_first_search_keeps_distinct_sources_and_marks_title_only_hits(tmp_path):
    path = source(tmp_path / "litcovid.csv", [
        ("11", "Biomarker panel", "Results described an unrelated assay.", ""),
        ("12", "Another biomarker cohort", "This biomarker cohort reported twelve patients.", ""),
    ])
    destination = tmp_path / "science.sqlite3"
    build(destination, [path])
    with closing(connect(destination)) as db:
        result = search(db, "biomarker", limit=3)
        assert {r["pmid"]: r["match"] for r in result["candidates"]} == {
            "11": "document_context", "12": "passage_terms",
        }
        strict = search(db, "biomarker", limit=3, require_all=True)
        assert [r["pmid"] for r in strict["candidates"]] == ["12"]


def test_structured_abstract_filters_methods_and_results_on_same_source(tmp_path):
    path = source(tmp_path / "litcovid.csv", [
        ("11", "Clinical registry",
         "OBJECTIVE. Evaluate disease patterns. PATIENTS AND METHODS. We enrolled patients. RESULTS: 417 patients completed follow-up.", ""),
        ("12", "Unstructured abstract",
         "We enrolled patients. Small sample results. Data were not available.", ""),
    ])
    destination = tmp_path / "science.sqlite3"
    report = build(destination, [path])
    assert report["structured_papers"] == 1
    with closing(connect(destination)) as db:
        methods = search(db, "patients", pmid="11", section="methods")["candidates"]
        results = search(db, "patients", pmid="11", section="results")["candidates"]
        assert methods[0]["quote"] == "PATIENTS AND METHODS. We enrolled patients."
        assert results[0]["quote"] == "RESULTS: 417 patients completed follow-up."
        assert search(db, "patients", pmid="12", section="results")["status"] == "no_matching_passage"
