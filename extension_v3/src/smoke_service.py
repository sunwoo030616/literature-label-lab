import sys, json
from pathlib import Path
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "followup_v2/src"))
from common import data
from service import app

ROOT = Path(__file__).resolve().parents[2]
S, Y, TUNE, CAL = data()
r = S["audit"][3]
doc = {"title": r["title"], "abstract": r["text"][len(r["title"]) :].strip()}
checks = []
with TestClient(app) as client:
    a = client.post("/documents", json=doc)
    assert a.status_code == 200
    obj = a.json()
    checks.append({"check": "actual_prediction_registration", "response": obj})
    b = client.post("/documents", json=doc)
    assert b.status_code == 200 and b.json()["id"] == obj["id"]
    checks.append({"check": "duplicate_registration_same_id", "id": obj["id"]})
    id = obj["id"]
    rev = obj["revision"]
    x = client.patch(
        f"/documents/{id}/review",
        json={"expected_revision": rev, "labels": r["labels"], "evidence": []},
    )
    assert x.status_code == 200
    checks.append({"check": "review_saved", "revision": x.json()["revision"]})
    x = client.patch(
        f"/documents/{id}/review", json={"expected_revision": rev, "labels": [], "evidence": []}
    )
    assert x.status_code == 409
    checks.append({"check": "stale_review_rejected", "status": x.status_code})
    x = client.get(f"/documents/{id}")
    assert x.json()["review"]["labels"] == r["labels"]
    checks.append({"check": "read_back_saved_review", "status": x.json()["status"]})
with TestClient(app) as client:
    before = client.get(f"/documents/{id}").json()
    result = client.post(f"/documents/{id}/evidence")
    assert result.status_code == 200
    llm_result = result.json()["result"]
    assert 0 < llm_result["used_document_tokens"] <= llm_result["max_document_tokens"]
    assert llm_result["document_truncated"] == (
        llm_result["input_tokens"] > llm_result["max_document_tokens"]
    )
    after = client.get(f"/documents/{id}").json()
    assert after["review"] == before["review"] and after["prediction"] == before["prediction"]
    checks.append({"check": "actual_local_llm_evidence", "response": result.json()})
    history = client.get(f"/documents/{id}/evidence").json()
    assert len(history) >= 1
    checks.append({"check": "llm_evidence_persisted_separately", "runs": len(history)})
(ROOT / "extension_v3/results/service_smoke.json").write_text(
    json.dumps(checks, ensure_ascii=False, indent=2)
)
print("Service actual-model smoke passed", len(checks))
