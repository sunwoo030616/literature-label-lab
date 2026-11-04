import json
from fastapi.testclient import TestClient
from common import V2, data
import api_v2

S, Y, TUNE, CAL = data()
checks = []
with TestClient(api_v2.app) as c:
    r = c.get("/health")
    assert r.status_code == 200 and r.json()["ready"]
    checks.append({"check": "health", "status": r.status_code})
    r = c.post("/classify", json={"title": "x", "abstract": "short"})
    assert r.status_code == 422
    checks.append({"check": "invalid", "status": r.status_code})
    row = S["audit"][0]
    doc = {"title": row["title"], "abstract": row["text"][len(row["title"]) :].strip()}
    api_v2.lock.acquire()
    try:
        r = c.post("/classify", json=doc)
        assert r.status_code == 429
        checks.append({"check": "busy", "status": r.status_code})
    finally:
        api_v2.lock.release()
    for row in S["audit"][:3]:
        r = c.post(
            "/classify",
            json={"title": row["title"], "abstract": row["text"][len(row["title"]) :].strip()},
        )
        assert r.status_code == 200
        obj = r.json()
        assert len(obj["scores"]) == 7 and all(0 <= v <= 1 for v in obj["scores"].values())
        checks.append(
            {
                "check": "actual_finetuned_inference",
                "pmid": row["pmid"],
                "status": r.status_code,
                "response": obj,
            }
        )
(V2 / "results/api_smoke.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2))
print("V2 actual model API passed", len(checks))
