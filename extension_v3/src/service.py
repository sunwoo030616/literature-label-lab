from pathlib import Path
import sys, json

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "followup_v2/src"))
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import api_v2
from common import LABELS, V2
from review_store import ReviewStore, ConflictError

STORE = ReviewStore(ROOT / "extension_v3/data/review.sqlite3")


@asynccontextmanager
async def lifespan(app):
    async with api_v2.lifespan(app):
        benchmark = ROOT / "extension_v3/results/merge_benchmark.json"
        api_v2.state["merged"] = False
        if benchmark.exists():
            checked = json.loads(benchmark.read_text())
            if (
                checked["max_probability_difference"] <= 1e-3
                and checked["document_labelset_agreement"] == 1
            ):
                from lora_serving import merge_for_inference

                api_v2.state["merged"] = merge_for_inference(api_v2.state["model"]) == 24
        yield


app = FastAPI(title="Literature research review workflow", lifespan=lifespan)


class Review(BaseModel):
    expected_revision: int = Field(ge=0)
    labels: list[str]
    evidence: list[dict] = Field(default_factory=list)


@app.post("/documents")
def register(doc: api_v2.Document):
    pred = api_v2.classify(doc)
    cfg = api_v2.state["cfg"]
    pred["serving_merged"] = api_v2.state.get("merged", False)
    version = cfg["checkpoint_sha256"] + (":merged-v1" if pred["serving_merged"] else ":unmerged")
    calibration = ROOT / "extension_v3/results/locked_temperature.json"
    if calibration.exists():
        import hashlib
        import numpy as np
        from scipy.special import expit, logit

        policy = json.loads(calibration.read_text())
        temperature = policy["temperature"]
        raw = np.array([pred["scores"][label] for label in LABELS], dtype=np.float64)
        adjusted = expit(logit(np.clip(raw, 1e-7, 1 - 1e-7)) / temperature)
        pred["calibrated_scores"] = dict(zip(LABELS, adjusted.tolist()))
        pred["calibration"] = {"temperature": temperature, "quality_guaranteed": False}
        version += ":" + hashlib.sha256(calibration.read_bytes()).hexdigest()
    return STORE.register(doc.title, doc.abstract, version, pred)


@app.get("/documents/{id}")
def get(id: int):
    try:
        return STORE.get(id)
    except KeyError:
        raise HTTPException(404, "Document not found")


@app.patch("/documents/{id}/review")
def review(id: int, req: Review):
    try:
        return STORE.review(id, req.expected_revision, req.labels, req.evidence, LABELS)
    except KeyError:
        raise HTTPException(404, "Document not found")
    except ConflictError:
        raise HTTPException(409, "Stale revision; reload before reviewing")
    except ValueError as e:
        raise HTTPException(422, str(e))


@app.post("/documents/{id}/evidence")
def evidence(id: int):
    try:
        row = STORE.get(id)
    except KeyError:
        raise HTTPException(404, "Document not found")
    if not api_v2.lock.acquire(blocking=False):
        raise HTTPException(429, "Local GPU busy")
    try:
        from llm_evidence_service import generate_evidence

        result = generate_evidence(row["title"], row["abstract"])
        return STORE.append_llm_run(id, result)
    finally:
        api_v2.lock.release()


@app.get("/documents/{id}/evidence")
def evidence_history(id: int):
    try:
        return STORE.llm_runs(id)
    except KeyError:
        raise HTTPException(404, "Document not found")
