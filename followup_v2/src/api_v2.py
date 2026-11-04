"""Versioned fine-tuned inference endpoint. V1 API remains unchanged."""

from contextlib import asynccontextmanager
import threading, time, json
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer
from common import ROOT, V2, LABELS
from finetune import TopicModel, attach_lora

state = {}
lock = threading.Lock()


@asynccontextmanager
async def lifespan(app):
    configs = json.loads((V2 / "results/locked_finetune_policies.json").read_text())
    cfg = next(c for c in configs if c["variant"] == "lora_seed17")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = attach_lora(TopicModel().to(device))
    model.load_state_dict(torch.load(V2 / cfg["checkpoint"], map_location="cpu", weights_only=True))
    model.eval()
    state.update(
        model=model,
        device=device,
        cfg=cfg,
        tok=AutoTokenizer.from_pretrained(ROOT / "models/biomedbert", local_files_only=True),
    )
    yield
    state.clear()


app = FastAPI(title="Literature Label Lab — fine-tuned V2", lifespan=lifespan)


class Document(BaseModel):
    title: str = Field(min_length=1, max_length=1000)
    abstract: str = Field(min_length=20, max_length=20000)


@app.get("/health")
def health():
    return {"ready": bool(state), "version": "v2", "variant": "lora_seed17", "paid_api": False}


@app.post("/classify")
def classify(doc: Document):
    if not lock.acquire(blocking=False):
        raise HTTPException(429, "Encoder busy")
    try:
        start = time.perf_counter()
        tok = state["tok"]
        text = doc.title + " " + doc.abstract
        length = len(tok.encode(text))
        x = tok(text, padding=True, truncation=True, max_length=256, return_tensors="pt").to(
            state["device"]
        )
        with torch.inference_mode():
            p = torch.sigmoid(state["model"](**x))[0].cpu().numpy()
        t = np.array(state["cfg"]["thresholds"])
        pred = p >= t
        return {
            "version": "v2",
            "variant": state["cfg"]["variant"],
            "selected_epoch": state["cfg"]["epoch"],
            "topics": [l for j, l in enumerate(LABELS) if pred[j]],
            "scores": dict(zip(LABELS, p.tolist())),
            "input_tokens": length,
            "truncated": length > 256,
            "threshold_margin": float(np.min(np.abs(p - t))),
            "review_reasons": (["input_truncated"] if length > 256 else [])
            + (["near_classification_boundary"] if np.min(np.abs(p - t)) < 0.05 else []),
            "latency_seconds": time.perf_counter() - start,
            "limitations": [
                "Scores are not calibrated probabilities.",
                "Review reasons are heuristics, not quality guarantees.",
                "COVID-19 literature task; other domains unvalidated.",
            ],
        }
    finally:
        lock.release()
