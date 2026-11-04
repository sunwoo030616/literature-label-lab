import sys, json, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V3 = ROOT / "extension_v3"
V2 = ROOT / "followup_v2"
sys.path.insert(0, str(V2 / "src"))
from common import data
from finetune import TopicModel, attach_lora, infer, tokens
from lora_serving import merge_for_inference
from transformers import AutoTokenizer
import torch, numpy as np

if __name__ == "__main__":
    torch.set_num_threads(4)
    S, Y, TUNE, CAL = data()
    cfg = next(
        c
        for c in json.loads((V2 / "results/locked_finetune_policies.json").read_text())
        if c["variant"] == "lora_seed17"
    )
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(ROOT / "models/biomedbert", local_files_only=True)
    model = attach_lora(TopicModel().to(device))
    model.load_state_dict(torch.load(V2 / cfg["checkpoint"], map_location="cpu", weights_only=True))
    model.eval()
    ix = sorted(CAL, key=lambda i: S["dev"][int(i)]["sha256"])[:32]
    rows = [S["dev"][int(i)] for i in ix]
    enc = tokens(rows, tok)
    warm = tokens(rows[:8], tok)
    out = []
    predictions = {}
    merged_count = 0
    for phase in ["unmerged", "merged"]:
        if phase == "merged":
            merged_count = merge_for_inference(model)
            assert merged_count == 24
        infer(model, warm, tok, device, batch=8)
        for repeat in range(3):
            if device == "mps":
                torch.mps.synchronize()
            t = time.perf_counter()
            p, _ = infer(model, enc, tok, device, batch=8)
            if device == "mps":
                torch.mps.synchronize()
            seconds = time.perf_counter() - t
            predictions[phase] = p
            out.append(
                {
                    "phase": phase,
                    "repeat": repeat,
                    "seconds": seconds,
                    "documents_per_second": 32 / seconds,
                    "driver_mb": (
                        torch.mps.driver_allocated_memory() / 2**20 if device == "mps" else None
                    ),
                }
            )
    diff = np.abs(predictions["unmerged"] - predictions["merged"])
    t = np.array(cfg["thresholds"])
    agreement = np.mean(
        np.all((predictions["unmerged"] >= t) == (predictions["merged"] >= t), axis=1)
    )
    result = {
        "device": device,
        "documents": 32,
        "batch": 8,
        "repeats": 3,
        "merged_layers": merged_count,
        "max_probability_difference": float(diff.max()),
        "mean_probability_difference": float(diff.mean()),
        "document_labelset_agreement": float(agreement),
        "runs": out,
        "note": "Same32 calibration documents, warm-up per condition; encoder+head path only. Fixed unmerged→merged order, no API/network/load time. No global equivalence guarantee from this small sample.",
    }
    (V3 / "results/merge_benchmark.json").write_text(json.dumps(result, indent=2))
    print("Merge benchmark", result, flush=True)
