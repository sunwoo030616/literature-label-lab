"""Optional time-budget-driven second-seed replication, never chosen by test score."""

import sys, json, time, gc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "followup_v2/src"))
import finetune512 as ft
import torch, numpy as np
from common import data
from transformers import AutoTokenizer

if __name__ == "__main__":
    torch.set_num_threads(4)
    S, Y, TUNE, CAL = data()
    ft.YDEV = Y["dev"]
    tok = AutoTokenizer.from_pretrained(ROOT / "models/biomedbert", local_files_only=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    cfg = ft.train_one(
        "lora512",
        42,
        ft.tokens(S["train"], tok),
        ft.tokens(S["dev"], tok),
        Y["train"],
        TUNE,
        tok,
        device,
    )
    ft.dump("locked_replica512_policy.json", cfg)
    model = ft.attach_lora(ft.TopicModel().to(device))
    model.load_state_dict(
        torch.load(ft.V2 / cfg["checkpoint"], map_location="cpu", weights_only=True)
    )
    out = []
    for split in ["test", "audit"]:
        p, sec = ft.infer(model, ft.tokens(S[split], tok), tok, device)
        np.save(ft.V2 / "results" / f"lora512_seed42_{split}_probs.npy", p)
        out.append(
            {
                "variant": "lora512_seed42",
                "split": split,
                "inference_seconds": sec,
                **ft.metrics(Y[split], p, cfg["thresholds"]),
            }
        )
    ft.dump("replica512_results.json", out)
    print("512 replication complete", [(r["split"], r["macro_f1"]) for r in out])
