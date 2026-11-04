"""Post-hoc long-input diagnosis; never changes checkpoints or thresholds."""

import json, csv
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[2]
V2 = ROOT / "followup_v2"
V3 = ROOT / "extension_v3"


def read(p):
    return json.loads(p.read_text())


if __name__ == "__main__":
    labels = list(read(ROOT / "results/label_support.json")["test"])
    manifest = read(ROOT / "results/split_manifest.json")

    def yof(rows):
        return np.array([[int(l in r["labels"]) for l in labels] for r in rows])

    def selected(split):
        raw = {r["pmid"]: r for r in csv.DictReader((ROOT / "data/raw" / f"{split}.csv").open())}
        return [{"labels": raw[r["pmid"]]["label"].split(";")} for r in manifest[split]]

    dev = yof(selected("dev"))
    tune = np.random.default_rng(20261104).permutation(1000)[:600]
    configs = {
        c["variant"]: c
        for p in [
            V2 / "results/locked_finetune_policies.json",
            V3 / "results/locked_finetune_policies.json",
        ]
        for c in read(p)
    }
    allresults = read(V2 / "results/finetune_results.json") + read(
        V3 / "results/finetune_results.json"
    )
    summary = []
    for split, rows in [
        ("test", selected("test")),
        ("audit", read(V2 / "data/audit_manifest.json")),
    ]:
        y = yof(rows)
        conditions = {}
        for name, home in [("lora_seed17", V2), ("lora512_seed17", V3)]:
            cfg = configs[name]
            p = np.load(home / "results" / f"{name}_{split}_probs.npy")
            pred = p >= np.array(cfg["thresholds"])
            recorded = next(r for r in allresults if r["variant"] == name and r["split"] == split)
            conditions[name] = {
                "macro_f1": recorded["macro_f1"],
                "macro_average_precision": recorded["macro_ap"],
                "labels": {},
            }
            for j, l in enumerate(labels):
                tp = int((y[:, j] * pred[:, j]).sum())
                fp = int(((1 - y[:, j]) * pred[:, j]).sum())
                fn = int((y[:, j] * (1 - pred[:, j])).sum())
                support = int(y[:, j].sum())
                conditions[name]["labels"][l] = {
                    "support": support,
                    "tune_support": int(dev[tune, j].sum()),
                    "threshold": cfg["thresholds"][j],
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0,
                    "average_precision": (
                        float(average_precision_score(y[:, j], p[:, j])) if support else 0
                    ),
                    "mean_score_positive": float(p[y[:, j] == 1, j].mean()) if support else None,
                    "mean_score_negative": float(p[y[:, j] == 0, j].mean()),
                }
        summary.append({"split": split, "conditions": conditions})
    out = {
        "splits": summary,
        "interpretation": "Descriptive fixed-model comparison. AP evaluates ranking without the selected threshold; F1 evaluates chosen decisions. Divergence can motivate calibration/threshold hypotheses, but does not prove a cause. Padding, physical batch/dropout RNG and training input all differ; one additional512seed. No retuning from these reused-test/audit labels. Final new internal-audit comparison was predefined separately.",
    }
    (V3 / "results/long_input_diagnosis.json").write_text(json.dumps(out, indent=2))
    print(
        [
            (
                r["split"],
                {
                    k: (v["macro_f1"], v["macro_average_precision"])
                    for k, v in r["conditions"].items()
                },
            )
            for r in summary
        ]
    )
