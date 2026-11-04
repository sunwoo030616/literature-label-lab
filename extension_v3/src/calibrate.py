"""Binary multi-label temperature scaling; preserves classification thresholds."""

import sys, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V3 = ROOT / "extension_v3"
sys.path.insert(0, str(ROOT / "followup_v2/src"))
from common import data, LABELS
import numpy as np
from scipy.special import expit, logit
from scipy.optimize import minimize_scalar


def quality(y, p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    bce = float(np.mean(-y * np.log(p) - (1 - y) * np.log(1 - p)))
    brier = float(np.mean((p - y) ** 2))
    eces = []
    for j in range(y.shape[1]):
        ece = 0
        for i in range(10):
            keep = (p[:, j] >= i / 10) & (p[:, j] < (i + 1) / 10 if i < 9 else p[:, j] <= 1)
            if keep.any():
                ece += keep.mean() * abs(p[keep, j].mean() - y[keep, j].mean())
        eces.append(ece)
    return {
        "binary_nll": bce,
        "brier_score": brier,
        "macro_label_ece_10bins": float(np.mean(eces)),
        "per_label_ece": dict(zip(LABELS, eces)),
    }


if __name__ == "__main__":
    S, Y, TUNE, CAL = data()
    v2 = ROOT / "followup_v2"
    cfg = next(
        c
        for c in json.loads((v2 / "results/locked_finetune_policies.json").read_text())
        if c["variant"] == "lora_seed17"
    )
    dev = np.load(v2 / "results/lora_seed17_selected_dev_probs.npy").astype(np.float64)
    z = logit(np.clip(dev[CAL], 1e-7, 1 - 1e-7))
    fit = minimize_scalar(
        lambda t: quality(Y["dev"][CAL], expit(z / t))["binary_nll"],
        bounds=(0.25, 5),
        method="bounded",
    )
    T = float(fit.x)
    policy = {
        "temperature": T,
        "fitted_on": "calibration400 only",
        "calibration_nll_before": quality(Y["dev"][CAL], dev[CAL])["binary_nll"],
        "calibration_nll_after": quality(Y["dev"][CAL], expit(z / T))["binary_nll"],
        "thresholds_after": expit(logit(np.clip(cfg["thresholds"], 1e-7, 1 - 1e-7)) / T).tolist(),
    }
    (V3 / "results/locked_temperature.json").write_text(json.dumps(policy, indent=2))
    results = []
    for split in ["test", "audit"]:
        p = np.load(v2 / "results" / f"lora_seed17_{split}_probs.npy").astype(np.float64)
        q = expit(logit(np.clip(p, 1e-7, 1 - 1e-7)) / T)
        assert np.array_equal(
            p >= np.array(cfg["thresholds"]), q >= np.array(policy["thresholds_after"])
        )
        results.append(
            {
                "split": split,
                "before": quality(Y[split], p),
                "after": quality(Y[split], q),
                "classification_decisions_unchanged": True,
            }
        )
    (V3 / "results/calibration_results.json").write_text(
        json.dumps({"policy": policy, "results": results}, indent=2)
    )
    print("Calibration complete", T, results)
