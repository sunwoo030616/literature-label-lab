"""Post-test-motivated development branch; fit decisions only within original tune600."""

import json, csv, hashlib
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
V2 = ROOT / "followup_v2"
V3 = ROOT / "extension_v3"
GRID = np.arange(0.1, 0.81, 0.05)
LAMBDAS = [0, 0.25, 0.5, 0.75, 1]


def read(p):
    return json.loads(p.read_text())


def f1labels(y, p):
    tp = (y * p).sum(0)
    fp = ((1 - y) * p).sum(0)
    fn = (y * (1 - p)).sum(0)
    d = 2 * tp + fp + fn
    return np.divide(2 * tp, d, out=np.zeros(y.shape[1], dtype=float), where=d > 0)


def fit(y, p):
    scores = np.array([f1labels(y, p >= t) for t in GRID])
    return GRID[scores.argmax(0)], float(GRID[scores.mean(1).argmax()])


def metrics(y, p, t):
    yp = p >= np.array(t)
    tp = (y * yp).sum()
    fp = ((1 - y) * yp).sum()
    fn = (y * (1 - yp)).sum()
    return {
        "macro_f1": float(f1labels(y, yp).mean()),
        "micro_f1": float(2 * tp / (2 * tp + fp + fn)),
        "hamming_loss": float((y != yp).mean()),
        "subset_accuracy": float((y == yp).all(1).mean()),
    }


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
    folds = np.array_split(np.random.default_rng(20261105).permutation(600), 5)
    configs = {
        c["variant"]: c
        for p in [
            V2 / "results/locked_finetune_policies.json",
            V3 / "results/locked_finetune_policies.json",
        ]
        for c in read(p)
    }
    locked = []
    histories = []
    for source, home, name in [
        ("lora_seed17", V2, "threshold_cv_lora256_seed17"),
        ("lora512_seed17", V3, "threshold_cv_lora512_seed17"),
    ]:
        pp = np.load(home / "results" / f"{source}_selected_dev_probs.npy")[tune]
        yy = dev[tune]
        scores = {lam: [] for lam in LAMBDAS}
        for held in folds:
            train = np.setdiff1d(np.arange(600), held)
            per, glob = fit(yy[train], pp[train])
            for lam in LAMBDAS:
                scores[lam].append(
                    float(f1labels(yy[held], pp[held] >= (lam * per + (1 - lam) * glob)).mean())
                )
        lam = max(LAMBDAS, key=lambda x: np.mean(scores[x]))
        per, glob = fit(yy, pp)
        np.testing.assert_allclose(per, configs[source]["thresholds"], rtol=0, atol=1e-12)
        threshold = lam * per + (1 - lam) * glob
        cfg = {
            "variant": name,
            "source_variant": source,
            "lambda": lam,
            "thresholds": threshold.tolist(),
            "unshrunk_thresholds": per.tolist(),
            "global_threshold": glob,
            "fit_on": "original tune600 only, fivefold internal selection; CAL400 untouched",
            "tie_rule": "first maximum in ascending lambda order, prefer more shared threshold on exact tie",
            "cv_mean_macro_f1": float(np.mean(scores[lam])),
            "source_checkpoint_sha256": configs[source]["checkpoint_sha256"],
        }
        locked.append(cfg)
        histories.append(
            {
                "variant": name,
                "lambda_fold_scores": scores,
                "tune_support": dict(zip(labels, yy.sum(0).tolist())),
            }
        )
    (V3 / "results/threshold_cv_history.json").write_text(json.dumps(histories, indent=2))
    (V3 / "results/locked_threshold_cv_policies.json").write_text(json.dumps(locked, indent=2))
    # Existing development test/audit only after locking; never choose lambda from these scores.
    results = []
    for cfg in locked:
        home = V2 if cfg["source_variant"] == "lora_seed17" else V3
        for split, rows in [
            ("test", selected("test")),
            ("audit", read(V2 / "data/audit_manifest.json")),
        ]:
            p = np.load(home / "results" / f"{cfg['source_variant']}_{split}_probs.npy")
            results.append(
                {
                    "variant": cfg["variant"],
                    "source_variant": cfg["source_variant"],
                    "split": split,
                    **metrics(yof(rows), p, cfg["thresholds"]),
                }
            )
    (V3 / "results/threshold_cv_results.json").write_text(json.dumps(results, indent=2))
    print(
        "Threshold CV locked", [(c["variant"], c["lambda"], c["cv_mean_macro_f1"]) for c in locked]
    )
    print(results)
