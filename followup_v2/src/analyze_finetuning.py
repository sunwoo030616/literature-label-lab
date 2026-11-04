"""Post-evaluation paired uncertainty and error transitions; no model selection."""

from common import ROOT, V2, LABELS, data, dump
import numpy as np, json


def macro(y, p):
    tp = (y * p).sum(0)
    fp = ((1 - y) * p).sum(0)
    fn = (y * (1 - p)).sum(0)
    den = 2 * tp + fp + fn
    return float(np.divide(2 * tp, den, out=np.zeros(7, dtype=float), where=den > 0).mean())


if __name__ == "__main__":
    S, Y, _, _ = data()
    oldcfg = next(
        c
        for c in json.loads((ROOT / "results/locked_policies.json").read_text())
        if c["seed"] == 17 and c["method"] == "biomedbert"
    )
    tuned = json.loads((V2 / "results/locked_classifier_policy.json").read_text())
    ft = {
        c["variant"]: c
        for c in json.loads((V2 / "results/locked_finetune_policies.json").read_text())
    }
    summary = {}
    examples = []
    for split in ["test", "audit"]:
        yp = {}
        p = (
            np.load(ROOT / "results/probabilities_biomedbert.npy")
            if split == "test"
            else np.load(V2 / "results/v1_frozen256_audit_probs.npy")
        )
        yp["v1"] = p >= np.array(oldcfg["thresholds"])
        p = np.load(V2 / "results" / f"tuned_frozen256_{split}_probs.npy")
        yp["tuned"] = p >= np.array(tuned["thresholds"])
        for name in ["last2_seed17", "lora_seed17"]:
            yp[name] = np.load(V2 / "results" / f"{name}_{split}_probs.npy") >= np.array(
                ft[name]["thresholds"]
            )
        results = {}
        y = Y[split]
        for a, b in [("v1", "tuned"), ("v1", "last2_seed17"), ("tuned", "lora_seed17")]:
            correct_a = (yp[a] == y).all(1)
            correct_b = (yp[b] == y).all(1)
            delta = []
            rng = np.random.default_rng(20261104)
            for _ in range(1000):
                ix = rng.integers(0, len(y), len(y))
                delta.append(macro(y[ix], yp[b][ix]) - macro(y[ix], yp[a][ix]))
            results[f"{b}_minus_{a}"] = {
                "point_delta_macro_f1": macro(y, yp[b]) - macro(y, yp[a]),
                "bootstrap_ci95": np.quantile(delta, [0.025, 0.975]),
                "exact_match_fixed": int((~correct_a & correct_b).sum()),
                "exact_match_broken": int((correct_a & ~correct_b).sum()),
                "per_label": {
                    l: {
                        n: {
                            "tp": int((y[:, j] * pp[:, j]).sum()),
                            "fp": int(((1 - y[:, j]) * pp[:, j]).sum()),
                            "fn": int((y[:, j] * (1 - pp[:, j])).sum()),
                        }
                        for n, pp in [(a, yp[a]), (b, yp[b])]
                    }
                    for j, l in enumerate(LABELS)
                },
            }
        for name in ["last2_seed17", "lora_seed17"]:
            wrong = np.flatnonzero(np.any(yp[name] != y, axis=1))[:5]
            for i in wrong:
                examples.append(
                    {
                        "split": split,
                        "variant": name,
                        "pmid": S[split][i]["pmid"],
                        "title": S[split][i]["title"],
                        "gold": S[split][i]["labels"],
                        "predicted": [l for j, l in enumerate(LABELS) if yp[name][i, j]],
                        "false_positive": [
                            l for j, l in enumerate(LABELS) if yp[name][i, j] and not y[i, j]
                        ],
                        "false_negative": [
                            l for j, l in enumerate(LABELS) if y[i, j] and not yp[name][i, j]
                        ],
                    }
                )
        summary[split] = results
    dump(
        "finetuning_error_analysis.json",
        {
            "comparisons": summary,
            "examples": examples,
            "note": "Paired bootstrap accounts only for document resampling with fixed trained models, not hyperparameter/model selection or new domains. Examples are descriptive first errors by locked order, not causal explanations.",
        },
    )
    print("Paired model/error analysis completed.")
