"""Predefined fixed-model paired document bootstrap after the final audit; no selection."""

import sys, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V2 = ROOT / "followup_v2"
V3 = ROOT / "extension_v3"
sys.path.insert(0, str(V2 / "src"))
from common import yof, LABELS
import numpy as np

PAIRS = [
    ("v1_frozen256", "tuned_frozen256"),
    ("v1_frozen256", "last2_seed17"),
    ("tuned_tfidf_full", "lora_seed17"),
    ("lora_seed17", "lora512_seed17"),
    ("uniform_attention_seed17", "attention_seed17"),
    ("random_graph_seed17", "label_graph_seed17"),
]


def macro(y, p):
    tp = (y * p).sum(0)
    fp = ((1 - y) * p).sum(0)
    fn = (y * (1 - p)).sum(0)
    d = 2 * tp + fp + fn
    return float(np.divide(2 * tp, d, out=np.zeros(7, dtype=float), where=d > 0).mean())


if __name__ == "__main__":
    read = lambda p: json.loads(p.read_text())
    y = yof(read(V3 / "data/final_audit_documents.json"))
    thresholds = {}
    old = read(ROOT / "results/locked_policies.json")
    thresholds["v1_frozen256"] = next(
        c["thresholds"] for c in old if c["seed"] == 17 and c["method"] == "biomedbert"
    )
    thresholds["tuned_frozen256"] = read(V2 / "results/locked_classifier_policy.json")["thresholds"]
    for p in [
        V2 / "results/locked_finetune_policies.json",
        V3 / "results/locked_finetune_policies.json",
        V3 / "results/locked_policies.json",
        V3 / "results/locked_strong_tfidf_policies.json",
    ]:
        for c in read(p):
            thresholds[c["variant"]] = c["thresholds"]
    pred = {
        name: np.load(V3 / "results" / f"final_audit_{name}_probs.npy")
        >= np.array(thresholds[name])
        for pair in PAIRS
        for name in pair
    }
    out = []
    for a, b in PAIRS:
        rng = np.random.default_rng(20261104)
        delta = []
        for _ in range(1000):
            ix = rng.integers(0, len(y), len(y))
            delta.append(macro(y[ix], pred[b][ix]) - macro(y[ix], pred[a][ix]))
        out.append(
            {
                "comparison": b + " minus " + a,
                "delta_macro_f1": macro(y, pred[b]) - macro(y, pred[a]),
                "paired_document_bootstrap_ci95": np.quantile(delta, [0.025, 0.975]).tolist(),
                "resamples": 1000,
            }
        )
    result = {
        "comparisons": out,
        "label_support": dict(zip(LABELS, y.sum(0).tolist())),
        "scope": "One new internal1000-document set; same source corpus, not external validation. Models, thresholds and comparisons fixed before this set. Percentile intervals only resample documents for fixed models; no model/selection/domain uncertainty and no familywise multiple-comparison correction. Not a global significance claim.",
    }
    (V3 / "results/final_audit_uncertainty.json").write_text(json.dumps(result, indent=2))
    print(result)
