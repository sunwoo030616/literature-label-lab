"""Descriptive geometry/length analysis; not used for model selection."""

import numpy as np, json
from transformers import AutoTokenizer
from sklearn.preprocessing import StandardScaler, normalize
from common import ROOT, V2, data, train_ids, frozen_features, dump, row_f1

S, Y, TUNE, CAL = data()
X = frozen_features()
ids = train_ids(17)
x = X["train"][ids]
y = Y["train"][ids]
rng = np.random.default_rng(20261104)
a = rng.integers(0, len(x), 20000)
b = rng.integers(0, len(x), 20000)
same = (y[a] * y[b]).sum(1) > 0
geometry = {}
for name, v in [
    ("raw", x),
    ("centered", x - x.mean(0)),
    ("standardized", StandardScaler().fit_transform(x)),
]:
    u = normalize(v)
    cos = (u[a] * u[b]).sum(1)
    geometry[name] = {
        "random_pair_cosine_mean": float(cos.mean()),
        "random_pair_cosine_std": float(cos.std()),
        "shared_label_mean": float(cos[same].mean()),
        "no_shared_label_mean": float(cos[~same].mean()),
    }
tok = AutoTokenizer.from_pretrained(ROOT / "models/biomedbert", local_files_only=True)
length = np.array(
    [len(v) for v in tok([r["text"] for r in S["test"]], truncation=False)["input_ids"]]
)
oldcfg = next(
    c
    for c in json.loads((ROOT / "results/locked_policies.json").read_text())
    if c["seed"] == 17 and c["method"] == "biomedbert"
)
newcfg = json.loads((V2 / "results/locked_classifier_policy.json").read_text())
old = np.load(ROOT / "results/probabilities_biomedbert.npy") >= np.array(oldcfg["thresholds"])
new = np.load(V2 / "results/tuned_frozen256_test_probs.npy") >= np.array(newcfg["thresholds"])
groups = []
for lo, hi in [(0, 128), (129, 256), (257, 512), (513, 100000)]:
    m = (length >= lo) & (length <= hi)
    if m.any():
        groups.append(
            {
                "tokens": [lo, hi],
                "count": int(m.sum()),
                "old_sample_f1": float(row_f1(Y["test"][m], old[m]).mean()),
                "new_sample_f1": float(row_f1(Y["test"][m], new[m]).mean()),
                "old_exact_match": float((old[m] == Y["test"][m]).all(1).mean()),
                "new_exact_match": float((new[m] == Y["test"][m]).all(1).mean()),
            }
        )
dump(
    "feature_diagnostics.json",
    {
        "geometry_on_train": geometry,
        "test_length_groups": groups,
        "note": "Descriptive geometry is not causal proof of classifier performance. Shared-label cosine also depends on label prevalence. Length slices are post-hoc descriptions, not selection.",
    },
)
print(geometry, groups)
