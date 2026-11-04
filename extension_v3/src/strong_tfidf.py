"""Tune classical baseline on tune only; keep the original C1 baseline intact."""

import sys, json, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V3 = ROOT / "extension_v3"
sys.path.insert(0, str(ROOT / "followup_v2/src"))
from common import data, train_ids, thresholds, metrics
import numpy as np, joblib
from transformers import AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression


def dump(name, obj):
    (V3 / "results" / name).write_text(
        json.dumps(
            obj,
            ensure_ascii=False,
            indent=2,
            default=lambda a: a.tolist() if isinstance(a, np.ndarray) else float(a),
        )
    )


if __name__ == "__main__":
    S, Y, TUNE, CAL = data()
    ids = train_ids(17)
    tok = AutoTokenizer.from_pretrained(ROOT / "models/biomedbert", local_files_only=True)
    locked = []
    objects = {}
    alltexts = {}
    grids = []
    for truncated in [False, True]:
        name = "tuned_tfidf_truncated256" if truncated else "tuned_tfidf_full"
        texts = {k: [r["text"] for r in rows] for k, rows in S.items()}
        if truncated:
            texts = {
                k: tok.batch_decode(
                    tok(v, truncation=True, max_length=256, padding=False)["input_ids"],
                    skip_special_tokens=True,
                )
                for k, v in texts.items()
            }
        alltexts[name] = texts
        vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        x = vec.fit_transform([texts["train"][i] for i in ids])
        xd = vec.transform(texts["dev"])
        best = None
        bestclf = None
        for C in [0.001, 0.01, 0.1, 1, 10]:
            start = time.perf_counter()
            clf = OneVsRestClassifier(
                LogisticRegression(C=C, max_iter=1500, solver="liblinear")
            ).fit(x, Y["train"][ids])
            p = clf.predict_proba(xd)
            t = thresholds(Y["dev"][TUNE], p[TUNE])
            score = metrics(Y["dev"][TUNE], p[TUNE], t)["macro_f1"]
            row = {
                "variant": name,
                "C": C,
                "thresholds": t,
                "tune_macro_f1": score,
                "fit_seconds": time.perf_counter() - start,
            }
            grids.append(row)
            if best is None or score > best["tune_macro_f1"] + 1e-12:
                best = row
                bestclf = clf
        locked.append(best)
        objects[name] = (vec, bestclf)
        joblib.dump(
            {"vectorizer": vec, "classifier": bestclf, "policy": best},
            V3 / "models" / f"{name}.joblib",
        )
    dump("strong_tfidf_grid.json", grids)
    dump("locked_strong_tfidf_policies.json", locked)
    results = []
    for cfg in locked:
        vec, clf = objects[cfg["variant"]]
        for split in ["test", "audit"]:
            p = clf.predict_proba(vec.transform(alltexts[cfg["variant"]][split]))
            np.save(V3 / "results" / f"{cfg['variant']}_{split}_probs.npy", p)
            results.append(
                {
                    "variant": cfg["variant"],
                    "split": split,
                    **metrics(Y[split], p, cfg["thresholds"]),
                }
            )
    dump("strong_tfidf_results.json", results)
    print(
        "Strong baseline complete",
        [(r["variant"], r["split"], r["macro_f1"]) for r in results],
        flush=True,
    )
