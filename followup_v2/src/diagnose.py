"""CPU diagnostics; classifier selection uses tune only."""

import json, time
import numpy as np
from transformers import AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
import joblib
from common import ROOT, V2, data, train_ids, frozen_features, dump, thresholds, metrics, LABELS


def fit_grid(x, y, xd, yd, tune):
    choices = []
    best = None
    bestobj = None
    for scaling in ["raw", "standard", "l2"]:
        tr = (
            StandardScaler() if scaling == "standard" else Normalizer() if scaling == "l2" else None
        )
        xx = tr.fit_transform(x) if tr else x
        dd = tr.transform(xd) if tr else xd
        for C in [0.001, 0.01, 0.1, 1, 10]:
            start = time.perf_counter()
            lr = OneVsRestClassifier(
                LogisticRegression(C=C, max_iter=1500, solver="liblinear")
            ).fit(xx, y)
            p = lr.predict_proba(dd)
            t = thresholds(yd[tune], p[tune])
            trainp = lr.predict_proba(xx)
            m = metrics(yd[tune], p[tune], t)
            row = {
                "scaling": scaling,
                "C": C,
                "thresholds": t,
                "tune_macro_f1": m["macro_f1"],
                "train_macro_f1": f1_score(y, trainp >= t, average="macro", zero_division=0),
                "fit_seconds": time.perf_counter() - start,
            }
            choices.append(row)
            if best is None or m["macro_f1"] > best["tune_macro_f1"] + 1e-12:
                best = row
                bestobj = (tr, lr)
    return best, bestobj, choices


if __name__ == "__main__":
    S, Y, TUNE, CAL = data()
    X = frozen_features()
    ids = train_ids(17)
    tok = AutoTokenizer.from_pretrained(ROOT / "models/biomedbert", local_files_only=True)
    lengths = {}
    for split, rows in S.items():
        ll = np.array(
            [
                len(v)
                for v in tok(
                    [r["text"] for r in rows],
                    truncation=False,
                    padding=False,
                    add_special_tokens=True,
                )["input_ids"]
            ]
        )
        lengths[split] = {
            "median": np.median(ll),
            "p90": np.quantile(ll, 0.9),
            "over256": np.mean(ll > 256),
            "over512": np.mean(ll > 512),
            "by_label": {
                l: {
                    "count": int(Y[split][:, j].sum()),
                    "over256": float(np.mean(ll[Y[split][:, j] == 1] > 256)),
                }
                for j, l in enumerate(LABELS)
            },
        }
    dump("length_audit.json", lengths)
    best, obj, grid = fit_grid(X["train"][ids], Y["train"][ids], X["dev"], Y["dev"], TUNE)
    dump("classifier_grid.json", grid)
    dump("locked_classifier_policy.json", best)
    joblib.dump(
        {"transform": obj[0], "classifier": obj[1], "policy": best},
        V2 / "models/tuned_frozen256.joblib",
    )
    tr, lr = obj
    pt = lr.predict_proba(tr.transform(X["test"]) if tr else X["test"])
    np.save(V2 / "results/tuned_frozen256_test_probs.npy", pt)
    results = [
        {
            "variant": "tuned_frozen256",
            "split": "test",
            **metrics(Y["test"], pt, best["thresholds"]),
        }
    ]
    for trunc in [False, True]:
        name = "tfidf_truncated256" if trunc else "tfidf_full_refit"
        texts = {k: [r["text"] for r in rows] for k, rows in S.items() if k != "audit"}
        if trunc:
            texts = {
                k: tok.batch_decode(
                    tok(v, truncation=True, max_length=256, padding=False)["input_ids"],
                    skip_special_tokens=True,
                )
                for k, v in texts.items()
            }
        vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        x = vec.fit_transform([texts["train"][i] for i in ids])
        clf = OneVsRestClassifier(LogisticRegression(C=1, max_iter=1000, solver="liblinear")).fit(
            x, Y["train"][ids]
        )
        pd = clf.predict_proba(vec.transform(texts["dev"]))
        t = thresholds(Y["dev"][TUNE], pd[TUNE])
        policy = {
            "variant": name,
            "thresholds": t,
            "tune_macro_f1": metrics(Y["dev"][TUNE], pd[TUNE], t)["macro_f1"],
        }
        dump(f"{name}_policy.json", policy)
        joblib.dump(
            {"vectorizer": vec, "classifier": clf, "policy": policy},
            V2 / "models" / f"{name}.joblib",
        )
        p = clf.predict_proba(vec.transform(texts["test"]))
        results.append({"variant": name, "split": "test", **metrics(Y["test"], p, t)})
    dump("diagnostic_results.json", results)
    print("DIAGNOSTICS COMPLETE", [(r["variant"], r["macro_f1"]) for r in results], flush=True)
