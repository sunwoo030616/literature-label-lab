import time, gc, json
import numpy as np
import torch
import joblib
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import normalize
from common import ROOT, V2, data, train_ids, dump, metrics, thresholds, frozen_features
from diagnose import fit_grid


def encode(rows, tok, model, device, length, batch=8):
    means = []
    clss = []
    start = time.perf_counter()
    with torch.inference_mode():
        for i in range(0, len(rows), batch):
            x = tok(
                [r["text"] for r in rows[i : i + batch]],
                padding=True,
                truncation=True,
                max_length=length,
                return_tensors="pt",
            ).to(device)
            h = model(**x).last_hidden_state
            mask = x["attention_mask"].unsqueeze(-1)
            means.append(((h * mask).sum(1) / mask.sum(1).clamp(min=1)).cpu().numpy())
            clss.append(h[:, 0].cpu().numpy())
            if i % 256 == 0:
                print(
                    "encode",
                    length,
                    i,
                    "/",
                    len(rows),
                    "elapsed",
                    round(time.perf_counter() - start, 1),
                    flush=True,
                )
    return np.concatenate(means), np.concatenate(clss), time.perf_counter() - start


if __name__ == "__main__":
    torch.set_num_threads(4)
    S, Y, TUNE, CAL = data()
    tok = AutoTokenizer.from_pretrained(ROOT / "models/biomedbert", local_files_only=True)
    model = AutoModel.from_pretrained(ROOT / "models/biomedbert", local_files_only=True).eval()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)
    allrows = sum(S.values(), [])
    cache = V2 / "data/features512.npz"
    if cache.exists():
        z = np.load(cache)
        mean, cls = z["mean"], z["cls"]
        sec = 0
    else:
        mean, cls, sec = encode(allrows, tok, model, device, 512)
        np.savez_compressed(cache, mean=mean, cls=cls)
    audit256, _, sec256 = encode(S["audit"], tok, model, device, 256)
    np.save(V2 / "data/audit_mean256.npy", audit256)
    dump(
        "input_encoding.json",
        {
            "device": device,
            "mean_cls512_seconds": sec,
            "audit256_seconds": sec256,
            "count512": len(allrows),
        },
    )
    del model
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    ids = train_ids(17)
    locked = []
    objects = {}
    features = {}
    for name, emb in [("mean512", mean), ("cls512", cls)]:
        xs = {}
        n = 0
        for k, rows in S.items():
            xs[k] = emb[n : n + len(rows)]
            n += len(rows)
        features[name] = xs
        best, obj, grid = fit_grid(xs["train"][ids], Y["train"][ids], xs["dev"], Y["dev"], TUNE)
        dump(f"{name}_grid.json", grid)
        locked.append({"variant": name, **best})
        objects[name] = obj
        joblib.dump(
            {"transform": obj[0], "classifier": obj[1], "policy": best},
            V2 / "models" / f"{name}.joblib",
        )
    dump("locked_input_policies.json", locked)
    results = []
    for cfg in locked:
        name = cfg["variant"]
        tr, lr = objects[name]
        for split in ["test", "audit"]:
            x = features[name][split]
            p = lr.predict_proba(tr.transform(x) if tr else x)
            np.save(V2 / "results" / f"{name}_{split}_probs.npy", p)
            results.append(
                {"variant": name, "split": split, **metrics(Y[split], p, cfg["thresholds"])}
            )
    # Evaluate original frozen256 and tune-selected frozen256 on the untouched audit.
    old = joblib.load(ROOT / "models/classifiers.joblib")
    cfg = next(c for c in old["policies"] if c["method"] == "biomedbert")
    p = old["classifier"].predict_proba(old["scaler"].transform(audit256))
    results.append(
        {"variant": "v1_frozen256", "split": "audit", **metrics(Y["audit"], p, cfg["thresholds"])}
    )
    np.save(V2 / "results/v1_frozen256_audit_probs.npy", p)
    b = joblib.load(V2 / "models/tuned_frozen256.joblib")
    p = b["classifier"].predict_proba(
        b["transform"].transform(audit256) if b["transform"] else audit256
    )
    results.append(
        {
            "variant": "tuned_frozen256",
            "split": "audit",
            **metrics(Y["audit"], p, b["policy"]["thresholds"]),
        }
    )
    np.save(V2 / "results/tuned_frozen256_audit_probs.npy", p)
    for name in ["tfidf_full_refit", "tfidf_truncated256"]:
        b = joblib.load(V2 / "models" / f"{name}.joblib")
        texts = [r["text"] for r in S["audit"]]
        if name == "tfidf_truncated256":
            texts = tok.batch_decode(
                tok(texts, truncation=True, max_length=256, padding=False)["input_ids"],
                skip_special_tokens=True,
            )
        p = b["classifier"].predict_proba(b["vectorizer"].transform(texts))
        results.append(
            {"variant": name, "split": "audit", **metrics(Y["audit"], p, b["policy"]["thresholds"])}
        )
        np.save(V2 / "results" / f"{name}_audit_probs.npy", p)
    dump("input_ablation_results.json", results)
    print(
        "INPUT ABLATION COMPLETE",
        [(r["variant"], r["split"], r["macro_f1"]) for r in results],
        flush=True,
    )
