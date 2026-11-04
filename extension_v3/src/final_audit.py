"""One final internal confirmation set after all model/prompt choices are frozen."""

import sys, json, hashlib, csv, re, gc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V2 = ROOT / "followup_v2"
V3 = ROOT / "extension_v3"
sys.path.insert(0, str(V2 / "src"))
from common import data, LABELS, yof, metrics, llm_path
from finetune import TopicModel, attach_lora, infer, tokens
from input_ablation import encode
from bag_model import AttentionHead, predict, make_bags
import numpy as np, torch, joblib
from transformers import AutoTokenizer, AutoModel


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
    required = [
        V2 / "results/locked_finetune_policies.json",
        V2 / "results/locked_input_policies.json",
        V3 / "results/locked_finetune_policies.json",
        V3 / "results/locked_policies.json",
    ]
    assert all(p.exists() for p in required)
    required += [
        V2 / "src/common.py",
        V2 / "src/finetune.py",
        V3 / "src/finetune512.py",
        V3 / "src/bag_model.py",
        V3 / "src/final_audit.py",
    ]
    required += [V2 / "src/llm_review.py", V3 / "src/llm_fewshot.py"]
    for optional in [
        V2 / "results/locked_classifier_policy.json",
        V3 / "results/locked_strong_tfidf_policies.json",
        V3 / "results/locked_replica512_policy.json",
        V3 / "results/locked_temperature.json",
        V2 / "results/llm_sample_manifest.json",
        V3 / "results/llm_training_examples.json",
        llm_path() / "manifest.local.json",
    ]:
        if optional.exists():
            required.append(optional)
    required += [
        p
        for directory in [V2 / "models", V3 / "models"]
        for p in directory.glob("*")
        if p.is_file()
    ]
    dump(
        "final_audit_locked_model_hashes.json",
        {
            (
                str(p.relative_to(ROOT.parent))
                if p.is_relative_to(ROOT.parent)
                else "external_llm_snapshot/" + p.name
            ): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in required
        },
    )
    S, Y, TUNE, CAL = data()
    usedids = {r["pmid"] for rows in S.values() for r in rows}
    usedhash = {r["sha256"] for rows in S.values() for r in rows}
    raws = {}
    useddoi = set()
    for split in ["train", "dev", "test"]:
        rr = list(csv.DictReader((ROOT / "data/raw" / f"{split}.csv").open()))
        raws[split] = rr
        for r in rr:
            if r["pmid"] in usedids and r["doi"].strip():
                useddoi.add(re.sub(r"^https?://(?:dx\.)?doi\.org/", "", r["doi"].strip().lower()))
    candidates = []
    for r in raws["train"]:
        text = r["title"].strip() + " " + r["abstract"].strip()
        h = hashlib.sha256(" ".join(text.lower().split()).encode()).hexdigest()
        doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", r["doi"].strip().lower())
        if (
            not r["abstract"].strip()
            or r["pmid"] in usedids
            or h in usedhash
            or (doi and doi in useddoi)
        ):
            continue
        usedids.add(r["pmid"])
        usedhash.add(h)
        if doi:
            useddoi.add(doi)
        candidates.append(
            {
                "pmid": r["pmid"],
                "title": r["title"],
                "text": text,
                "labels": r["label"].split(";"),
                "sha256": h,
            }
        )
    rows = sorted(candidates, key=lambda r: r["sha256"])[:1000]
    assert len(rows) == 1000
    y = yof(rows)
    dump("final_audit_manifest.json", [{"pmid": r["pmid"], "sha256": r["sha256"]} for r in rows])
    (V3 / "data/final_audit_documents.json").write_text(json.dumps(rows, ensure_ascii=False))
    torch.set_num_threads(4)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(ROOT / "models/biomedbert", local_files_only=True)
    model = (
        AutoModel.from_pretrained(ROOT / "models/biomedbert", local_files_only=True)
        .eval()
        .to(device)
    )
    f256, _, _ = encode(rows, tok, model, device, 256)
    f512, c512, _ = encode(rows, tok, model, device, 512)
    bags, mask, windows = make_bags(rows, tok, model, device)
    del model
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    results = []

    def add(name, p, t):
        np.save(V3 / "results" / f"final_audit_{name}_probs.npy", p)
        results.append({"variant": name, "split": "final_internal_audit1000", **metrics(y, p, t)})

    old = joblib.load(ROOT / "models/classifiers.joblib")
    cfg = next(c for c in old["policies"] if c["method"] == "biomedbert")
    add(
        "v1_frozen256",
        old["classifier"].predict_proba(old["scaler"].transform(f256)),
        cfg["thresholds"],
    )
    cfg = next(c for c in old["policies"] if c["method"] == "tfidf")
    add(
        "v1_tfidf",
        old["tfidf"].predict_proba(old["vectorizer"].transform([r["text"] for r in rows])),
        cfg["thresholds"],
    )
    for name, x in [("tuned_frozen256", f256), ("mean512", f512), ("cls512", c512)]:
        b = joblib.load(V2 / "models" / f"{name}.joblib")
        p = b["classifier"].predict_proba(b["transform"].transform(x) if b["transform"] else x)
        add(name, p, b["policy"]["thresholds"])
    for name in ["tuned_tfidf_full", "tuned_tfidf_truncated256"]:
        path = V3 / "models" / f"{name}.joblib"
        if path.exists():
            b = joblib.load(path)
            texts = [r["text"] for r in rows]
            if name.endswith("truncated256"):
                texts = tok.batch_decode(
                    tok(texts, truncation=True, max_length=256, padding=False)["input_ids"],
                    skip_special_tokens=True,
                )
            add(
                name,
                b["classifier"].predict_proba(b["vectorizer"].transform(texts)),
                b["policy"]["thresholds"],
            )
    configs = [
        (V2, c) for c in json.loads((V2 / "results/locked_finetune_policies.json").read_text())
    ] + [(V3, c) for c in json.loads((V3 / "results/locked_finetune_policies.json").read_text())]
    if (V3 / "results/locked_replica512_policy.json").exists():
        configs.append((V3, json.loads((V3 / "results/locked_replica512_policy.json").read_text())))
    for home, cfg in configs:
        model = TopicModel(
            attention_backend=("eager" if cfg["mode"] == "last2" else "sdpa"),
            attention_dropout=(0.1 if cfg["mode"] == "last2" else 0.0),
        ).to(device)
        if cfg["mode"].startswith("lora"):
            attach_lora(model)
        model.load_state_dict(
            torch.load(home / cfg["checkpoint"], map_location="cpu", weights_only=True)
        )
        enc = tokens(rows, tok, maxlen=cfg["max_length"])
        p, _ = infer(model, enc, tok, device, batch=8)
        add(cfg["variant"], p, cfg["thresholds"])
        del model
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
    for cfg in json.loads((V3 / "results/locked_policies.json").read_text()):
        key = cfg["variant"]
        if key.startswith("bag_mean"):
            b = joblib.load(V3 / "models" / f"{key}.joblib")
            x = b["scaler"].transform(bags.reshape(-1, 768)).reshape(bags.shape)
            avg = (x * mask[:, :, None]).sum(1) / mask.sum(1)[:, None]
            p = b["classifier"].predict_proba(avg)
        else:
            scaler = joblib.load(V3 / "models" / f"{key}_scaler.joblib")
            x = scaler.transform(bags.reshape(-1, 768)).reshape(bags.shape).astype(np.float32)
            head = AttentionHead()
            head.load_state_dict(
                torch.load(V3 / "models" / f"{key}.pt", map_location="cpu", weights_only=True)
            )
            p, _ = predict(head, torch.tensor(x), torch.tensor(mask))
        add(key, p, cfg["thresholds"])
    dump("final_audit_results.json", results)
    print(
        "FINAL INTERNAL AUDIT COMPLETE",
        [(r["variant"], r["macro_f1"]) for r in results],
        flush=True,
    )
