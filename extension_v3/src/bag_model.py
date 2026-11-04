"""Multi-label attention over token-window bags, not a T-MDML reproduction."""

import sys, json, time, gc, hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V3 = ROOT / "extension_v3"
sys.path.insert(0, str(ROOT / "followup_v2/src"))
from common import data, train_ids, LABELS, thresholds, metrics
import numpy as np
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


def dump(name, obj):
    (V3 / "results" / name).write_text(
        json.dumps(
            obj,
            ensure_ascii=False,
            indent=2,
            default=lambda x: x.tolist() if isinstance(x, np.ndarray) else float(x),
        )
    )


class AttentionHead(nn.Module):
    def __init__(self, d=768):
        super().__init__()
        self.query = nn.Parameter(torch.randn(7, d) * 0.01)
        self.classifier = nn.Parameter(torch.randn(7, d) * 0.01)
        self.bias = nn.Parameter(torch.zeros(7))
        self.dropout = nn.Dropout(0.1)

    def make_uniform(self):
        """Matched control: same classifier/training, but no learned chunk weighting."""
        with torch.no_grad():
            self.query.zero_()
        self.query.requires_grad_(False)
        return self

    def forward(self, x, mask):
        scores = torch.einsum("bkd,ld->blk", x, self.query) / np.sqrt(x.shape[-1])
        scores = scores.masked_fill(~mask[:, None, :], -1e4)
        attention = torch.softmax(scores, -1)
        contexts = torch.einsum("blk,bkd->bld", attention, self.dropout(x))
        logits = (contexts * self.classifier[None, :, :]).sum(-1) + self.bias
        return logits, attention


def bert_chunk(content_ids, tok):
    """Pinned single-sequence BERT layout; tested against tokenizer-native encoding."""
    assert len(content_ids) <= 126
    assert tok.cls_token_id is not None and tok.sep_token_id is not None
    return [tok.cls_token_id] + list(content_ids) + [tok.sep_token_id]


def make_bags(rows, tok, model, device, profile_path=None):
    allchunks = []
    mapping = []
    texts = []
    for row in rows:
        ids = tok.encode(row["text"], add_special_tokens=False)
        chunks = [ids[i : i + 126] for i in range(0, len(ids), 126)] or [[]]
        selected = np.linspace(0, len(chunks) - 1, min(4, len(chunks)), dtype=int)
        mapping.append((len(allchunks), len(selected)))
        chosen = [chunks[i] for i in selected]
        allchunks.extend([bert_chunk(c, tok) for c in chosen])
        texts.append([tok.decode(c) for c in chosen])
    embeddings = []
    start = time.perf_counter()
    max_driver_mb_observed = 0
    with torch.inference_mode():
        for i in range(0, len(allchunks), 32):
            x = tok.pad(
                [{"input_ids": c, "attention_mask": [1] * len(c)} for c in allchunks[i : i + 32]],
                padding=True,
                return_tensors="pt",
            ).to(device)
            h = model(**x).last_hidden_state
            m = x["attention_mask"].unsqueeze(-1)
            embeddings.append(((h * m).sum(1) / m.sum(1).clamp(min=1)).cpu().numpy())
            if device == "mps":
                max_driver_mb_observed = max(
                    max_driver_mb_observed, torch.mps.driver_allocated_memory() / 2**20
                )
                if (i // 32 + 1) % 20 == 0:
                    torch.mps.synchronize()
                    torch.mps.empty_cache()
            if i % 512 == 0:
                print(
                    "bag chunks",
                    i,
                    "/",
                    len(allchunks),
                    "seconds",
                    round(time.perf_counter() - start, 1),
                    flush=True,
                )
    embedding_loop_seconds = time.perf_counter() - start
    e = np.concatenate(embeddings)
    out = np.zeros((len(rows), 4, 768), np.float32)
    mask = np.zeros((len(rows), 4), bool)
    for i, (start, count) in enumerate(mapping):
        out[i, :count] = e[start : start + count]
        mask[i, :count] = True
    if profile_path is not None:
        profile_path.write_text(
            json.dumps(
                {
                    "device": device,
                    "documents": len(rows),
                    "windows": len(allchunks),
                    "batch": 32,
                    "max_window_tokens": 128,
                    "mps_cache_release_every_batches": 20,
                    "embedding_loop_seconds": embedding_loop_seconds,
                    "max_driver_mb_observed_after_batches": max_driver_mb_observed,
                    "note": "Driver memory includes allocator caches; sampled after each batch, not a true instantaneous peak. Host desktop load is not controlled. Tokenization and feature compression are outside the timed embedding loop.",
                },
                indent=2,
            )
        )
    return out, mask, texts


def predict(model, x, mask):
    model.eval()
    pp = []
    att = []
    with torch.inference_mode():
        for i in range(0, len(x), 128):
            logits, a = model(x[i : i + 128], mask[i : i + 128])
            pp.append(torch.sigmoid(logits).numpy())
            att.append(a.numpy())
    return np.concatenate(pp), np.concatenate(att)


def train_head(x, mask, y, xd, md, yd, graph, lam, seed, uniform=False):
    torch.manual_seed(seed)
    model = AttentionHead()
    if uniform:
        model.make_uniform()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    history = []
    best = -1
    snapshot = None
    policy = None
    for epoch in range(1, 31):
        model.train()
        order = np.random.default_rng(seed + epoch).permutation(len(y))
        total = 0
        for start in range(0, len(order), 64):
            ix = order[start : start + 64]
            logits, _ = model(x[ix], mask[ix])
            loss = nn.functional.binary_cross_entropy_with_logits(logits, y[ix])
            diff = model.classifier[:, None, :] - model.classifier[None, :, :]
            reg = (graph * diff.square().mean(-1)).sum() / graph.sum().clamp(min=1)
            loss = loss + lam * reg
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            opt.step()
            total += loss.item() * len(ix)
        p, _ = predict(model, xd, md)
        t = thresholds(yd, p)
        score = f1_score(yd, p >= t, average="macro", zero_division=0)
        history.append({"epoch": epoch, "loss": total / len(y), "tune_macro_f1": score})
        if score > best + 1e-12:
            best = score
            snapshot = {k: v.detach().clone() for k, v in model.state_dict().items()}
            policy = {"epoch": epoch, "thresholds": t, "tune_macro_f1": score, "lambda": lam}
    model.load_state_dict(snapshot)
    return model, policy, history


def main():
    torch.set_num_threads(4)
    S, Y, TUNE, CAL = data()
    cache = V3 / "data/bags.npz"
    allrows = sum(S.values(), [])
    provenance = {
        "schema": 1,
        "ordered_document_sha256": hashlib.sha256(
            "\n".join(r["sha256"] for r in allrows).encode()
        ).hexdigest(),
        "base_weights_sha256": hashlib.sha256(
            (ROOT / "models/biomedbert/pytorch_model.bin").read_bytes()
        ).hexdigest(),
        "content_tokens_per_window": 126,
        "max_windows": 4,
        "selection": "uniform indices including endpoints",
        "pooling": "attention-mask mean including special tokens",
    }
    cache_manifest = V3 / "data/bag_cache_manifest.json"
    if cache.exists():
        assert cache_manifest.exists(), "Cache provenance missing; regenerate features explicitly"
        assert json.loads(cache_manifest.read_text()) == provenance, "Stale bag cache"
        z = np.load(cache)
        bags, mask = z["bags"], z["mask"]
    else:
        tok = AutoTokenizer.from_pretrained(ROOT / "models/biomedbert", local_files_only=True)
        model = AutoModel.from_pretrained(ROOT / "models/biomedbert", local_files_only=True).eval()
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        model.to(device)
        encoding_start = time.perf_counter()
        bags, mask, texts = make_bags(
            allrows, tok, model, device, V3 / "results/bag_encoding_profile.json"
        )
        profile = json.loads((V3 / "results/bag_encoding_profile.json").read_text())
        profile["tokenization_and_embedding_seconds"] = time.perf_counter() - encoding_start
        (V3 / "results/bag_encoding_profile.json").write_text(json.dumps(profile, indent=2))
        np.savez_compressed(cache, bags=bags, mask=mask)
        cache_manifest.write_text(json.dumps(provenance, indent=2))
        (V3 / "data/bag_windows.json").write_text(json.dumps(texts, ensure_ascii=False))
        del model
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
    dump(
        "bag_feature_provenance.json",
        {
            **provenance,
            "shape": list(bags.shape),
            "valid_windows": int(mask.sum()),
            "dtype": str(bags.dtype),
        },
    )
    xs = {}
    ms = {}
    start = 0
    for k, v in S.items():
        xs[k] = bags[start : start + len(v)]
        ms[k] = mask[start : start + len(v)]
        start += len(v)
    locked = []
    objects = {}
    for seed in [17, 42, 73]:
        ids = train_ids(seed)
        scaler = StandardScaler().fit(xs["train"][ids][ms["train"][ids]])
        xx = {
            k: scaler.transform(v.reshape(-1, 768)).reshape(v.shape).astype(np.float32)
            for k, v in xs.items()
        }
        tx = {k: torch.tensor(v) for k, v in xx.items()}
        tm = {k: torch.tensor(v) for k, v in ms.items()}
        y = Y["train"][ids]
        cy = y.T @ y
        g = cy / np.sqrt(np.maximum(1, np.outer(y.sum(0), y.sum(0))))
        np.fill_diagonal(g, 0)
        perm = np.random.default_rng(991).permutation(7)
        graphs = {
            "attention": np.zeros_like(g),
            "label_graph": g,
            "random_graph": g[np.ix_(perm, perm)],
            "uniform_attention": np.zeros_like(g),
        }
        zero_fit = None
        # Shared bag features and train-only scaler; each condition has a matched lambda selection budget.
        for name, graph in graphs.items():
            best = None
            bestmodel = None
            histories = []
            uniform = name == "uniform_attention"
            for lam in ([0] if name in ["attention", "uniform_attention"] else [0, 0.01, 0.1]):
                if lam == 0 and zero_fit is not None and not uniform:
                    # With zero penalty this is exactly the same seeded objective.
                    # Reuse it rather than repeating an identical training run.
                    model, policy, history = zero_fit
                else:
                    model, policy, history = train_head(
                        tx["train"][ids],
                        tm["train"][ids],
                        torch.tensor(y, dtype=torch.float32),
                        tx["dev"][TUNE],
                        tm["dev"][TUNE],
                        Y["dev"][TUNE],
                        torch.tensor(graph, dtype=torch.float32),
                        lam,
                        seed,
                        uniform=uniform,
                    )
                    if lam == 0 and not uniform:
                        zero_fit = (model, policy, history)
                histories.append({"lambda": lam, "history": history})
                if best is None or policy["tune_macro_f1"] > best["tune_macro_f1"] + 1e-12:
                    best = policy
                    bestmodel = model
            key = f"{name}_seed{seed}"
            torch.save(bestmodel.state_dict(), V3 / "models" / f"{key}.pt")
            __import__("joblib").dump(scaler, V3 / "models" / f"{key}_scaler.joblib")
            locked.append({"variant": key, "seed": seed, **best})
            objects[key] = bestmodel
            dump(f"{key}_history.json", histories)
            print("bag tuned", key, best["tune_macro_f1"], flush=True)
        # Mean pooling on exactly the same bag features, C selected on tune.
        avg = {k: (v * ms[k][:, :, None]).sum(1) / ms[k].sum(1)[:, None] for k, v in xx.items()}
        best = None
        clfbest = None
        for C in [0.001, 0.01, 0.1, 1, 10]:
            clf = OneVsRestClassifier(
                LogisticRegression(C=C, max_iter=1500, solver="liblinear")
            ).fit(avg["train"][ids], y)
            p = clf.predict_proba(avg["dev"][TUNE])
            t = thresholds(Y["dev"][TUNE], p)
            score = f1_score(Y["dev"][TUNE], p >= t, average="macro", zero_division=0)
            if best is None or score > best["tune_macro_f1"] + 1e-12:
                best = {"C": C, "thresholds": t, "tune_macro_f1": score}
                clfbest = clf
        key = f"bag_mean_seed{seed}"
        locked.append({"variant": key, "seed": seed, **best})
        objects[key] = clfbest
        __import__("joblib").dump(
            {"scaler": scaler, "classifier": clfbest}, V3 / "models" / f"{key}.joblib"
        )
    dump("locked_policies.json", locked)
    results = []
    for cfg in locked:
        seed = cfg["seed"]
        ids = train_ids(seed)
        scaler = StandardScaler().fit(xs["train"][ids][ms["train"][ids]])
        for split in ["test", "audit"]:
            x = (
                scaler.transform(xs[split].reshape(-1, 768))
                .reshape(xs[split].shape)
                .astype(np.float32)
            )
            key = cfg["variant"]
            model = objects[key]
            if key.startswith("bag_mean"):
                avg = (x * ms[split][:, :, None]).sum(1) / ms[split].sum(1)[:, None]
                p = model.predict_proba(avg)
            else:
                p, att = predict(model, torch.tensor(x), torch.tensor(ms[split]))
                np.save(V3 / "results" / f"{key}_{split}_attention.npy", att)
            np.save(V3 / "results" / f"{key}_{split}_probs.npy", p)
            results.append(
                {"variant": key, "split": split, **metrics(Y[split], p, cfg["thresholds"])}
            )
    dump("bag_results.json", results)
    print("BAG COMPLETE", [(r["variant"], r["split"], r["macro_f1"]) for r in results], flush=True)


if __name__ == "__main__":
    main()
