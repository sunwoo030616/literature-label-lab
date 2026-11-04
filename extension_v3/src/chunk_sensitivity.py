"""Post-hoc fixed-model chunk removal on reused audit, not causal rationale validation."""

import sys, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V3 = ROOT / "extension_v3"
sys.path.insert(0, str(ROOT / "followup_v2/src"))
from common import data, metrics, LABELS
from bag_model import AttentionHead, predict
import numpy as np, torch, joblib


def main():
    torch.set_num_threads(4)
    S, Y, _, _ = data()
    z = np.load(V3 / "data/bags.npz")
    offset = sum(len(S[k]) for k in ["train", "dev", "test"])
    raw, mask = z["bags"][offset:], z["mask"][offset:]
    assert len(raw) == len(S["audit"])
    cfg = next(
        c
        for c in json.loads((V3 / "results/locked_policies.json").read_text())
        if c["variant"] == "attention_seed17"
    )
    scaler = joblib.load(V3 / "models/attention_seed17_scaler.joblib")
    x = scaler.transform(raw.reshape(-1, 768)).reshape(raw.shape).astype(np.float32)
    model = AttentionHead()
    model.load_state_dict(
        torch.load(V3 / "models/attention_seed17.pt", map_location="cpu", weights_only=True)
    )
    p, att = predict(model, torch.tensor(x), torch.tensor(mask))
    ix = np.flatnonzero(mask.sum(1) >= 2)
    pred = p >= np.array(cfg["thresholds"])
    # Aggregate across labels predicted by this model, or all labels if none predicted.
    labelweights = pred.astype(float)
    labelweights[labelweights.sum(1) == 0] = 1
    importance = (att * labelweights[:, :, None]).sum(1) / labelweights.sum(1)[:, None]
    highest = np.where(mask, importance, -np.inf).argmax(1)
    lowest = np.where(mask, importance, np.inf).argmin(1)
    rng = np.random.default_rng(20261104)
    random = np.array([rng.choice(np.flatnonzero(m)) for m in mask])
    result = {
        "variant": "attention_seed17",
        "split": "audit_reused",
        "eligible_documents": len(ix),
        "total_documents": len(mask),
        "selection": "all documents with at least two valid chunks; label-averaged attention over predicted topics, all topics fallback",
        "baseline": metrics(Y["audit"][ix], p[ix], cfg["thresholds"]),
        "conditions": [],
        "limitations": "Fixed-model masking sensitivity only. Removal alters context and renormalizes attention. It does not validate human rationales, semantic entailment, or causality; one random draw per document, one trained model.",
    }
    detailed = []
    for name, indices in [
        ("highest_attention", highest),
        ("lowest_attention", lowest),
        ("random", random),
    ]:
        changed = mask[ix].copy()
        changed[np.arange(len(ix)), indices[ix]] = False
        pp, _ = predict(model, torch.tensor(x[ix]), torch.tensor(changed))
        delta = pp - p[ix]
        result["conditions"].append(
            {
                "removed": name,
                **metrics(Y["audit"][ix], pp, cfg["thresholds"]),
                "mean_absolute_probability_change": float(np.abs(delta).mean()),
                "any_label_flip_fraction": float(
                    ((pp >= np.array(cfg["thresholds"])) != pred[ix]).any(1).mean()
                ),
                "mean_predicted_label_probability_drop": float(
                    np.mean(((p[ix] - pp) * labelweights[ix]).sum(1) / labelweights[ix].sum(1))
                ),
            }
        )
        if name == "highest_attention":
            windows = json.loads((V3 / "results/bag_windows.json").read_text())[offset:]
            for local, global_i in enumerate(ix[:10]):
                detailed.append(
                    {
                        "pmid": S["audit"][global_i]["pmid"],
                        "title": S["audit"][global_i]["title"],
                        "gold": S["audit"][global_i]["labels"],
                        "predicted": [l for j, l in enumerate(LABELS) if pred[global_i, j]],
                        "removed_chunk_index": int(indices[global_i]),
                        "removed_tokenizer_decoded_window": windows[global_i][indices[global_i]],
                        "before_probabilities": p[global_i].tolist(),
                        "after_probabilities": pp[local].tolist(),
                        "selection": "first ten eligible documents in locked audit order, not manually curated",
                    }
                )
    result["examples"] = detailed
    (V3 / "results/chunk_sensitivity.json").write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
            default=lambda x: x.tolist() if isinstance(x, np.ndarray) else float(x),
        )
    )
    print(
        [(r["removed"], r["macro_f1"], r["any_label_flip_fraction"]) for r in result["conditions"]]
    )


if __name__ == "__main__":
    main()
