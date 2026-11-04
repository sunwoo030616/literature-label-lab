"""Descriptive mechanism checks for locked bag models; no retuning."""

import sys, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V3 = ROOT / "extension_v3"
sys.path.insert(0, str(ROOT / "followup_v2/src"))
from common import data, train_ids
from bag_model import AttentionHead, predict
import numpy as np, torch, joblib


def main():
    torch.set_num_threads(4)
    S, Y, _, _ = data()
    z = np.load(V3 / "data/bags.npz")
    bags, mask = z["bags"], z["mask"]
    trainx = bags[: len(S["train"])]
    trainmask = mask[: len(S["train"])]
    offset = sum(len(S[k]) for k in ["train", "dev", "test"])
    auditx, auditmask = bags[offset:], mask[offset:]
    policies = json.loads((V3 / "results/locked_policies.json").read_text())
    out = []
    for cfg in policies:
        name = cfg["variant"]
        if name.startswith("bag_mean"):
            continue
        scaler = joblib.load(V3 / "models" / f"{name}_scaler.joblib")
        x = scaler.transform(auditx.reshape(-1, 768)).reshape(auditx.shape).astype(np.float32)
        head = AttentionHead()
        head.load_state_dict(
            torch.load(V3 / "models" / f"{name}.pt", map_location="cpu", weights_only=True)
        )
        pp, att = predict(head, torch.tensor(x), torch.tensor(auditmask))
        eligible = auditmask.sum(1) > 1
        entropy = -(att * np.log(np.clip(att, 1e-12, 1))).sum(-1)
        normalized = entropy[eligible] / np.log(auditmask.sum(1)[eligible, None])
        ids = train_ids(cfg["seed"])
        y = Y["train"][ids]
        graph = (y.T @ y) / np.sqrt(np.maximum(1, np.outer(y.sum(0), y.sum(0))))
        np.fill_diagonal(graph, 0)
        if name.startswith("random_graph"):
            perm = np.random.default_rng(991).permutation(7)
            graph = graph[np.ix_(perm, perm)]
        if name.startswith("attention") or name.startswith("uniform_attention"):
            graph = np.zeros_like(graph)
        weights = head.classifier.detach().numpy()
        reg = float(
            (graph * ((weights[:, None] - weights[None, :]) ** 2).mean(-1)).sum()
            / max(1, graph.sum())
        )
        xx = (
            scaler.transform(trainx[ids].reshape(-1, 768))
            .reshape(trainx[ids].shape)
            .astype(np.float32)
        )
        trainp, _ = predict(head, torch.tensor(xx), torch.tensor(trainmask[ids]))
        pc = np.clip(trainp, 1e-7, 1 - 1e-7)
        bce = float(-(y * np.log(pc) + (1 - y) * np.log(1 - pc)).mean())
        out.append(
            {
                "variant": name,
                "selected_lambda": cfg["lambda"],
                "selected_epoch": cfg["epoch"],
                "mean_normalized_attention_entropy_audit": float(normalized.mean()),
                "per_label_normalized_attention_entropy_audit": normalized.mean(0).tolist(),
                "query_l2_per_label": head.query.detach().norm(dim=1).tolist(),
                "selected_model_train_eval_bce": bce,
                "unweighted_graph_penalty": reg,
                "weighted_graph_penalty": cfg["lambda"] * reg,
                "weighted_penalty_to_train_eval_bce_ratio": cfg["lambda"] * reg / bce,
            }
        )
    report = {
        "models": out,
        "interpretation": "Entropy near 1 means nearly uniform attention, near 0 means concentrated. Descriptive association only. Graph/BCE ratio uses selected-model eval mode, not exact stochastic training loss; a small penalty is consistent with weak influence but not proof of why performance changed. Lambda0 means the selected model did not use graph regularization.",
    }
    (V3 / "results/bag_mechanism_analysis.json").write_text(json.dumps(report, indent=2))
    print(
        [
            (
                r["variant"],
                r["selected_lambda"],
                round(r["mean_normalized_attention_entropy_audit"], 4),
            )
            for r in out
        ]
    )


if __name__ == "__main__":
    main()
