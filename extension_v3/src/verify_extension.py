"""Cross-artifact verification after the one-shot final audit, no model selection."""

import sys, json, hashlib, csv, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V2 = ROOT / "followup_v2"
V3 = ROOT / "extension_v3"
sys.path.insert(0, str(V2 / "src"))
from common import data, train_ids, llm_path, yof
import numpy as np, torch


def read(p):
    return json.loads(p.read_text())


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


if __name__ == "__main__":
    old = read(V2 / "results/v1_preservation_sha256.json")
    assert all(digest(ROOT / p) == h for p, h in old.items())
    S, Y, _, _ = data()
    fresh = read(V3 / "data/final_audit_documents.json")
    assert len(fresh) == 1000
    for k, rows in S.items():
        assert not {r["pmid"] for r in fresh} & {r["pmid"] for r in rows}
        assert not {r["sha256"] for r in fresh} & {r["sha256"] for r in rows}
    doi_by_id = {}
    for split in ["train", "dev", "test"]:
        for row in csv.DictReader((ROOT / "data/raw" / f"{split}.csv").open()):
            doi_by_id[row["pmid"]] = re.sub(
                r"^https?://(?:dx\.)?doi\.org/", "", row["doi"].strip().lower()
            )
    existing_doi = {
        doi_by_id[r["pmid"]] for rows in S.values() for r in rows if doi_by_id[r["pmid"]]
    }
    final_doi = [doi_by_id[r["pmid"]] for r in fresh if doi_by_id[r["pmid"]]]
    assert not existing_doi & set(final_doi)
    assert len(final_doi) == len(set(final_doi))
    locked = read(V3 / "results/final_audit_locked_model_hashes.json")
    for p, h in locked.items():
        path = (
            llm_path() / Path(p).name if p.startswith("external_llm_snapshot/") else ROOT.parent / p
        )
        assert digest(path) == h, p
    result = read(V3 / "results/final_audit_results.json")
    assert len({r["variant"] for r in result}) == len(result)
    expected = {"v1_frozen256", "v1_tfidf", "tuned_frozen256", "mean512", "cls512"}
    old_policies = read(ROOT / "results/locked_policies.json")
    thresholds = {
        "v1_frozen256": next(
            c["thresholds"] for c in old_policies if c["seed"] == 17 and c["method"] == "biomedbert"
        ),
        "v1_tfidf": next(
            c["thresholds"] for c in old_policies if c["seed"] == 17 and c["method"] == "tfidf"
        ),
        "tuned_frozen256": read(V2 / "results/locked_classifier_policy.json")["thresholds"],
    }
    thresholds.update(
        {c["variant"]: c["thresholds"] for c in read(V2 / "results/locked_input_policies.json")}
    )
    for policy in [
        V2 / "results/locked_finetune_policies.json",
        V3 / "results/locked_finetune_policies.json",
        V3 / "results/locked_policies.json",
        V3 / "results/locked_strong_tfidf_policies.json",
        V3 / "results/locked_threshold_cv_policies.json",
    ]:
        expected.update(c["variant"] for c in read(policy))
        thresholds.update({c["variant"]: c["thresholds"] for c in read(policy)})
    replica = V3 / "results/locked_replica512_policy.json"
    if replica.exists():
        expected.add(read(replica)["variant"])
        thresholds[read(replica)["variant"]] = read(replica)["thresholds"]
    assert {r["variant"] for r in result} == expected
    uniform_checks = []
    for seed in [17, 42, 73]:
        weights = torch.load(
            V3 / "models" / f"uniform_attention_seed{seed}.pt",
            map_location="cpu",
            weights_only=True,
        )
        assert torch.count_nonzero(weights["query"]).item() == 0
        uniform_checks.append(seed)
    for cfg in read(V3 / "results/locked_threshold_cv_policies.json"):
        source = np.load(V3 / "results" / f"final_audit_{cfg['source_variant']}_probs.npy")
        changed = np.load(V3 / "results" / f"final_audit_{cfg['variant']}_probs.npy")
        assert np.array_equal(source, changed)
    fresh_y = yof(fresh)
    for row in result:
        p = np.load(V3 / "results" / f"final_audit_{row['variant']}_probs.npy")
        assert p.shape == (1000, 7) and np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()
        assert 0 <= row["macro_f1"] <= 1 and 0 <= row["micro_f1"] <= 1
        predicted = p >= np.array(thresholds[row["variant"]])
        tp = (fresh_y * predicted).sum(0)
        fp = ((1 - fresh_y) * predicted).sum(0)
        fn = (fresh_y * (1 - predicted)).sum(0)
        denominator = 2 * tp + fp + fn
        label_f1 = np.divide(
            2 * tp, denominator, out=np.zeros(7, dtype=float), where=denominator > 0
        )
        assert abs(float(label_f1.mean()) - row["macro_f1"]) < 1e-10
        assert abs(float(2 * tp.sum() / denominator.sum()) - row["micro_f1"]) < 1e-10
    configs = read(V3 / "results/locked_finetune_policies.json")
    optional = V3 / "results/locked_replica512_policy.json"
    if optional.exists():
        configs.append(read(optional))
    original = torch.load(
        ROOT / "models/biomedbert/pytorch_model.bin", map_location="cpu", weights_only=True
    )
    weightchecks = []
    for cfg in configs:
        path = V3 / cfg["checkpoint"]
        assert digest(path) == cfg["checkpoint_sha256"]
        state = torch.load(path, map_location="cpu", weights_only=True)
        compared = 0
        for key, value in state.items():
            if not key.startswith("encoder.") or key.endswith(".A") or key.endswith(".B"):
                continue
            source = "bert." + key[len("encoder.") :].replace(".base.", ".")
            if source in original:
                assert torch.equal(value, original[source]), key
                compared += 1
        assert compared > 100
        bnorm = sum(float(v.norm()) for k, v in state.items() if k.endswith(".B"))
        assert bnorm > 0 and cfg["max_length"] == 512
        weightchecks.append(
            {
                "variant": cfg["variant"],
                "base_tensors_exactly_unchanged": compared,
                "adapter_B_l2_sum": bnorm,
                "checkpoint_sha256_verified": True,
            }
        )
        del state
    zero = read(V2 / "results/llm_raw_results.json")
    few = read(V3 / "results/llm_fewshot_raw.json")
    assert len(zero) == len(few) == 40 and [r["pmid"] for r in zero] == [r["pmid"] for r in few]
    trainset = {S["train"][int(i)]["pmid"] for i in train_ids(17)}
    examples = read(V3 / "results/llm_training_examples.json")
    assert {r["pmid"] for r in examples} <= trainset
    assert not {r["pmid"] for r in examples} & {r["pmid"] for r in few}
    smoke = read(V3 / "results/service_smoke.json")
    assert len(smoke) == 7
    start = read(V3 / "results/extra_phase_start.json")
    end = read(V3 / "results/extra_phase_end.json")
    assert end["elapsed_minutes"] > 60
    report = {
        "v1_files_unchanged": len(old),
        "final_internal_documents": len(fresh),
        "final_audit_disjoint": True,
        "final_audit_doi_overlap": 0,
        "final_audit_missing_doi": 1000 - len(final_doi),
        "uniform_heads_query_exactly_zero_seeds": uniform_checks,
        "locked_hashes_unchanged": len(locked),
        "final_audit_conditions": len(result),
        "fresh_metrics_recomputed_from_saved_probabilities_and_locked_thresholds": True,
        "lora512_weight_checks": weightchecks,
        "same40_llm_comparison": True,
        "fewshot_examples_train_only": True,
        "actual_service_checks": len(smoke),
        "extra_elapsed_minutes": end["elapsed_minutes"],
    }
    (V3 / "results/final_verification.json").write_text(json.dumps(report, indent=2))
    print("Extension verification passed", report)
