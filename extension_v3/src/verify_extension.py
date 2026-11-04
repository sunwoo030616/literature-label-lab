"""Cross-artifact verification after the one-shot final audit, no model selection."""

import sys, json, hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V2 = ROOT / "followup_v2"
V3 = ROOT / "extension_v3"
sys.path.insert(0, str(V2 / "src"))
from common import data, train_ids, llm_path
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
    locked = read(V3 / "results/final_audit_locked_model_hashes.json")
    for p, h in locked.items():
        path = (
            llm_path() / Path(p).name if p.startswith("external_llm_snapshot/") else ROOT.parent / p
        )
        assert digest(path) == h, p
    result = read(V3 / "results/final_audit_results.json")
    assert len({r["variant"] for r in result}) == len(result)
    for row in result:
        p = np.load(V3 / "results" / f"final_audit_{row['variant']}_probs.npy")
        assert p.shape == (1000, 7) and np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()
        assert 0 <= row["macro_f1"] <= 1 and 0 <= row["micro_f1"] <= 1
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
        "locked_hashes_unchanged": len(locked),
        "final_audit_conditions": len(result),
        "lora512_weight_checks": weightchecks,
        "same40_llm_comparison": True,
        "fewshot_examples_train_only": True,
        "actual_service_checks": len(smoke),
        "extra_elapsed_minutes": end["elapsed_minutes"],
    }
    (V3 / "results/final_verification.json").write_text(json.dumps(report, indent=2))
    print("Extension verification passed", report)
