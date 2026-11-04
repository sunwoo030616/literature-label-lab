import json, hashlib, csv
import numpy as np
import torch
from common import ROOT, V2, data, dump


def main():
    old = json.loads((V2 / "results/v1_preservation_sha256.json").read_text())
    changed = [
        p for p, h in old.items() if hashlib.sha256((ROOT / p).read_bytes()).hexdigest() != h
    ]
    assert not changed, changed
    S, Y, TUNE, CAL = data()
    assert set(TUNE).isdisjoint(CAL)
    assert len(TUNE) == 600 and len(CAL) == 400
    for split in ["train", "dev", "test"]:
        assert not {r["pmid"] for r in S[split]} & {r["pmid"] for r in S["audit"]}
        assert not {r["sha256"] for r in S[split]} & {r["sha256"] for r in S["audit"]}
    cfg = json.loads((V2 / "results/locked_finetune_policies.json").read_text())
    assert len(cfg) == 3
    original = torch.load(
        ROOT / "models/biomedbert/pytorch_model.bin", map_location="cpu", weights_only=True
    )
    checks = []
    for c in cfg:
        state = torch.load(V2 / c["checkpoint"], map_location="cpu", weights_only=True)
        entry = {
            "variant": c["variant"],
            "last_block_delta_l2": float(
                torch.linalg.vector_norm(
                    state["encoder.encoder.layer.11.output.dense.weight"]
                    - original["bert.encoder.layer.11.output.dense.weight"]
                )
            ),
            "first_block_delta_l2": float(
                torch.linalg.vector_norm(
                    state["encoder.encoder.layer.0.output.dense.weight"]
                    - original["bert.encoder.layer.0.output.dense.weight"]
                )
            ),
        }
        if c["mode"] == "last2":
            assert entry["last_block_delta_l2"] > 0 and entry["first_block_delta_l2"] == 0
        else:
            entry["adapter_B_l2_sum"] = sum(
                float(torch.linalg.vector_norm(v)) for k, v in state.items() if k.endswith(".B")
            )
            assert (
                entry["last_block_delta_l2"] == 0
                and entry["first_block_delta_l2"] == 0
                and entry["adapter_B_l2_sum"] > 0
            )
        compared = 0
        changed_base = 0
        for key, value in state.items():
            if not key.startswith("encoder.") or key.endswith(".A") or key.endswith(".B"):
                continue
            source = "bert." + key[len("encoder.") :].replace(".base.", ".")
            if source in original:
                compared += 1
                changed_base += int(not torch.equal(value, original[source]))
        entry["base_tensors_compared"] = compared
        entry["base_tensors_changed"] = changed_base
        if c["mode"].startswith("lora"):
            assert changed_base == 0 and compared > 100
        checks.append(entry)
    results = json.loads((V2 / "results/finetune_results.json").read_text())
    assert len(results) == 6
    for r in results:
        assert 0 <= r["macro_f1"] <= 1 and 0 <= r["micro_f1"] <= 1
    assert (V2 / "results/locked_finetune_policies.json").stat().st_mtime <= (
        V2 / "results/finetune_results.json"
    ).stat().st_mtime
    dump(
        "verification.json",
        {
            "v1_files_unchanged": len(old),
            "audit_disjoint": True,
            "actual_encoder_weight_updates": checks,
            "finetune_conditions_complete": 6,
        },
    )
    print("V2 verification passed", checks)


if __name__ == "__main__":
    main()
