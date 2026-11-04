"""Small adapter/head exports; base model still required, original checkpoints intact."""

import sys, json, hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V2 = ROOT / "followup_v2"
V3 = ROOT / "extension_v3"
sys.path.insert(0, str(V2 / "src"))
import torch
from safetensors.torch import save_file, load_file

if __name__ == "__main__":
    configs = [
        (V2, c)
        for c in json.loads((V2 / "results/locked_finetune_policies.json").read_text())
        if c["mode"].startswith("lora")
    ] + [
        (V3, c)
        for c in json.loads((V3 / "results/locked_finetune_policies.json").read_text())
        if c["mode"].startswith("lora")
    ]
    out = []
    for home, cfg in configs:
        checkpoint = home / cfg["checkpoint"]
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        adapters = {
            k: v.contiguous()
            for k, v in state.items()
            if k.endswith(".A") or k.endswith(".B") or k.startswith("head.")
        }
        assert sum(v.numel() for v in adapters.values()) == 300295
        target = V3 / "models" / f"{cfg['variant']}_adapter.safetensors"
        save_file(adapters, str(target))
        restored = load_file(str(target))
        assert set(restored) == set(adapters) and all(
            torch.equal(restored[k], v) for k, v in adapters.items()
        )
        manifest = {
            "variant": cfg["variant"],
            "base_model": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
            "base_revision": "d673b8835373c6fa116d6d8006b33d48734e305d",
            "base_weights_sha256": hashlib.sha256(
                (ROOT / "models/biomedbert/pytorch_model.bin").read_bytes()
            ).hexdigest(),
            "base_model_required": True,
            "rank": 8,
            "alpha": 16,
            "dropout": 0.05,
            "max_length": cfg["max_length"],
            "thresholds": cfg["thresholds"],
            "trained_parameters": 300295,
            "adapter_bytes": target.stat().st_size,
            "full_checkpoint_bytes": checkpoint.stat().st_size,
            "tensor_roundtrip_exact": True,
        }
        (V3 / "models" / f"{cfg['variant']}_adapter.json").write_text(
            json.dumps(manifest, indent=2)
        )
        out.append(manifest)
    (V3 / "results/adapter_exports.json").write_text(json.dumps(out, indent=2))
    print([(r["variant"], r["adapter_bytes"]) for r in out])
