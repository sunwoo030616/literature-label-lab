"""Evaluation-only LoRA merge. Original checkpoint files are never modified."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "followup_v2/src"))
import torch
from finetune import LoRALinear


def merge_for_inference(model):
    if model.training:
        raise ValueError(
            "Set eval() before merging: training dropout cannot be folded into a weight."
        )
    count = 0
    with torch.no_grad():
        for layer in model.encoder.encoder.layer:
            for name in ["query", "value"]:
                adapter = getattr(layer.attention.self, name)
                if isinstance(adapter, LoRALinear):
                    adapter.base.weight.add_((adapter.B @ adapter.A) * adapter.scale)
                    setattr(layer.attention.self, name, adapter.base)
                    count += 1
    return count
