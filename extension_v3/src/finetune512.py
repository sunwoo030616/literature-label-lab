"""Actual supervised partial/LoRA BiomedBERT fine tuning, local only."""

import os, json, time, math, gc, random
import numpy as np
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "followup_v2/src"))
from common import ROOT, LABELS, data, train_ids, thresholds, metrics, sha

V2 = ROOT / "extension_v3"


def dump(name, obj):
    (V2 / "results" / name).write_text(
        json.dumps(
            obj,
            ensure_ascii=False,
            indent=2,
            default=lambda a: a.tolist() if isinstance(a, np.ndarray) else float(a),
        )
    )


class LoRALinear(nn.Module):
    def __init__(self, base, rank=8, alpha=16):
        super().__init__()
        self.base = base
        self.A = nn.Parameter(
            torch.empty(rank, base.in_features, device=base.weight.device, dtype=base.weight.dtype)
        )
        self.B = nn.Parameter(
            torch.zeros(base.out_features, rank, device=base.weight.device, dtype=base.weight.dtype)
        )
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scale = alpha / rank
        self.dropout = nn.Dropout(0.05)
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.base(x) + (self.dropout(x) @ self.A.T @ self.B.T) * self.scale


def attach_lora(model):
    for p in model.encoder.parameters():
        p.requires_grad = False
    for layer in model.encoder.encoder.layer:
        layer.attention.self.query = LoRALinear(layer.attention.self.query)
        layer.attention.self.value = LoRALinear(layer.attention.self.value)
    return model


class TopicModel(nn.Module):
    def __init__(self, attention_backend="sdpa", attention_dropout=0.0):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
            ROOT / "models/biomedbert",
            local_files_only=True,
            attn_implementation=attention_backend,
            attention_probs_dropout_prob=attention_dropout,
        )
        self.dropout = nn.Dropout(0.1)
        self.head = nn.Linear(768, len(LABELS))

    def forward(self, **x):
        h = self.encoder(**x).last_hidden_state
        mask = x["attention_mask"].unsqueeze(-1)
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.head(self.dropout(pooled))


def tokens(rows, tok, maxlen=512):
    return tok([r["text"] for r in rows], truncation=True, max_length=maxlen, padding=False)


def batch_tokens(enc, ids, tok, device):
    return tok.pad(
        [{k: enc[k][int(i)] for k in enc.keys()} for i in ids],
        padding="max_length",
        max_length=512,
        return_tensors="pt",
    ).to(device)


def infer(model, enc, tok, device, batch=16):
    model.eval()
    out = []
    start = time.perf_counter()
    with torch.inference_mode():
        for i in range(0, len(enc["input_ids"]), batch):
            out.append(
                torch.sigmoid(
                    model(
                        **batch_tokens(
                            enc, np.arange(i, min(i + batch, len(enc["input_ids"]))), tok, device
                        )
                    )
                )
                .cpu()
                .numpy()
            )
    return np.concatenate(out), time.perf_counter() - start


def train_one(mode, seed, trainenc, devenc, y, tune, tok, device):
    name = f"{mode}_seed{seed}"
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == "mps":
        torch.mps.manual_seed(seed)
    model = TopicModel(
        attention_backend=("eager" if mode == "last2" else "sdpa"),
        attention_dropout=(0.1 if mode == "last2" else 0.0),
    ).to(device)
    if mode == "last2":
        for p in model.encoder.parameters():
            p.requires_grad = False
        for layer in model.encoder.encoder.layer[-2:]:
            for p in layer.parameters():
                p.requires_grad = True
        # pooler is unused; encoder dropout remains active during training, consistently with full mode.
    if mode.startswith("lora"):
        attach_lora(model)
    ids = train_ids(seed)
    nparams = sum(p.numel() for p in model.parameters() if p.requires_grad)
    groups = [
        {
            "params": [p for p in model.encoder.parameters() if p.requires_grad],
            "lr": (5e-4 if mode.startswith("lora") else 2e-5),
        },
        {"params": model.head.parameters(), "lr": 1e-3},
    ]
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    physical_batch = 4 if mode.startswith("lora") else 8
    accum = 4 if mode.startswith("lora") else 2
    steps_per_epoch = math.ceil(len(ids) / (physical_batch * accum))
    total = 3 * steps_per_epoch
    warmup = max(1, int(0.1 * total))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / warmup if s < warmup else max(0, (total - s) / (total - warmup))
    )
    history = []
    best = -1
    bestpolicy = None
    started = time.perf_counter()
    global_step = 0
    for epoch in range(1, 2 if mode.startswith("lora") else 4):
        model.train()
        order = np.random.default_rng(seed + epoch).permutation(ids)
        loss_sum = 0
        seen = 0
        opt.zero_grad(set_to_none=True)
        for step, start in enumerate(range(0, len(order), physical_batch)):
            ix = order[start : start + physical_batch]
            x = batch_tokens(trainenc, ix, tok, device)
            target = torch.tensor(y[ix], dtype=torch.float32, device=device)
            logits = model(**x)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, target)
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite loss")
            (loss / accum).backward()
            loss_sum += loss.item() * len(ix)
            seen += len(ix)
            if (step + 1) % accum == 0 or start + physical_batch >= len(order):
                nn.utils.clip_grad_norm_(model.parameters(), 1)
                opt.step()
                scheduler.step()
                opt.zero_grad(set_to_none=True)
                global_step += 1
                if device == "mps" and global_step % 25 == 0:
                    torch.mps.synchronize()
                    torch.mps.empty_cache()
                    print(
                        "MPS cache release",
                        global_step,
                        "driver_mb",
                        round(torch.mps.driver_allocated_memory() / 2**20, 1),
                        flush=True,
                    )
            if (step + 1) % 50 == 0:
                print(
                    name,
                    "epoch",
                    epoch,
                    "batch",
                    step + 1,
                    "loss",
                    round(loss_sum / seen, 5),
                    "elapsed",
                    round(time.perf_counter() - started, 1),
                    flush=True,
                )
        probs, evalsec = infer(model, devenc, tok, device)
        t = thresholds(YDEV[tune], probs[tune])
        m = metrics(YDEV[tune], probs[tune], t)
        history.append(
            {
                "epoch": epoch,
                "train_bce": loss_sum / seen,
                "tune": m,
                "thresholds": t,
                "elapsed_seconds": time.perf_counter() - started,
                "dev_inference_seconds": evalsec,
            }
        )
        if m["macro_f1"] > best + 1e-12:
            best = m["macro_f1"]
            bestpolicy = {
                "variant": name,
                "mode": mode,
                "seed": seed,
                "epoch": epoch,
                "thresholds": t,
                "tune_macro_f1": best,
                "trainable_parameters": nparams,
                "physical_batch": physical_batch,
                "padding_strategy": "fixed512",
                "mps_cache_release_every_optimizer_steps": 25,
                "gradient_accumulation": accum,
                "effective_batch": 16,
                "max_length": 512,
                "encoder_lr": (5e-4 if mode.startswith("lora") else 2e-5),
                "adapter_rank": (8 if mode.startswith("lora") else None),
                "adapter_alpha": (16 if mode.startswith("lora") else None),
                "head_lr": 1e-3,
                "attention_backend": ("eager" if mode == "last2" else "sdpa"),
                "scheduled_optimizer_steps": total,
                "completed_optimizer_steps": global_step,
                "epoch_budget": (1 if mode.startswith("lora") else 3),
                "attention_dropout": (0.1 if mode == "last2" else 0.0),
                "checkpoint": str((V2 / "models" / f"{name}.pt").relative_to(V2)),
            }
            torch.save(
                {k: v.detach().cpu() for k, v in model.state_dict().items()},
                V2 / "models" / f"{name}.pt",
            )
            np.save(V2 / "results" / f"{name}_selected_dev_probs.npy", probs)
        dump(f"{name}_history.json", history)
        dump(f"{name}_selection.json", bestpolicy)
        print(
            "EPOCH",
            name,
            epoch,
            "train_bce",
            round(loss_sum / seen, 5),
            "tune_macro",
            round(m["macro_f1"], 4),
            "BEST",
            round(best, 4),
            flush=True,
        )
    bestpolicy["training_seconds"] = time.perf_counter() - started
    bestpolicy["checkpoint_sha256"] = sha(V2 / bestpolicy["checkpoint"])
    dump(f"{name}_selection.json", bestpolicy)
    del model, opt, scheduler
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    return bestpolicy


if __name__ == "__main__":
    torch.set_num_threads(4)
    S, Y, TUNE, CAL = data()
    YDEV = Y["dev"]
    tok = AutoTokenizer.from_pretrained(ROOT / "models/biomedbert", local_files_only=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("TRAINING DEVICE", device, flush=True)
    trainenc = tokens(S["train"], tok)
    devenc = tokens(S["dev"], tok)
    configs = []
    for mode, seed in [("lora512", 17)]:
        selection = V2 / "results" / f"{mode}_seed{seed}_selection.json"
        if mode == "last2" and selection.exists():
            configs.append(json.loads(selection.read_text()))
            print("REUSE COMPLETED LAST2", flush=True)
        else:
            configs.append(train_one(mode, seed, trainenc, devenc, Y["train"], TUNE, tok, device))
    dump("locked_finetune_policies.json", configs)
    # Both old test and fresh audit are evaluated only after all checkpoint choices are frozen.
    results = []
    for cfg in configs:
        torch.manual_seed(cfg["seed"])
        model = TopicModel(
            attention_backend=("eager" if cfg["mode"] == "last2" else "sdpa"),
            attention_dropout=(0.1 if cfg["mode"] == "last2" else 0.0),
        ).to(device)
        model = attach_lora(model) if cfg["mode"].startswith("lora") else model
        model.load_state_dict(
            torch.load(V2 / cfg["checkpoint"], map_location="cpu", weights_only=True)
        )
        model.eval()
        for split in ["test", "audit"]:
            probs, sec = infer(model, tokens(S[split], tok), tok, device)
            np.save(V2 / "results" / f"{cfg['variant']}_{split}_probs.npy", probs)
            results.append(
                {
                    "variant": cfg["variant"],
                    "split": split,
                    "selected_epoch": cfg["epoch"],
                    "inference_seconds": sec,
                    **metrics(Y[split], probs, cfg["thresholds"]),
                }
            )
        del model
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
    dump("finetune_results.json", results)
    print(
        "FINETUNING COMPLETE",
        [(r["variant"], r["split"], r["macro_f1"]) for r in results],
        flush=True,
    )
