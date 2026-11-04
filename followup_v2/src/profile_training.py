import time, json, torch
from transformers import AutoTokenizer
from common import ROOT, V2, data, train_ids
from finetune import TopicModel

S, Y, _, _ = data()
tok = AutoTokenizer.from_pretrained(ROOT / "models/biomedbert", local_files_only=True)
torch.set_num_threads(4)
device = "mps"
model = TopicModel().to(device).train()
opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
out = []
for batch in [2, 2, 2]:
    x = tok(
        [r["text"] for r in S["train"][:batch]],
        max_length=256,
        truncation=True,
        padding=True,
        return_tensors="pt",
    ).to(device)
    y = torch.tensor(Y["train"][:batch], dtype=torch.float32, device=device)
    torch.mps.synchronize()
    t = time.perf_counter()
    z = model(**x)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
    torch.mps.synchronize()
    f = time.perf_counter()
    loss.backward()
    torch.mps.synchronize()
    b = time.perf_counter()
    opt.step()
    opt.zero_grad(set_to_none=True)
    torch.mps.synchronize()
    e = time.perf_counter()
    row = {
        "batch": batch,
        "forward": f - t,
        "backward": b - f,
        "optimizer": e - b,
        "total": e - t,
        "allocated_mb": torch.mps.current_allocated_memory() / 2**20,
        "driver_mb": torch.mps.driver_allocated_memory() / 2**20,
    }
    out.append(row)
    print(row, flush=True)
(V2 / "results/training_profile.json").write_text(json.dumps(out, indent=2))
