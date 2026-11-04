import sys, os, json, hashlib, csv
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
V2 = ROOT / "followup_v2"
sys.path.insert(0, str(ROOT / "src"))
sys.path.append(str(ROOT / ".deps"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/literature-v2-mpl")
from experiment import LABELS, yof, thresholds, metrics, row_f1


def dump(name, obj):
    p = V2 / "results" / name
    p.parent.mkdir(exist_ok=True, parents=True)
    p.write_text(
        json.dumps(
            obj,
            ensure_ascii=False,
            indent=2,
            default=lambda a: a.tolist() if isinstance(a, np.ndarray) else float(a),
        )
    )


def read(name):
    return json.loads((V2 / "results" / name).read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def data():
    # Ignored artifact directories are absent in a fresh Git clone.
    for home in [V2, ROOT / "extension_v3"]:
        for directory in ["data", "models", "results", "docs"]:
            (home / directory).mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / "results/split_manifest.json").read_text())
    splits = {}
    for k in ["train", "dev", "test"]:
        raw = {r["pmid"]: r for r in csv.DictReader((ROOT / "data/raw" / f"{k}.csv").open())}
        rows = []
        for m in manifest[k]:
            r = raw[m["pmid"]]
            text = r["title"].strip() + " " + r["abstract"].strip()
            assert (
                hashlib.sha256(" ".join(text.lower().split()).encode()).hexdigest() == m["sha256"]
            )
            rows.append(
                {
                    "pmid": r["pmid"],
                    "title": r["title"],
                    "text": text,
                    "labels": r["label"].split(";"),
                    "sha256": m["sha256"],
                }
            )
        splits[k] = rows
    audit_path = V2 / "data/audit_manifest.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
    else:
        seenid = {r["pmid"] for rows in splits.values() for r in rows}
        seenhash = {r["sha256"] for rows in splits.values() for r in rows}
        pool = []
        for r in csv.DictReader((ROOT / "data/raw/train.csv").open()):
            text = r["title"].strip() + " " + r["abstract"].strip()
            h = hashlib.sha256(" ".join(text.lower().split()).encode()).hexdigest()
            if not r["abstract"].strip() or r["pmid"] in seenid or h in seenhash:
                continue
            seenid.add(r["pmid"])
            seenhash.add(h)
            pool.append(
                {
                    "pmid": r["pmid"],
                    "title": r["title"],
                    "text": text,
                    "labels": r["label"].split(";"),
                    "sha256": h,
                }
            )
        audit = sorted(pool, key=lambda r: r["sha256"])[:1000]
        audit_path.write_text(json.dumps(audit, ensure_ascii=False))
    splits["audit"] = audit
    ys = {k: yof(v) for k, v in splits.items()}
    order = np.random.default_rng(20261104).permutation(1000)
    return splits, ys, order[:600], order[600:]


def train_ids(seed):
    return np.sort(np.random.default_rng(seed).choice(5000, 4000, replace=False))


def frozen_features():
    e = np.load(ROOT / "data/embeddings.npz")["embeddings"]
    return {"train": e[:5000], "dev": e[5000:6000], "test": e[6000:]}


def llm_path():
    """Explicit override for reproducing outside the original user's workspace."""
    from pathlib import Path

    default = ROOT.parent / "kkbb-ddook-ddak-back/var/research_models/canvas_generator"
    return Path(os.environ.get("LITERATURE_LLM_PATH", str(default))).expanduser()
