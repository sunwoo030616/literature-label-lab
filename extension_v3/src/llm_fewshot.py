"""Training-only in-context examples; same fixed40 development papers, no prompt retries."""

import sys, json, time, hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V3 = ROOT / "extension_v3"
V2 = ROOT / "followup_v2"
sys.path.insert(0, str(V2 / "src"))
from common import data, train_ids, LABELS, metrics, llm_path
from llm_review import SYSTEM, DEFINITIONS, parse
import numpy as np


def dump(name, obj):
    (V3 / "results" / name).write_text(
        json.dumps(
            obj,
            ensure_ascii=False,
            indent=2,
            default=lambda a: a.tolist() if isinstance(a, np.ndarray) else float(a),
        )
    )


if __name__ == "__main__":
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler

    S, Y, TUNE, CAL = data()
    manifest = json.loads((V2 / "results/llm_sample_manifest.json").read_text())
    ix = [r["index"] for r in manifest]
    pool = sorted([S["train"][int(i)] for i in train_ids(17)], key=lambda r: r["sha256"])
    uncovered = set(LABELS)
    examples = []
    while uncovered:
        row = max(pool, key=lambda r: len(uncovered & set(r["labels"])))
        gain = uncovered & set(row["labels"])
        assert gain
        examples.append(row)
        uncovered -= gain
        pool = [r for r in pool if r["pmid"] != row["pmid"]]
    modelpath = llm_path()
    model, tok = load(str(modelpath), tokenizer_config={"trust_remote_code": False})
    example_text = "\n\n".join(
        f"TRAINING EXAMPLE {j+1}:\n"
        + tok.decode(tok.encode(r["text"])[:220])
        + "\nGOLD_TOPICS: "
        + json.dumps(r["labels"])
        for j, r in enumerate(examples)
    )
    dump(
        "llm_training_examples.json",
        [{"pmid": r["pmid"], "sha256": r["sha256"], "labels": r["labels"]} for r in examples],
    )
    out = []
    system = (
        SYSTEM
        + "\nTraining examples demonstrate topic annotation only. Quote evidence ONLY from the CURRENT DOCUMENT, never from training examples. Do not copy example text as evidence."
    )
    for n, i in enumerate(ix):
        row = S["dev"][i]
        visible = tok.decode(tok.encode(row["text"])[:1600])
        user = (
            "TOPIC DEFINITIONS:\n"
            + DEFINITIONS
            + "\nTRAINING EXAMPLES (not current document):\n"
            + example_text
            + "\nEND EXAMPLES\nCURRENT DOCUMENT:\n"
            + visible
            + "\nEND CURRENT DOCUMENT"
        )
        prompt = tok.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        t = time.perf_counter()
        error = None
        try:
            raw = generate(
                model,
                tok,
                prompt=prompt,
                max_tokens=384,
                sampler=make_sampler(temp=0),
                verbose=False,
            )
            parsed = parse(raw, row["text"])
            for item in parsed["items"]:
                item["exact_quote_valid"] = bool(
                    item["exact_quote_valid"] and item["evidence"] in visible
                )
            parsed["verified_topics"] = [
                l
                for l in parsed["verified_topics"]
                if any(e["label"] == l and e["exact_quote_valid"] for e in parsed["items"])
            ]
        except Exception as exc:
            raw = ""
            error = repr(exc)
            parsed = {"valid_json": False, "raw_topics": [], "verified_topics": [], "items": []}
        out.append(
            {
                "index": i,
                "pmid": row["pmid"],
                "gold": row["labels"],
                "latency_seconds": time.perf_counter() - t,
                "prompt_tokens": len(tok.encode(prompt)),
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "raw_output": raw,
                "error": error,
                **parsed,
            }
        )
        dump("llm_fewshot_raw.json", out)
        print("FEWSHOT", n + 1, "/40", round(out[-1]["latency_seconds"], 2), flush=True)
    yy = Y["dev"][ix]
    conditions = []
    for field in ["raw_topics", "verified_topics"]:
        p = np.array([[int(l in r[field]) for l in LABELS] for r in out])
        conditions.append({"variant": "fewshot_" + field, **metrics(yy, p, 0.5)})
    evidence = [e for r in out for e in r["items"]]
    summary = {
        "scope": "Same40 development calibration papers; exploratory, not final test",
        "training_examples": len(examples),
        "selection": "Greedy label coverage, hash-order ties, training seed17 only; no calibration error selection",
        "valid_json_rate": np.mean([r["valid_json"] for r in out]),
        "verbatim_evidence_rate": (
            np.mean([e["exact_quote_valid"] for e in evidence]) if evidence else 0
        ),
        "median_seconds": np.median([r["latency_seconds"] for r in out]),
        "p95_seconds": np.quantile([r["latency_seconds"] for r in out], 0.95),
        "conditions": conditions,
        "note": "In-context examples do not update Qwen model weights. No prompt tuning/retry after seeing40 outputs. Example text and current input are token-truncated; quotation occurrence is not entailment.",
    }
    dump("llm_fewshot_summary.json", summary)
    print("FEWSHOT COMPLETE", [(r["variant"], r["macro_f1"]) for r in conditions], flush=True)
