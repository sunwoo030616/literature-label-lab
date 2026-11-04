"""Optional local generative evidence; never overwrites classifier or human labels."""

import sys, time, json, hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "followup_v2/src"))
from llm_review import SYSTEM, DEFINITIONS, parse
from common import llm_path

_CACHE = {}


def generate_evidence(title, abstract):
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler

    path = llm_path()
    if "model" not in _CACHE:
        model, tok = load(str(path), tokenizer_config={"trust_remote_code": False})
        _CACHE.update(
            model=model, tok=tok, identity=json.loads((path / "manifest.local.json").read_text())
        )
    model = _CACHE["model"]
    tok = _CACHE["tok"]
    original = title + " " + abstract
    visible = tok.decode(tok.encode(original)[:1600])
    user = (
        "Allowed topic definitions:\n" + DEFINITIONS + "\nDOCUMENT:\n" + visible + "\nEND DOCUMENT"
    )
    prompt = tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    start = time.perf_counter()
    raw = generate(
        model, tok, prompt=prompt, max_tokens=384, sampler=make_sampler(temp=0), verbose=False
    )
    result = parse(raw, original)
    for item in result["items"]:
        item["exact_quote_valid"] = bool(item["exact_quote_valid"] and item["evidence"] in visible)
    result["verified_topics"] = [
        l
        for l in result["verified_topics"]
        if any(i["label"] == l and i["exact_quote_valid"] for i in result["items"])
    ]
    return {
        "model": _CACHE["identity"]["repo_id"],
        "revision": _CACHE["identity"]["revision"],
        "raw_output": raw,
        "latency_seconds": time.perf_counter() - start,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "status": "parsed" if result["valid_json"] else "invalid_output",
        **result,
        "limitations": [
            "Quote occurrence is not semantic entailment.",
            "Generated topics do not replace classifier predictions or reviewed labels.",
        ],
    }
