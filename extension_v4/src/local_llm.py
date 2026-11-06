"""Lazy local-only Qwen generation with immutable prompt identities."""

import hashlib
import json
import os
import time
from pathlib import Path
from settings import ROOT, V4


class LocalLLM:
    def __init__(self, registry_path=V4 / "prompts/registry.json"):
        from mlx_lm import load

        path = Path(
            os.environ.get(
                "LITERATURE_LLM_PATH",
                str(
                    ROOT.parent
                    / "kkbb-ddook-ddak-back/var/research_models/canvas_generator"
                ),
            )
        )
        self.identity = json.loads((path / "manifest.local.json").read_text())
        self.model, self.tokenizer = load(
            str(path), tokenizer_config={"trust_remote_code": False}
        )
        self.registry = json.loads(Path(registry_path).read_text(encoding="utf-8"))

    def generate(self, version, payload):
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        spec = self.registry[version]
        prompt = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": spec["system"]},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        start = time.perf_counter()
        raw = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=spec["max_tokens"],
            sampler=make_sampler(temp=0),
            verbose=False,
        )
        return dict(
            raw_output=raw,
            seconds=time.perf_counter() - start,
            prompt_version=version,
            prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
            input_tokens=len(self.tokenizer.encode(prompt)),
            max_output_tokens=spec["max_tokens"],
            model=self.identity["repo_id"],
            revision=self.identity["revision"],
        )
