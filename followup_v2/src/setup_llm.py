"""Optional explicit download of the pinned free MLX model for a fresh clone."""

import argparse, hashlib, json
from pathlib import Path
from huggingface_hub import snapshot_download

REPO = "mlx-community/Qwen3-4B-4bit"
REVISION = "4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25"
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download approximately2.3GB of public model weights; no paid inference API."
    )
    parser.add_argument("--destination", required=True)
    args = parser.parse_args()
    dest = Path(args.destination).expanduser().resolve()
    snapshot_download(
        REPO,
        revision=REVISION,
        local_dir=dest,
        allow_patterns=["*.json", "*.txt", "*.safetensors", "*.model"],
    )
    files = {}
    for p in sorted(dest.iterdir()):
        if p.is_file() and p.name != "manifest.local.json":
            files[p.name] = {
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                "bytes": p.stat().st_size,
            }
    (dest / "manifest.local.json").write_text(
        json.dumps({"repo_id": REPO, "revision": REVISION, "files": files}, indent=2)
    )
    print("Ready. Set LITERATURE_LLM_PATH to", dest)
