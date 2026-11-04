"""Streaming verification of the reused pinned local generative-model snapshot."""

import hashlib, json
from common import llm_path, V2

if __name__ == "__main__":
    path = llm_path()
    manifest = json.loads((path / "manifest.local.json").read_text())
    assert manifest["repo_id"] == "mlx-community/Qwen3-4B-4bit"
    assert manifest["revision"] == "4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25"
    total = 0
    for name, expected in manifest["files"].items():
        p = path / name
        assert p.parent == path and p.is_file()
        h = hashlib.sha256()
        with p.open("rb") as f:
            while chunk := f.read(8 * 1024 * 1024):
                h.update(chunk)
        assert h.hexdigest() == expected["sha256"], name
        assert p.stat().st_size == expected["bytes"], name
        total += expected["bytes"]
    report = {
        "repo_id": manifest["repo_id"],
        "revision": manifest["revision"],
        "files_verified": len(manifest["files"]),
        "bytes_verified": total,
        "all_sha256_match": True,
    }
    (V2 / "results/llm_snapshot_verification.json").write_text(json.dumps(report, indent=2))
    print(report)
