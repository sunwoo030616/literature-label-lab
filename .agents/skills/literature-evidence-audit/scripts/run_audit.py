"""Execute repository audit or a new bounded question; never overwrite locked runs."""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question")
    parser.add_argument("--output")
    parser.add_argument("--preserve-identifiers", action="store_true")
    parser.add_argument("--repair-edge-ellipsis", action="store_true")
    parser.add_argument("--require-numeric-anchors", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[4]
    if args.question:
        if not args.output:
            parser.error("--question requires a new --output JSON path")
        output = Path(args.output).resolve()
        if output.exists():
            parser.error("output already exists; choose a new file")
        command = [
            sys.executable,
            str(root / "extension_v4/src/agent.py"),
            "--question",
            args.question,
            "--output",
            str(output),
        ]
        if args.preserve_identifiers:
            command.append("--preserve-identifiers")
        if args.repair_edge_ellipsis:
            command.append("--repair-edge-ellipsis")
        if args.require_numeric_anchors:
            command.append("--require-numeric-anchors")
    else:
        if (
            args.output
            or args.preserve_identifiers
            or args.repair_edge_ellipsis
            or args.require_numeric_anchors
        ):
            parser.error("output and quality options require --question")
        command = [sys.executable, str(root / "extension_v4/src/verify_v4.py")]
    return subprocess.run(command, cwd=root, check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
