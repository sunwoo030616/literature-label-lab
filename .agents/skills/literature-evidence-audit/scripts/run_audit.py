"""Execute repository audit or a new bounded question; never overwrite locked runs."""
import argparse
import subprocess
import sys
from pathlib import Path

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--question')
    parser.add_argument('--output')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[4]
    if args.question:
        if not args.output: parser.error('--question requires a new --output JSON path')
        output=Path(args.output).resolve()
        if output.exists(): parser.error('output already exists; choose a new file')
        command=[sys.executable,str(root/'extension_v4/src/agent.py'),'--question',args.question,'--output',str(output)]
    else:
        if args.output: parser.error('--output is only used with --question')
        command=[sys.executable,str(root/'extension_v4/src/verify_v4.py')]
    return subprocess.run(command,cwd=root,check=False).returncode

if __name__=='__main__':sys.exit(main())
