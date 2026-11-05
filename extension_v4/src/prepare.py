"""Optional public downloads with pinned identities; never calls a paid API."""
import argparse
import json
from pathlib import Path
from urllib.request import urlretrieve
from settings import ROOT,V4,dump,sha

MINILM_REV='1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
QWEN_REV='4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--qwen',action='store_true',help='Also download the ~2.3 GB free local Qwen model');args=parser.parse_args()
    recorded=json.loads((V4/'results/ingestion.json').read_text())['first_pass']
    expected={r['source']:r['source_sha256'] for r in recorded}
    base='https://ftp.ncbi.nlm.nih.gov/pub/lu/LitCovid/biocreative/'
    for split,suffix in [('train','Train'),('dev','Dev'),('test','Test-GS')]:
        path=ROOT/'data/raw'/f'{split}.csv';path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists():urlretrieve(base+f'BC7-LitCovid-{suffix}.csv',path)
        if sha(path)!=expected[path.name]:raise ValueError(f'Official source differs from recorded version: {path.name}')
    from huggingface_hub import snapshot_download
    path=V4/'models/minilm'
    snapshot_download('sentence-transformers/all-MiniLM-L6-v2',revision=MINILM_REV,local_dir=path,allow_patterns=['config.json','model.safetensors','tokenizer.json','tokenizer_config.json','special_tokens_map.json','vocab.txt','1_Pooling/config.json','sentence_bert_config.json','modules.json'])
    dump(path/'source.json',dict(repo_id='sentence-transformers/all-MiniLM-L6-v2',revision=MINILM_REV))
    if args.qwen:
        path=V4/'models/qwen'
        snapshot_download('mlx-community/Qwen3-4B-4bit',revision=QWEN_REV,local_dir=path,allow_patterns=['*.json','*.txt','*.safetensors'])
        dump(path/'manifest.local.json',dict(repo_id='mlx-community/Qwen3-4B-4bit',revision=QWEN_REV))
        print('Set LITERATURE_LLM_PATH to',path)
    print('Public inputs ready. No inference or evaluation results were overwritten.')

if __name__=='__main__':main()
