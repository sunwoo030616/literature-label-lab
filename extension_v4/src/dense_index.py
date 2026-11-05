"""Pinned free MiniLM, card-compatible masked-mean embeddings (256 tokens)."""
import argparse
import json
import sqlite3
import time

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from settings import DB, V4, dump, sha

class Encoder:
    def __init__(self, device=None):
        path = V4/'models/minilm'
        self.device = device or ('mps' if torch.backends.mps.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = AutoModel.from_pretrained(path, local_files_only=True).eval().to(self.device)

    def encode(self, texts, batch_size=32):
        output=[]
        with torch.inference_mode():
            for start in range(0,len(texts),batch_size):
                inputs=self.tokenizer(texts[start:start+batch_size],return_tensors='pt',truncation=True,max_length=256,padding='max_length').to(self.device)
                hidden=self.model(**inputs).last_hidden_state
                mask=inputs['attention_mask'].unsqueeze(-1)
                pooled=(hidden*mask).sum(1)/mask.sum(1).clamp(min=1)
                output.append(torch.nn.functional.normalize(pooled,p=2,dim=1).cpu().numpy())
                if self.device=='mps' and (start//batch_size+1)%20==0:
                    torch.mps.synchronize(); torch.mps.empty_cache()
        return np.concatenate(output).astype(np.float32) if output else np.empty((0,384),dtype=np.float32)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--device'); args=parser.parse_args()
    c=sqlite3.connect(DB)
    rows=c.execute('SELECT pmid,title,abstract,content_hash FROM documents ORDER BY pmid').fetchall()
    encoder=Encoder(args.device)
    matrix=np.lib.format.open_memmap(V4/'data/minilm.npy',mode='w+',dtype=np.float32,shape=(len(rows),384))
    start=time.perf_counter()
    for offset in range(0,len(rows),512):
        batch=rows[offset:offset+512]
        matrix[offset:offset+len(batch)]=encoder.encode([r[1]+' '+r[2] for r in batch]); matrix.flush()
        print(json.dumps(dict(encoded=min(offset+len(batch),len(rows)),total=len(rows),seconds=round(time.perf_counter()-start,2))),flush=True)
    dump(V4/'data/dense_manifest.json',[dict(pmid=r[0],content_hash=r[3]) for r in rows])
    dump(V4/'results/dense_index.json',dict(model=json.loads((V4/'models/minilm/source.json').read_text()),device=encoder.device,documents=len(rows),dimensions=384,max_length=256,seconds=time.perf_counter()-start,embedding_sha256=sha(V4/'data/minilm.npy'),manifest_sha256=sha(V4/'data/dense_manifest.json'),model_files={p.name:sha(p) for p in sorted((V4/'models/minilm').glob('*')) if p.is_file()}))

if __name__=='__main__': main()
