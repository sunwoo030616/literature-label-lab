"""Download only official public dataset and MIT model. No paid API."""
from pathlib import Path
from urllib.request import urlretrieve
from huggingface_hub import snapshot_download
ROOT=Path(__file__).resolve().parents[1]
base='https://ftp.ncbi.nlm.nih.gov/pub/lu/LitCovid/biocreative/'
for split,suffix in [('train','Train'),('dev','Dev'),('test','Test-GS')]:
 p=ROOT/'data/raw'/f'{split}.csv';p.parent.mkdir(parents=True,exist_ok=True)
 if not p.exists():urlretrieve(base+f'BC7-LitCovid-{suffix}.csv',p)
snapshot_download('microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract',local_dir=ROOT/'models/biomedbert',revision='d673b8835373c6fa116d6d8006b33d48734e305d',allow_patterns=['config.json','pytorch_model.bin','vocab.txt','tokenizer_config.json','special_tokens_map.json'])
