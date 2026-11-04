"""Descriptive post-test analysis only; never used to select models."""
from pathlib import Path
import json,csv,numpy as np
ROOT=Path(__file__).resolve().parents[1]
def main():
 rr=ROOT/'results';raw={r['pmid']:r for r in csv.DictReader((ROOT/'data/raw/test.csv').open())};ids=json.loads((rr/'split_manifest.json').read_text())['test'];labels=['Treatment','Diagnosis','Prevention','Mechanism','Transmission','Epidemic Forecasting','Case Report'];y=np.array([[int(l in raw[r['pmid']]['label'].split(';')) for l in labels] for r in ids]);cfg=json.loads((rr/'locked_policies.json').read_text());pred={}
 for name in ['biomedbert','relation']:
  t=next(c['thresholds'] for c in cfg if c['seed']==17 and c['method']==name);pred[name]=np.load(rr/f'probabilities_{name}.npy')>=np.array(t)
 b=pred['biomedbert'];c=pred['relation'];correct_b=(b==y).all(1);correct_c=(c==y).all(1)
 summary={'exact_match_fixed_by_relation':int((~correct_b&correct_c).sum()),'exact_match_broken_by_relation':int((correct_b&~correct_c).sum()),'documents_with_changed_labels':int(np.any(b!=c,axis=1).sum()),'per_label_counts':{}}
 for j,l in enumerate(labels):
  summary['per_label_counts'][l]={n:{'tp':int((pp[:,j]&y[:,j].astype(bool)).sum()),'fp':int((pp[:,j]&~y[:,j].astype(bool)).sum()),'fn':int((~pp[:,j]&y[:,j].astype(bool)).sum())} for n,pp in pred.items()}
 (rr/'error_transitions.json').write_text(json.dumps(summary,indent=2));print('Descriptive error transitions saved.')
if __name__=='__main__':main()
