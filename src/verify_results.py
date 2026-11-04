"""Validate evaluation artifacts and split integrity after execution."""
import json
from experiment import ROOT
manifest=json.loads((ROOT/'results/split_manifest.json').read_text())
for field in ['pmid','sha256']:
 sets={k:{r[field] for r in v} for k,v in manifest.items()}
 assert all(len(sets[k])==len(manifest[k]) for k in sets)
 assert not sets['train']&sets['dev'] and not sets['train']&sets['test'] and not sets['dev']&sets['test']
results=json.loads((ROOT/'results/test_results.json').read_text());assert len(results)==24
assert {(r['seed'],r['method']) for r in results}=={(s,m) for s in [17,42,73] for m in ['tfidf','biomedbert','relation']+[f'shuffled_{i}' for i in range(5)]}
for r in results:
 assert all(0<=r[k]<=1 for k in ['macro_f1','micro_f1','hamming_loss','subset_accuracy','multi_topic_recall'])
 assert sum(v['support'] for v in r['per_label'].values())>len(manifest['test'])
assert (ROOT/'results/locked_policies.json').stat().st_mtime<=(ROOT/'results/test_results.json').stat().st_mtime
print('PASS: split IDs/hashes disjoint, 24 complete conditions, valid metrics, policies saved before final results.')

graph=json.loads((ROOT/'results/graph_test_results.json').read_text());assert len(graph)==6
assert {(r['seed'],r['method']) for r in graph}=={(s,m) for s in [17,42,73] for m in ['feature_graph','random_neighbors']}
assert all(0<=r['macro_f1']<=1 and 0<=r['micro_f1']<=1 for r in graph)
assert (ROOT/'results/graph_locked_policies.json').stat().st_mtime<=(ROOT/'results/graph_test_results.json').stat().st_mtime
print('PASS: 6 graph conditions complete; graph policies precede test results.')
