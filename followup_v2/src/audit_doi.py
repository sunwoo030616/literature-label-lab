"""Additional descriptive duplicate audit; does not alter locked splits."""
from common import ROOT,V2,data,dump
import csv,re
S,Y,_,_=data();byid={}
for split in ['train','dev','test']:
 for r in csv.DictReader((ROOT/'data/raw'/f'{split}.csv').open()):
  doi=r.get('doi','').strip().lower();doi=re.sub(r'^https?://(?:dx\.)?doi\.org/','',doi);byid[r['pmid']]=doi
maps={k:{} for k in S}
for k,rows in S.items():
 for r in rows:
  doi=byid.get(r['pmid'],'')
  if doi:maps[k].setdefault(doi,[]).append(r['pmid'])
collisions={}
for a,b in [('train','dev'),('train','test'),('train','audit'),('dev','test'),('dev','audit'),('test','audit')]:
 overlap=set(maps[a])&set(maps[b]);collisions[f'{a}_{b}']=[{'doi':d,a:maps[a][d],b:maps[b][d]} for d in sorted(overlap)]
dump('doi_audit.json',{'cross_split_collisions':collisions,'note':'DOI not present for all articles; absence of exact DOI matches does not exclude near-duplicate content.'});print({k:len(v) for k,v in collisions.items()})
