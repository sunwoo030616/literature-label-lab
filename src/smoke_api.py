"""Actual encoder/API smoke, input validation, controlled busy response."""
import json,csv,time
from fastapi.testclient import TestClient
from api import app,lock
from experiment import ROOT
rows=list(csv.DictReader((ROOT/'data/raw/test.csv').open()))
checks=[]
with TestClient(app) as client:
 r=client.get('/health');assert r.status_code==200 and r.json()['ready'];checks.append({'check':'health','status':r.status_code})
 r=client.post('/classify',json={'title':'','abstract':'short'});assert r.status_code==422;checks.append({'check':'invalid_input','status':r.status_code})
 req={'title':rows[0]['title'],'abstract':rows[0]['abstract'],'top_k':3}
 lock.acquire()
 try:
  r=client.post('/classify',json=req);assert r.status_code==429;checks.append({'check':'controlled_busy','status':r.status_code})
 finally:lock.release()
 for ix in [0,4,9]:
  row=rows[ix];r=client.post('/classify',json={'title':row['title'],'abstract':row['abstract'],'top_k':3});assert r.status_code==200
  data=r.json();assert len(data['neighbors'])==3 and len(data['scores'])==7
  assert all(0<=v<=1 for v in data['scores'].values())
  checks.append({'check':'actual_inference','pmid':row['pmid'],'status':r.status_code,'gold':row['label'].split(';'),'response':data})
(ROOT/'results/api_smoke.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2));print('API smoke passed',len(checks))
