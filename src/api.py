"""Local inference and literature-neighbor API; no external paid calls."""
from contextlib import asynccontextmanager
import json, threading, time
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from experiment import ROOT,get_encoder,encode,corrected,confidence
state={};lock=threading.Lock()
@asynccontextmanager
async def lifespan(app):
 state['bundle']=joblib.load(ROOT/'models/classifiers.joblib')
 state['encoder']=get_encoder()
 state['index']=np.load(ROOT/'data/search_index.npz')['embeddings']
 state['index']=state['index']/np.maximum(np.linalg.norm(state['index'],axis=1,keepdims=True),1e-9)
 state['documents']=json.loads((ROOT/'results/search_documents.json').read_text())
 yield
app=FastAPI(title='Biomedical Literature Label Lab',lifespan=lifespan)
class Request(BaseModel):
 title:str=Field(min_length=1,max_length=1000)
 abstract:str=Field(min_length=20,max_length=20000)
 top_k:int=Field(default=3,ge=1,le=10)
@app.get('/health')
def health():return {'ready':bool(state),'model':'frozen BiomedBERT','paid_api':False}
@app.post('/classify')
def classify(req:Request):
 if not lock.acquire(blocking=False):raise HTTPException(429,'Local encoder busy; retry later')
 start=time.perf_counter()
 try:
  b=state['bundle'];tok,model,device=state['encoder'];x=encode([req.title+' '+req.abstract],tok,model,device,batch=1)
  p=b['classifier'].predict_proba(b['scaler'].transform(x));cfg=next(c for c in b['policies'] if c['method']=='relation')
  q=corrected(p,b['relation'],b['prior'],cfg['alpha']);t=np.array(cfg['thresholds']);pred=q[0]>=t
  review=confidence(q,t)[0]<cfg['review_policy']['selected']['margin']
  v=x[0]/max(np.linalg.norm(x[0]),1e-9);scores=state['index']@v;ix=np.argsort(scores)[-req.top_k:][::-1]
  return {'predicted_topics':[l for l,a in zip(b['labels'],pred) if a],'scores':dict(zip(b['labels'],q[0].tolist())),'review_recommended':bool(review),'review_policy_calibration_target_met':cfg['review_policy']['target_met'],'relation_strength':cfg['alpha'],'neighbors':[dict(state['documents'][int(i)],cosine_similarity=float(scores[i]),annotation_source='official training annotations') for i in ix],'elapsed_seconds':time.perf_counter()-start,'limitations':['Scores are not calibrated probabilities.','Neighbors are similarities, not evidence explaining predictions.','Trained for COVID-19 literature; unrelated text may receive incorrect labels.','Input is truncated to 256 tokens.','Human review is recommended; this is not medical guidance.']}
 finally:lock.release()
